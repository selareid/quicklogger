use chrono::{Datelike, LocalResult, TimeZone, Utc};
use reqwest::{blocking::Client, header::RETRY_AFTER, StatusCode};
use serde_json::{json, Value};
use std::{cmp::min, collections::HashSet, env, error::Error, fs, io::{self, BufRead, Write}, path::{Component, Path, PathBuf}, sync::mpsc::{self, Receiver, TryRecvError}, thread, time::{Duration, SystemTime}};

const DEFAULT_LOGS: &str = "./logs";
const DEFAULT_EXTRA_DATA: &str = "./extra_data";
const DEFAULT_MODEL: &str = "gpt-5.4-mini";
const SERVICE_TIER: &str = "flex";
const TIMEOUT_SECS: u64 = 900;
const DEFAULT_MAX_STEPS: usize = 30;
const LOG_DIR: &str = "./llm_logs";
const ENTRY_CHARS: usize = 500;
const RESULT_LIMIT: usize = 10;
const MAX_RESULT_LIMIT: usize = 25;
const EXTRA_SNIPPET_CHARS: usize = 800;
const EXTRA_READ_CHARS: usize = 4_000;
const MAX_EXTRA_READ_CHARS: usize = 12_000;
const COMPACT_AFTER_CHARS: usize = 18_000;
const KEEP_RECENT_ITEMS: usize = 8;
const SUMMARY_CHARS: usize = 6_000;
const RETRIES: usize = 8;
const MAX_BACKOFF: u64 = 180;

type AppResult<T> = Result<T, Box<dyn Error>>;

fn main() -> AppResult<()> {
    let cfg = Config::from_args()?;
    let mut input = UserInput::spawn();
    let mut log = RunLog::new(cfg.verbose)?;
    log.line(format!("run log path: {}", log.path.display()));
    log.line(format!("session snapshot path: {}", log.session_path.display()));
    log.line(format!("logs path: {}", cfg.logs_path.display()));
    log.line(format!("extra data path: {}", cfg.extra_data_path.display()));
    log.line(format!("model: {}", cfg.model));
    log.line(format!("service tier: {SERVICE_TIER}"));
    log.line(format!("HTTP timeout: {TIMEOUT_SECS}s"));
    log.line(format!("history compaction: after ~{COMPACT_AFTER_CHARS} chars, keep last {KEEP_RECENT_ITEMS} items"));

    let api_key = env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY must be set")?;
    let entries = load_entries(&cfg.logs_path)?;
    if entries.is_empty() { return Err(format!("No log entries found under {}", cfg.logs_path.display()).into()); }
    log.line(format!("loaded {} log entries", entries.len()));
    let extra_files = load_extra_files(&cfg.extra_data_path)?;
    log.line(format!("loaded {} UTF-8 extra_data file(s)", extra_files.len()));

    let client = OpenAiClient::new(api_key)?;
    let resume_path = cfg.resume_path()?;
    let resumed = resume_path.is_some();
    let mut h = if let Some(path) = resume_path.as_deref() {
        Harness::resume(entries, extra_files, log, path)?
    } else {
        let mut h = Harness::new(entries, extra_files, log);
        let goal = cfg.goal.as_deref().ok_or("Pass a goal with --goal \"...\" or use --resume/--resume-latest")?;
        h.push_user(format!("Goal: {goal}\n\nUse log_action to inspect QuickLogger logs and extra_data files. Keep notes concise. Prefer small targeted pages. If the goal is unclear, use action=wait. When done, use action=finish."));
        h
    };
    if resumed {
        if let Some(goal) = cfg.goal.as_deref() { h.add_user_messages("Resume user input", vec![goal.to_string()]); }
        h.save_session()?;
    }

    if !run_loop(&mut h, &client, &mut input, &cfg.model, cfg.max_steps, "initial run") { return Ok(()); }
    loop {
        h.log("waiting for follow-up input; type a message, /continue, or /quit");
        let p = parse_controls(input.wait_for_messages());
        if p.quit || p.empty() { break; }
        if !p.messages.is_empty() { h.add_user_messages("Follow-up", p.messages); h.save_session().ok(); }
        if !run_loop(&mut h, &client, &mut input, &cfg.model, cfg.max_steps, "follow-up run") { break; }
    }
    Ok(())
}

fn run_loop(h: &mut Harness, client: &OpenAiClient, input: &mut UserInput, model: &str, max_steps: usize, label: &str) -> bool {
    loop {
        match h.run(client, input, model, max_steps) {
            Ok(answer) => { h.log(format!("{label} finished")); h.save_session().ok(); println!("\n=== Answer ===\n{answer}\n\n=== Notes ===\n{}\n\n=== LLM Run Log ===\n{}\n\n=== Session Snapshot ===\n{}", h.notes.trim(), h.log.path.display(), h.log.session_path.display()); return true; }
            Err(e) => {
                h.log(format!("{label} paused after error: {e}"));
                h.save_session().ok();
                eprintln!("\n=== Run paused after error ===\n{e}\n\nLLM run log: {}\nSession snapshot: {}\n\nType /continue to retry, add context to retry with it, or /quit to exit.", h.log.path.display(), h.log.session_path.display());
                let p = parse_controls(input.wait_for_messages());
                if p.quit || p.empty() { return false; }
                if !p.messages.is_empty() { h.add_user_messages("User input after error", p.messages); h.save_session().ok(); }
                h.log("retrying from current compacted state");
            }
        }
    }
}

struct Config { goal: Option<String>, logs_path: PathBuf, extra_data_path: PathBuf, model: String, max_steps: usize, verbose: bool, resume: Option<PathBuf>, resume_latest: bool }
impl Config {
    fn from_args() -> AppResult<Self> {
        let mut goal = None; let mut logs_path = env::var("QUICKLOGGER_LOGS_PATH").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_LOGS)); let mut extra_data_path = env::var("QUICKLOGGER_EXTRA_DATA_PATH").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_EXTRA_DATA)); let mut model = env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()); let mut max_steps = DEFAULT_MAX_STEPS; let mut verbose = false; let mut resume = None; let mut resume_latest = false; let mut pos = Vec::new();
        let mut args = env::args().skip(1).peekable();
        while let Some(a) = args.next() { match a.as_str() { "--goal"|"-g" => goal = Some(next_arg(&mut args, "--goal")?), "--logs"|"-l" => logs_path = PathBuf::from(next_arg(&mut args, "--logs")?), "--extra-data" => extra_data_path = PathBuf::from(next_arg(&mut args, "--extra-data")?), "--model"|"-m" => model = next_arg(&mut args, "--model")?, "--max-steps" => max_steps = next_arg(&mut args, "--max-steps")?.parse()?, "--resume" => resume = Some(PathBuf::from(next_arg(&mut args, "--resume")?)), "--resume-latest" => resume_latest = true, "--verbose"|"-v" => verbose = true, "--help"|"-h" => { print_help(); std::process::exit(0); }, other => pos.push(other.to_string()) } }
        if goal.is_none() && !pos.is_empty() { goal = Some(pos.join(" ")); }
        if resume.is_some() && resume_latest { return Err("Use only one of --resume or --resume-latest".into()); }
        if goal.is_none() && resume.is_none() && !resume_latest { return Err("Pass a goal with --goal \"...\" or use --resume/--resume-latest".into()); }
        Ok(Self { goal, logs_path, extra_data_path, model, max_steps, verbose, resume, resume_latest })
    }
    fn resume_path(&self) -> AppResult<Option<PathBuf>> { if let Some(path) = &self.resume { Ok(Some(path.clone())) } else if self.resume_latest { Ok(Some(find_latest_session()?)) } else { Ok(None) } }
}
fn next_arg(args: &mut std::iter::Peekable<impl Iterator<Item = String>>, flag: &str) -> AppResult<String> { args.next().ok_or_else(|| format!("{flag} requires a value").into()) }
fn print_help() { eprintln!("cargo run --bin log_llm -- --goal \"summarise my logs\" [--logs ./logs] [--extra-data ./extra_data] [--model gpt-5.4-mini] [--verbose]\n\nResume:\n  cargo run --bin log_llm -- --resume ./llm_logs/<run>.session.json\n  cargo run --bin log_llm -- --resume-latest\n  cargo run --bin log_llm -- --resume-latest --goal \"call finish with what you've got\""); }
fn find_latest_session() -> AppResult<PathBuf> { let mut best: Option<(SystemTime, PathBuf)> = None; for e in fs::read_dir(LOG_DIR)? { let e = e?; let path = e.path(); if !path.is_file() { continue; } let name = path.file_name().and_then(|s| s.to_str()).unwrap_or(""); if !name.ends_with(".session.json") { continue; } let modified = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH); if best.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) { best = Some((modified, path)); } } best.map(|(_, p)| p).ok_or_else(|| format!("No .session.json files found in {LOG_DIR}").into()) }

struct UserInput { rx: Receiver<String>, closed: bool }
impl UserInput {
    fn spawn() -> Self { let (tx, rx) = mpsc::channel(); thread::spawn(move || { for line in io::stdin().lock().lines() { match line { Ok(line) => if tx.send(line).is_err() { break; }, Err(_) => break } } }); Self { rx, closed: false } }
    fn drain_pending(&mut self) -> Vec<String> { let mut out = Vec::new(); loop { match self.rx.try_recv() { Ok(s) => if let Some(s) = norm(&s) { out.push(s); }, Err(TryRecvError::Empty) => break, Err(TryRecvError::Disconnected) => { self.closed = true; break; } } } out }
    fn wait_for_messages(&mut self) -> Vec<String> { let mut out = self.drain_pending(); if !out.is_empty() || self.closed { return out; } loop { match self.rx.recv() { Ok(s) => if let Some(s) = norm(&s) { out.push(s); out.extend(self.drain_pending()); return out; }, Err(_) => { self.closed = true; return out; } } } }
}
#[derive(Default)] struct Parsed { quit: bool, continue_requested: bool, messages: Vec<String> }
impl Parsed { fn empty(&self) -> bool { !self.quit && !self.continue_requested && self.messages.is_empty() } }
fn parse_controls(messages: Vec<String>) -> Parsed { let mut p = Parsed::default(); for m in messages { let t = m.trim(); let l = t.to_lowercase(); if matches!(l.as_str(), "/quit"|"/exit"|":q"|"quit"|"exit") { p.quit = true; } else if l == "/continue" || l.starts_with("/continue ") { p.continue_requested = true; let rest = t["/continue".len()..].trim(); if !rest.is_empty() { p.messages.push(rest.to_string()); } } else if !t.is_empty() { p.messages.push(t.to_string()); } } p }
fn norm(s: &str) -> Option<String> { let t = s.trim(); (!t.is_empty()).then(|| t.to_string()) }

struct RunLog { path: PathBuf, session_path: PathBuf, file: fs::File, verbose: bool }
impl RunLog {
    fn new(verbose: bool) -> AppResult<Self> { fs::create_dir_all(LOG_DIR)?; let path = Path::new(LOG_DIR).join(format!("{}_{}.log", Utc::now().format("%Y%m%dT%H%M%SZ"), std::process::id())); let session_path = path.with_extension("session.json"); let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?; writeln!(file, "# QuickLogger compact LLM run\n---")?; Ok(Self { path, session_path, file, verbose }) }
    fn line(&mut self, msg: impl AsRef<str>) { let msg = msg.as_ref(); eprintln!("[log-llm] {msg}"); let _ = writeln!(self.file, "[{}] {msg}", Utc::now().to_rfc3339()); let _ = self.file.flush(); }
    fn value(&mut self, label: &str, v: &Value) { let text = serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()); if self.verbose && matches!(label, "openai_error"|"tool_result") { eprintln!("[log-llm] {label}: {}", trunc(&text, 2000)); } let _ = writeln!(self.file, "[{}] {label}:\n{text}\n---", Utc::now().to_rfc3339()); let _ = self.file.flush(); }
}

#[derive(Clone)] struct Entry { index: usize, timestamp: i64, date_utc: String, file: String, prefix: String, body: String, raw: String }
struct Partial { timestamp: i64, date_utc: String, file: String, prefix: String, body: String, raw: String }
fn load_entries(root: &Path) -> AppResult<Vec<Entry>> { let mut parsed = Vec::new(); for de in fs::read_dir(root)? { let de = de?; let path = de.path(); if !path.is_file() { continue; } let file = de.file_name().to_string_lossy().to_string(); let mut cur: Option<Partial> = None; for line in fs::read_to_string(path)?.lines() { if let Some(n) = parse_line(line, &file) { if let Some(p) = cur.take() { parsed.push(p); } cur = Some(n); } else if let Some(c) = cur.as_mut() { c.body.push('\n'); c.body.push_str(line); c.raw.push('\n'); c.raw.push_str(line); } } if let Some(p) = cur.take() { parsed.push(p); } } parsed.sort_by_key(|e| (e.timestamp, e.file.clone(), e.raw.clone())); Ok(parsed.into_iter().enumerate().map(|(index, e)| Entry { index, timestamp: e.timestamp, date_utc: e.date_utc, file: e.file, prefix: e.prefix, body: e.body, raw: e.raw }).collect()) }
fn parse_line(line: &str, file: &str) -> Option<Partial> { let (prefix, body) = line.split_once(": ")?; let timestamp = prefix.split_whitespace().next()?.parse::<i64>().ok()?; let date_utc = match Utc.timestamp_opt(timestamp, 0) { LocalResult::Single(dt) => format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day()), _ => return None }; Some(Partial { timestamp, date_utc, file: file.to_string(), prefix: prefix.to_string(), body: body.to_string(), raw: line.to_string() }) }

#[derive(Clone)] struct ExtraFile { index: usize, path: String, content: String, bytes: usize }
fn load_extra_files(root: &Path) -> AppResult<Vec<ExtraFile>> { if !root.exists() { return Ok(Vec::new()); } if !root.is_dir() { return Err(format!("extra data path is not a directory: {}", root.display()).into()); } let root = root.canonicalize()?; let mut raw = Vec::<(String, String, usize)>::new(); collect_extra_files(&root, &root, &mut raw)?; raw.sort_by_key(|(path, _, _)| path.clone()); Ok(raw.into_iter().enumerate().map(|(index, (path, content, bytes))| ExtraFile { index, path, content, bytes }).collect()) }
fn collect_extra_files(root: &Path, dir: &Path, out: &mut Vec<(String, String, usize)>) -> AppResult<()> { for de in fs::read_dir(dir)? { let de = de?; let path = de.path(); if path.is_dir() { collect_extra_files(root, &path, out)?; } else if path.is_file() { let bytes = de.metadata().map(|m| usize::try_from(m.len()).unwrap_or(usize::MAX)).unwrap_or(0); if let Ok(content) = fs::read_to_string(&path) { let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/"); out.push((rel, content, bytes)); } } } Ok(()) }
fn normalize_extra_path(path: &str) -> Option<String> { let p = Path::new(path); if p.is_absolute() { return None; } let mut parts = Vec::new(); for c in p.components() { match c { Component::Normal(part) => parts.push(part.to_string_lossy().to_string()), Component::CurDir => {}, _ => return None } } (!parts.is_empty()).then(|| parts.join("/")) }

struct OpenAiClient { key: String, http: Client }
impl OpenAiClient { fn new(key: String) -> AppResult<Self> { Ok(Self { key, http: Client::builder().timeout(Duration::from_secs(TIMEOUT_SECS)).build()? }) }
    fn response(&self, body: &Value, log: &mut RunLog) -> AppResult<Value> { for attempt in 0..=RETRIES { let res = self.http.post("https://api.openai.com/v1/responses").bearer_auth(&self.key).json(body).send(); let r = match res { Ok(r) => r, Err(e) => { log.value("openai_transport_error", &json!({"attempt":attempt+1,"error":e.to_string(),"timeout":e.is_timeout()})); if (e.is_timeout() || e.is_connect()) && attempt < RETRIES { let d = delay(None, "", attempt); log.line(format!("transport retry in {:.1}s", d.as_secs_f64())); thread::sleep(d); continue; } return Err(format!("OpenAI request failed: {e}").into()); } }; let status = r.status(); let retry_after = r.headers().get(RETRY_AFTER).and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<f64>().ok()).map(Duration::from_secs_f64); let text = r.text()?; if status.is_success() { return Ok(serde_json::from_str(&text)?); } log.value("openai_error", &json!({"status":status.as_u16(),"attempt":attempt+1,"body":parse_json(&text)})); if retryable(status, &text) && attempt < RETRIES { let d = delay(retry_after, &text, attempt); log.line(format!("retryable OpenAI error {status}; waiting {:.1}s", d.as_secs_f64())); thread::sleep(d); continue; } return Err(format!("OpenAI API error {status}: {text}").into()); } Err("OpenAI retry loop failed".into()) } }
fn retryable(status: StatusCode, text: &str) -> bool { status == StatusCode::REQUEST_TIMEOUT || status.is_server_error() || (status == StatusCode::TOO_MANY_REQUESTS && { let l = text.to_lowercase(); l.contains("rate_limit_exceeded") || l.contains("resource_unavailable") || l.contains("rate limit") || l.contains("resource unavailable") }) }
fn delay(retry_after: Option<Duration>, text: &str, attempt: usize) -> Duration { let mut d = retry_after.or_else(|| parse_try_again(text)).unwrap_or_else(|| Duration::from_secs(2u64.pow(attempt.min(6) as u32))); if is_token_limit(text) { d = d.max(Duration::from_secs(30 * (attempt as u64 + 1))); } min(d + Duration::from_secs(2), Duration::from_secs(MAX_BACKOFF)) }
fn parse_try_again(text: &str) -> Option<Duration> { let low = text.to_lowercase(); let i = low.find("try again in ")? + "try again in ".len(); let rest = text[i..].trim_start(); let n: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect(); if n.is_empty() { return None; } let v = n.parse::<f64>().ok()?; let unit = rest[n.len()..].trim_start().to_lowercase(); Some(Duration::from_secs_f64(if unit.starts_with("ms") { v / 1000.0 } else { v })) }
fn is_token_limit(text: &str) -> bool { let l = text.to_lowercase(); l.contains("tokens per min") || l.contains("tpm") || l.contains("\"type\": \"tokens\"") }

struct Harness { entries: Vec<Entry>, extra_files: Vec<ExtraFile>, cursor: Option<usize>, notes: String, summary: String, log: RunLog, input: Vec<Value> }
impl Harness {
    fn new(entries: Vec<Entry>, extra_files: Vec<ExtraFile>, log: RunLog) -> Self { Self { entries, extra_files, cursor: None, notes: String::new(), summary: String::new(), log, input: Vec::new() } }
    fn resume(entries: Vec<Entry>, extra_files: Vec<ExtraFile>, log: RunLog, path: &Path) -> AppResult<Self> { let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?; let mut h = Self::new(entries, extra_files, log); h.notes = value.get("notes").and_then(Value::as_str).unwrap_or("").to_string(); h.summary = value.get("summary").and_then(Value::as_str).unwrap_or("").to_string(); h.cursor = value.get("cursor").and_then(Value::as_u64).map(|v| v as usize); h.input = value.get("input").and_then(Value::as_array).cloned().unwrap_or_default(); h.log(format!("resumed session from {} with {} input item(s)", path.display(), h.input.len())); h.sanitize_function_pairs(); h.log(format!("new resumed-session snapshot path: {}", h.log.session_path.display())); Ok(h) }
    fn save_session(&mut self) -> AppResult<()> { self.sanitize_function_pairs(); let value = json!({"version":1,"saved_at_utc":Utc::now().to_rfc3339(),"notes":self.notes.clone(),"summary":self.summary.clone(),"cursor":self.cursor,"input":self.input.clone()}); fs::write(&self.log.session_path, serde_json::to_string_pretty(&value)?)?; self.log(format!("saved session snapshot: {}", self.log.session_path.display())); Ok(()) }
    fn log(&mut self, msg: impl AsRef<str>) { self.log.line(msg); }
    fn push_user(&mut self, content: String) { self.input.push(json!({"role":"user","content":content})); }
    fn add_user_messages(&mut self, label: &str, messages: Vec<String>) { for m in messages { self.log(format!("queued user message: {}", trunc(&m, 500))); self.push_user(format!("{label}:\n{m}")); } }
    fn run(&mut self, client: &OpenAiClient, user_input: &mut UserInput, model: &str, max_steps: usize) -> AppResult<String> { for step in 0..max_steps { self.drain_user(user_input)?; self.compact_history(); self.sanitize_function_pairs(); self.save_session()?; self.log(format!("step {}/{}: model request with {} items, approx {} chars", step+1, max_steps, self.input.len(), value_chars(&self.input))); let body = json!({"model":model,"service_tier":SERVICE_TIER,"store":false,"input":self.input.clone(),"instructions":self.instructions(),"tools":[self.tool()],"parallel_tool_calls":false}); self.log.value("openai_request", &body); let resp = client.response(&body, &mut self.log)?; self.log.value("openai_response", &resp); let output = resp.get("output").and_then(Value::as_array).cloned().unwrap_or_default(); let calls: Vec<Value> = output.iter().filter(|i| i.get("type").and_then(Value::as_str)==Some("function_call")).cloned().collect(); self.push_output_items(&output); self.save_session()?; if calls.is_empty() { let text = output_text(&resp); if text.trim().is_empty() { return Err("model produced no output".into()); } return Ok(text); } for call in calls { let call_id = call.get("call_id").and_then(Value::as_str).ok_or("missing call_id")?.to_string(); let args_s = call.get("arguments").and_then(Value::as_str).unwrap_or("{}"); let args: Value = serde_json::from_str(args_s).unwrap_or_else(|_| json!({})); let action = args.get("action").and_then(Value::as_str).unwrap_or(""); self.log(format!("log_action: {action} {}", trunc(args_s, 500))); if action == "finish" { let answer = args.get("answer").and_then(Value::as_str).unwrap_or("").trim().to_string(); if answer.is_empty() { return Err("finish missing answer".into()); } self.input.push(func_output(&call_id, &json!({"ok":true}))); self.save_session()?; return Ok(answer); } let result = if action == "wait" { self.wait(&args, user_input)? } else { self.action(&args) }; self.log_result(action, &result); self.log.value("tool_result", &json!({"action":action,"result":result.clone()})); self.input.push(func_output(&call_id, &result)); self.save_session()?; } } Err(format!("Reached --max-steps ({max_steps})").into()) }
    fn drain_user(&mut self, ui: &mut UserInput) -> AppResult<()> { let p = parse_controls(ui.drain_pending()); if p.quit { return Err("user requested exit".into()); } self.add_user_messages("Additional user input", p.messages); Ok(()) }
    fn compact_history(&mut self) { let chars = value_chars(&self.input); if chars < COMPACT_AFTER_CHARS || self.input.len() <= KEEP_RECENT_ITEMS + 2 { return; } let first = self.input.first().cloned(); let keep_start = self.input.len().saturating_sub(KEEP_RECENT_ITEMS); let old = self.input[..keep_start].to_vec(); let recent = self.input[keep_start..].to_vec(); let mut add = String::new(); for item in old.iter().skip(1) { add.push_str(&summarize_item(item)); add.push('\n'); } if !add.trim().is_empty() { if !self.summary.is_empty() { self.summary.push('\n'); } self.summary.push_str(&add); self.summary = tail_chars(&self.summary, SUMMARY_CHARS); } let mut new_input = Vec::new(); if let Some(f) = first { new_input.push(f); } new_input.push(json!({"role":"user","content":format!("Compacted conversation summary so far:\n{}\n\nCurrent notes:\n{}", self.summary.trim(), self.notes.trim())})); new_input.extend(recent); self.input = new_input; self.log(format!("compacted history: approx {chars} chars -> {} chars, {} items", value_chars(&self.input), self.input.len())); }
    fn sanitize_function_pairs(&mut self) { let output_ids: HashSet<String> = self.input.iter().filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output")).filter_map(|item| item.get("call_id").and_then(Value::as_str).map(ToString::to_string)).collect(); let call_ids: HashSet<String> = self.input.iter().filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call")).filter_map(|item| item.get("call_id").and_then(Value::as_str).map(ToString::to_string)).collect(); let before = self.input.len(); self.input.retain(|item| match item.get("type").and_then(Value::as_str) { Some("function_call") => item.get("call_id").and_then(Value::as_str).map(|id| output_ids.contains(id)).unwrap_or(false), Some("function_call_output") => item.get("call_id").and_then(Value::as_str).map(|id| call_ids.contains(id)).unwrap_or(false), _ => true }); let removed = before.saturating_sub(self.input.len()); if removed > 0 { self.log(format!("dropped {removed} orphaned function_call/function_call_output item(s) after compaction")); } }
    fn push_output_items(&mut self, output: &[Value]) { for item in output { match item.get("type").and_then(Value::as_str) { Some("function_call") => self.input.push(json!({"type":"function_call","call_id":item.get("call_id").cloned().unwrap_or(Value::Null),"name":item.get("name").cloned().unwrap_or(Value::Null),"arguments":item.get("arguments").cloned().unwrap_or_else(|| json!("{}"))})), Some("message") => { let t = msg_text(item); if !t.trim().is_empty() { self.input.push(json!({"role":"assistant","content":t})); } }, Some("reasoning") => self.log("skipped transient reasoning item because store=false"), _ => {} } } }
    fn instructions(&self) -> String { format!("You analyze QuickLogger logs and optional files under extra_data. There are {} parsed log entries and {} UTF-8 extra_data files. Use one tool: log_action. Keep requests targeted and page through results. Log actions: summary, get_index, previous, next, day, between, search, around. Extra-data actions: extra_summary, read_extra_file, search_extra_files/query_extra_files. Other actions: read_notes, write_notes, wait, finish. Dates are UTC YYYY-MM-DD. Tool outputs are intentionally compact; request another page or surrounding context if needed.", self.entries.len(), self.extra_files.len()) }
    fn tool(&self) -> Value { json!({"type":"function","name":"log_action","description":"Inspect QuickLogger logs, query extra_data files, update notes, wait for user, or finish.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["summary","get_index","previous","next","day","between","search","around","extra_summary","read_extra_file","search_extra_files","query_extra_files","read_notes","write_notes","wait","finish"]},"index":{"type":"integer"},"from_index":{"type":"integer"},"path":{"type":"string"},"date":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"query":{"type":"string"},"case_sensitive":{"type":"boolean"},"before":{"type":"integer"},"after":{"type":"integer"},"limit":{"type":"integer"},"offset":{"type":"integer"},"mode":{"type":"string","enum":["append","replace"]},"text":{"type":"string"},"question":{"type":"string"},"reason":{"type":"string"},"answer":{"type":"string"}},"required":["action"],"additionalProperties":false}}) }
    fn action(&mut self, args: &Value) -> Value { match args.get("action").and_then(Value::as_str).unwrap_or("") { "summary" => self.summary_action(), "get_index" => self.get_index(args), "previous" => self.prev(args), "next" => self.next(args), "day" => self.day(args), "between" => self.between(args), "search" => self.search(args), "around" => self.around(args), "extra_summary" => self.extra_summary(args), "read_extra_file" => self.read_extra_file(args), "search_extra_files"|"query_extra_files" => self.search_extra_files(args), "read_notes" => json!({"notes":self.notes}), "write_notes" => self.write_notes(args), other => json!({"error":format!("unknown action {other}")}) } }
    fn summary_action(&self) -> Value { let mut counts = Vec::<(String, usize)>::new(); for e in &self.entries { match counts.last_mut() { Some((d,c)) if *d==e.date_utc => *c += 1, _ => counts.push((e.date_utc.clone(),1)) } } json!({"entry_count":self.entries.len(),"extra_file_count":self.extra_files.len(),"cursor_index":self.cursor,"first":self.entries.first().map(|e| self.entry(e)),"last":self.entries.last().map(|e| self.entry(e)),"counts_by_day":counts.into_iter().map(|(date,count)|json!({"date":date,"count":count})).collect::<Vec<_>>()}) }
    fn get_index(&mut self, a: &Value) -> Value { let Some(i)=arg(a,"index") else { return json!({"error":"index required"}); }; match self.entries.get(i).cloned() { Some(e)=>{ self.cursor=Some(i); json!({"entry":self.entry(&e),"cursor_index":self.cursor}) }, None=>json!({"error":"index out of range","entry_count":self.entries.len()}) } }
    fn prev(&mut self, a: &Value) -> Value { let base=arg(a,"from_index").or(self.cursor).unwrap_or(self.entries.len()); if base==0 { return json!({"error":"already at first entry"}); } let i=min(base,self.entries.len())-1; self.cursor=Some(i); json!({"entry":self.entry(&self.entries[i]),"cursor_index":self.cursor}) }
    fn next(&mut self, a: &Value) -> Value { let i=arg(a,"from_index").or(self.cursor).and_then(|v|v.checked_add(1)).unwrap_or(0); if i>=self.entries.len() { return json!({"error":"already at last entry"}); } self.cursor=Some(i); json!({"entry":self.entry(&self.entries[i]),"cursor_index":self.cursor}) }
    fn day(&mut self, a: &Value) -> Value { let Some(d)=a.get("date").and_then(Value::as_str) else { return json!({"error":"date required"}); }; let idx=self.entries.iter().enumerate().filter_map(|(i,e)|(e.date_utc==d).then_some(i)).collect(); self.page(idx,a) }
    fn between(&mut self, a: &Value) -> Value { let Some(s)=a.get("start_date").and_then(Value::as_str) else { return json!({"error":"start_date required"}); }; let Some(e)=a.get("end_date").and_then(Value::as_str) else { return json!({"error":"end_date required"}); }; let idx=self.entries.iter().enumerate().filter_map(|(i,x)|(x.date_utc.as_str()>=s && x.date_utc.as_str()<=e).then_some(i)).collect(); self.page(idx,a) }
    fn search(&mut self, a: &Value) -> Value { let Some(q)=a.get("query").and_then(Value::as_str) else { return json!({"error":"query required"}); }; let case=a.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false); let qn=if case{q.to_string()}else{q.to_lowercase()}; let idx=self.entries.iter().enumerate().filter_map(|(i,e)|{ let b=if case{e.body.clone()}else{e.body.to_lowercase()}; b.contains(&qn).then_some(i)}).collect(); self.page(idx,a) }
    fn around(&mut self, a: &Value) -> Value { let Some(c)=arg(a,"index").or_else(||arg(a,"center_index")) else { return json!({"error":"index required"}); }; if c>=self.entries.len(){return json!({"error":"index out of range"});} let before=arg(a,"before").unwrap_or(5).min(25); let after=arg(a,"after").unwrap_or(5).min(25); let start=c.saturating_sub(before); let end=min(self.entries.len(),c+after+1); self.cursor=Some(c); json!({"start_index":start,"end_index_exclusive":end,"entries":self.entries[start..end].iter().map(|e|self.entry(e)).collect::<Vec<_>>()}) }
    fn page(&mut self, idx: Vec<usize>, a: &Value) -> Value { let total=idx.len(); let offset=arg(a,"offset").unwrap_or(0); let limit=arg(a,"limit").unwrap_or(RESULT_LIMIT).min(MAX_RESULT_LIMIT); let entries=idx.iter().skip(offset).take(limit).filter_map(|i|self.entries.get(*i)).map(|e|self.entry(e)).collect::<Vec<_>>(); if let Some(first)=entries.first(){self.cursor=first.get("index").and_then(Value::as_u64).map(|v|v as usize);} json!({"total_matches":total,"offset":offset,"limit":limit,"returned":entries.len(),"cursor_index":self.cursor,"entries":entries,"hint":"Results are compact; use offset or around for more context."}) }
    fn extra_summary(&self, a: &Value) -> Value { let offset=arg(a,"offset").unwrap_or(0); let limit=arg(a,"limit").unwrap_or(25).min(100); let files=self.extra_files.iter().skip(offset).take(limit).map(|f|json!({"index":f.index,"path":f.path,"bytes":f.bytes,"chars":f.content.chars().count()})).collect::<Vec<_>>(); json!({"extra_file_count":self.extra_files.len(),"offset":offset,"limit":limit,"returned":files.len(),"files":files}) }
    fn read_extra_file(&self, a: &Value) -> Value { let file = if let Some(i)=arg(a,"index") { self.extra_files.get(i) } else if let Some(path)=a.get("path").and_then(Value::as_str) { self.find_extra_file(path) } else { return json!({"error":"index or path required"}); }; let Some(f)=file else { return json!({"error":"extra_data file not found"}); }; let offset=arg(a,"offset").unwrap_or(0); let max_chars=arg(a,"limit").unwrap_or(EXTRA_READ_CHARS).min(MAX_EXTRA_READ_CHARS); let content=slice_chars(&f.content, offset, max_chars); json!({"index":f.index,"path":f.path,"bytes":f.bytes,"chars":f.content.chars().count(),"offset":offset,"limit":max_chars,"returned_chars":content.chars().count(),"content":content,"hint":"Use offset and limit to page through this file."}) }
    fn search_extra_files(&self, a: &Value) -> Value { let Some(q)=a.get("query").and_then(Value::as_str) else { return json!({"error":"query required"}); }; if q.trim().is_empty() { return json!({"error":"query must not be empty"}); } let case=a.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false); let qn=if case{q.to_string()}else{q.to_lowercase()}; let mut matches=Vec::new(); for f in &self.extra_files { let path_hay=if case{f.path.clone()}else{f.path.to_lowercase()}; let body_hay=if case{f.content.clone()}else{f.content.to_lowercase()}; if path_hay.contains(&qn) || body_hay.contains(&qn) { matches.push(f); } } let total=matches.len(); let offset=arg(a,"offset").unwrap_or(0); let limit=arg(a,"limit").unwrap_or(RESULT_LIMIT).min(MAX_RESULT_LIMIT); let files=matches.into_iter().skip(offset).take(limit).map(|f|json!({"index":f.index,"path":f.path,"bytes":f.bytes,"chars":f.content.chars().count(),"snippet":trunc(&f.content, EXTRA_SNIPPET_CHARS)})).collect::<Vec<_>>(); json!({"total_matches":total,"offset":offset,"limit":limit,"returned":files.len(),"files":files,"hint":"Use read_extra_file with index/path for full paged content."}) }
    fn find_extra_file(&self, path: &str) -> Option<&ExtraFile> { let normalized = normalize_extra_path(path)?; self.extra_files.iter().find(|f| f.path == normalized) }
    fn write_notes(&mut self, a: &Value) -> Value { let text=a.get("text").and_then(Value::as_str).unwrap_or(""); if a.get("mode").and_then(Value::as_str)==Some("replace") { self.notes=text.to_string(); } else { if !self.notes.trim().is_empty() && !text.trim().is_empty(){self.notes.push('\n');} self.notes.push_str(text); } self.notes=tail_chars(&self.notes, SUMMARY_CHARS); json!({"ok":true,"notes":self.notes}) }
    fn wait(&mut self, a: &Value, ui: &mut UserInput) -> AppResult<Value> { let q=a.get("question").and_then(Value::as_str).unwrap_or("Please provide more information."); self.log(format!("model is waiting for user input: {q}")); let p=parse_controls(ui.wait_for_messages()); if p.quit { return Err("user requested exit".into()); } Ok(json!({"status":if p.continue_requested{"continue"}else{"received"},"messages":p.messages})) }
    fn entry(&self, e: &Entry) -> Value { json!({"index":e.index,"timestamp":e.timestamp,"date_utc":e.date_utc,"file":e.file,"prefix":e.prefix,"body":trunc(&e.body,ENTRY_CHARS)}) }
    fn log_result(&mut self, action: &str, r: &Value) { self.log(format!("tool result: {action} returned={} total={} error={}", r.get("returned").unwrap_or(&Value::Null), r.get("total_matches").unwrap_or(&Value::Null), r.get("error").unwrap_or(&Value::Null))); }
}

fn func_output(call_id: &str, result: &Value) -> Value { json!({"type":"function_call_output","call_id":call_id,"output":result.to_string()}) }
fn arg(v:&Value,k:&str)->Option<usize>{v.get(k).and_then(Value::as_u64).and_then(|x|usize::try_from(x).ok())}
fn parse_json(text:&str)->Value{serde_json::from_str(text).unwrap_or_else(|_|json!({"raw":text}))}
fn value_chars(v:&impl serde::Serialize)->usize{serde_json::to_string(v).map(|s|s.chars().count()).unwrap_or(0)}
fn summarize_item(v:&Value)->String{ if let Some(role)=v.get("role").and_then(Value::as_str){return format!("{role}: {}",trunc(v.get("content").and_then(Value::as_str).unwrap_or(""),700));} match v.get("type").and_then(Value::as_str){Some("function_call")=>format!("tool_call {} {}",v.get("name").and_then(Value::as_str).unwrap_or(""),trunc(v.get("arguments").and_then(Value::as_str).unwrap_or(""),500)),Some("function_call_output")=>format!("tool_output {}",trunc(v.get("output").and_then(Value::as_str).unwrap_or(""),900)),_=>trunc(&v.to_string(),500)}}
fn output_text(resp:&Value)->String{resp.get("output").and_then(Value::as_array).map(|items|items.iter().filter(|i|i.get("type").and_then(Value::as_str)==Some("message")).map(msg_text).filter(|s|!s.is_empty()).collect::<Vec<_>>().join("\n")).unwrap_or_default()}
fn msg_text(item:&Value)->String{item.get("content").and_then(Value::as_array).map(|items|items.iter().filter(|c|c.get("type").and_then(Value::as_str)==Some("output_text")).filter_map(|c|c.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n")).unwrap_or_default()}
fn trunc(s:&str,max:usize)->String{let mut it=s.chars();let out=it.by_ref().take(max).collect::<String>();if it.next().is_some(){format!("{out}… [truncated]")}else{out}}
fn tail_chars(s:&str,max:usize)->String{let n=s.chars().count(); if n<=max{s.to_string()}else{s.chars().skip(n-max).collect()}}
fn slice_chars(s:&str,offset:usize,max:usize)->String{s.chars().skip(offset).take(max).collect()}
