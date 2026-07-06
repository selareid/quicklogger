use chrono::{Datelike, LocalResult, TimeZone, Utc};
use reqwest::{blocking::Client, header::RETRY_AFTER, StatusCode};
use serde_json::{json, Value};
use std::{
    cmp::min,
    collections::HashSet,
    env,
    error::Error,
    fs,
    io::{self, BufRead, Write},
    path::{Component, Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, SystemTime},
};

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
    log.line(format!(
        "history compaction: after ~{COMPACT_AFTER_CHARS} chars, keep last {KEEP_RECENT_ITEMS} items"
    ));

    let api_key = env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY must be set")?;
    let entries = load_entries(&cfg.logs_path)?;
    if entries.is_empty() {
        return Err(format!("No log entries found under {}", cfg.logs_path.display()).into());
    }
    log.line(format!("loaded {} log entries", entries.len()));

    let extra_files = load_extra_files(&cfg.extra_data_path)?;
    log.line(format!("loaded {} UTF-8 extra_data file(s)", extra_files.len()));

    let client = OpenAiClient::new(api_key)?;
    let resume_path = cfg.resume_path()?;
    let resumed = resume_path.is_some();
    let mut harness = if let Some(path) = resume_path.as_deref() {
        Harness::resume(entries, extra_files, log, path)?
    } else {
        let mut harness = Harness::new(entries, extra_files, log);
        let goal = cfg
            .goal
            .as_deref()
            .ok_or("Pass a goal with --goal \"...\" or use --resume/--resume-latest")?;
        harness.push_user(format!(
            "Goal: {goal}\n\nUse log_action to inspect QuickLogger logs and extra_data files. Keep notes concise. Prefer small targeted pages. If the goal is unclear, use action=wait. When done, use action=finish."
        ));
        harness
    };

    if resumed {
        if let Some(goal) = cfg.goal.as_deref() {
            harness.add_user_messages("Resume user input", vec![goal.to_string()]);
        }
        harness.save_session()?;
    }

    if !run_loop(&mut harness, &client, &mut input, &cfg.model, cfg.max_steps, "initial run") {
        return Ok(());
    }

    loop {
        harness.log("waiting for follow-up input; type a message, /continue, or /quit");
        let parsed = parse_controls(input.wait_for_messages());
        if parsed.quit || parsed.empty() {
            break;
        }
        if !parsed.messages.is_empty() {
            harness.add_user_messages("Follow-up", parsed.messages);
            let _ = harness.save_session();
        }
        if !run_loop(&mut harness, &client, &mut input, &cfg.model, cfg.max_steps, "follow-up run") {
            break;
        }
    }

    Ok(())
}

fn run_loop(
    harness: &mut Harness,
    client: &OpenAiClient,
    input: &mut UserInput,
    model: &str,
    max_steps: usize,
    label: &str,
) -> bool {
    loop {
        match harness.run(client, input, model, max_steps) {
            Ok(answer) => {
                harness.log(format!("{label} finished"));
                let _ = harness.save_session();
                println!(
                    "\n=== Answer ===\n{answer}\n\n=== Notes ===\n{}\n\n=== LLM Run Log ===\n{}\n\n=== Session Snapshot ===\n{}",
                    harness.notes.trim(),
                    harness.log.path.display(),
                    harness.log.session_path.display()
                );
                return true;
            }
            Err(error) => {
                harness.log(format!("{label} paused after error: {error}"));
                let _ = harness.save_session();
                eprintln!(
                    "\n=== Run paused after error ===\n{error}\n\nLLM run log: {}\nSession snapshot: {}\n\nType /continue to retry, add context to retry with it, or /quit to exit.",
                    harness.log.path.display(),
                    harness.log.session_path.display()
                );

                let parsed = parse_controls(input.wait_for_messages());
                if parsed.quit || parsed.empty() {
                    return false;
                }
                if !parsed.messages.is_empty() {
                    harness.add_user_messages("User input after error", parsed.messages);
                    let _ = harness.save_session();
                }
                harness.log("retrying from current compacted state");
            }
        }
    }
}

struct Config {
    goal: Option<String>,
    logs_path: PathBuf,
    extra_data_path: PathBuf,
    model: String,
    max_steps: usize,
    verbose: bool,
    resume: Option<PathBuf>,
    resume_latest: bool,
}

impl Config {
    fn from_args() -> AppResult<Self> {
        let mut goal = None;
        let mut logs_path = env::var("QUICKLOGGER_LOGS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOGS));
        let mut extra_data_path = env::var("QUICKLOGGER_EXTRA_DATA_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_EXTRA_DATA));
        let mut model = env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut max_steps = DEFAULT_MAX_STEPS;
        let mut verbose = false;
        let mut resume = None;
        let mut resume_latest = false;
        let mut positional = Vec::new();

        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--goal" | "-g" => goal = Some(next_arg(&mut args, "--goal")?),
                "--logs" | "-l" => logs_path = PathBuf::from(next_arg(&mut args, "--logs")?),
                "--extra-data" => extra_data_path = PathBuf::from(next_arg(&mut args, "--extra-data")?),
                "--model" | "-m" => model = next_arg(&mut args, "--model")?,
                "--max-steps" => max_steps = next_arg(&mut args, "--max-steps")?.parse()?,
                "--resume" => resume = Some(PathBuf::from(next_arg(&mut args, "--resume")?)),
                "--resume-latest" => resume_latest = true,
                "--verbose" | "-v" => verbose = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => positional.push(other.to_string()),
            }
        }

        if goal.is_none() && !positional.is_empty() {
            goal = Some(positional.join(" "));
        }
        if resume.is_some() && resume_latest {
            return Err("Use only one of --resume or --resume-latest".into());
        }
        if goal.is_none() && resume.is_none() && !resume_latest {
            return Err("Pass a goal with --goal \"...\" or use --resume/--resume-latest".into());
        }

        Ok(Self { goal, logs_path, extra_data_path, model, max_steps, verbose, resume, resume_latest })
    }

    fn resume_path(&self) -> AppResult<Option<PathBuf>> {
        if let Some(path) = &self.resume {
            Ok(Some(path.clone()))
        } else if self.resume_latest {
            Ok(Some(find_latest_session()?))
        } else {
            Ok(None)
        }
    }
}

fn next_arg(args: &mut std::iter::Peekable<impl Iterator<Item = String>>, flag: &str) -> AppResult<String> {
    args.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    eprintln!(
        "cargo run --bin log_llm -- --goal \"summarise my logs\" [--logs ./logs] [--extra-data ./extra_data] [--model gpt-5.4-mini] [--verbose]\n\nResume:\n  cargo run --bin log_llm -- --resume ./llm_logs/<run>.session.json\n  cargo run --bin log_llm -- --resume-latest\n  cargo run --bin log_llm -- --resume-latest --goal \"call finish with what you've got\""
    );
}

fn find_latest_session() -> AppResult<PathBuf> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(LOG_DIR)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !name.ends_with(".session.json") {
            continue;
        }
        let modified = entry.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
        if best.as_ref().map(|(time, _)| modified > *time).unwrap_or(true) {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path).ok_or_else(|| format!("No .session.json files found in {LOG_DIR}").into())
}

struct UserInput {
    receiver: Receiver<String>,
    closed: bool,
}

impl UserInput {
    fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            for line in io::stdin().lock().lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Self { receiver, closed: false }
    }

    fn drain_pending(&mut self) -> Vec<String> {
        let mut messages = Vec::new();
        loop {
            match self.receiver.try_recv() {
                Ok(line) => {
                    if let Some(message) = normalize_input(&line) {
                        messages.push(message);
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
        messages
    }

    fn wait_for_messages(&mut self) -> Vec<String> {
        let mut messages = self.drain_pending();
        if !messages.is_empty() || self.closed {
            return messages;
        }
        loop {
            match self.receiver.recv() {
                Ok(line) => {
                    if let Some(message) = normalize_input(&line) {
                        messages.push(message);
                        messages.extend(self.drain_pending());
                        return messages;
                    }
                }
                Err(_) => {
                    self.closed = true;
                    return messages;
                }
            }
        }
    }
}

#[derive(Default)]
struct ParsedControl {
    quit: bool,
    continue_requested: bool,
    messages: Vec<String>,
}

impl ParsedControl {
    fn empty(&self) -> bool {
        !self.quit && !self.continue_requested && self.messages.is_empty()
    }
}

fn parse_controls(messages: Vec<String>) -> ParsedControl {
    let mut parsed = ParsedControl::default();
    for message in messages {
        let trimmed = message.trim();
        let lower = trimmed.to_lowercase();
        if matches!(lower.as_str(), "/quit" | "/exit" | ":q" | "quit" | "exit") {
            parsed.quit = true;
        } else if lower == "/continue" || lower.starts_with("/continue ") {
            parsed.continue_requested = true;
            let rest = trimmed["/continue".len()..].trim();
            if !rest.is_empty() {
                parsed.messages.push(rest.to_string());
            }
        } else if !trimmed.is_empty() {
            parsed.messages.push(trimmed.to_string());
        }
    }
    parsed
}

fn normalize_input(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

struct RunLog {
    path: PathBuf,
    session_path: PathBuf,
    file: fs::File,
    verbose: bool,
}

impl RunLog {
    fn new(verbose: bool) -> AppResult<Self> {
        fs::create_dir_all(LOG_DIR)?;
        let path = Path::new(LOG_DIR).join(format!(
            "{}_{}.log",
            Utc::now().format("%Y%m%dT%H%M%SZ"),
            std::process::id()
        ));
        let session_path = path.with_extension("session.json");
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "# QuickLogger compact LLM run\n---")?;
        Ok(Self { path, session_path, file, verbose })
    }

    fn line(&mut self, message: impl AsRef<str>) {
        let message = message.as_ref();
        eprintln!("[log-llm] {message}");
        let _ = writeln!(self.file, "[{}] {message}", Utc::now().to_rfc3339());
        let _ = self.file.flush();
    }

    fn value(&mut self, label: &str, value: &Value) {
        let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        if self.verbose && matches!(label, "openai_error" | "tool_result") {
            eprintln!("[log-llm] {label}: {}", truncate_chars(&text, 2_000));
        }
        let _ = writeln!(self.file, "[{}] {label}:\n{text}\n---", Utc::now().to_rfc3339());
        let _ = self.file.flush();
    }
}

#[derive(Clone)]
struct Entry {
    index: usize,
    timestamp: i64,
    date_utc: String,
    file: String,
    prefix: String,
    body: String,
    raw: String,
}

struct PartialEntry {
    timestamp: i64,
    date_utc: String,
    file: String,
    prefix: String,
    body: String,
    raw: String,
}

fn load_entries(root: &Path) -> AppResult<Vec<Entry>> {
    let mut parsed = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file = entry.file_name().to_string_lossy().to_string();
        let mut current: Option<PartialEntry> = None;
        for line in fs::read_to_string(path)?.lines() {
            if let Some(next) = parse_log_line(line, &file) {
                if let Some(prev) = current.take() {
                    parsed.push(prev);
                }
                current = Some(next);
            } else if let Some(current) = current.as_mut() {
                current.body.push('\n');
                current.body.push_str(line);
                current.raw.push('\n');
                current.raw.push_str(line);
            }
        }
        if let Some(prev) = current.take() {
            parsed.push(prev);
        }
    }

    parsed.sort_by_key(|entry| (entry.timestamp, entry.file.clone(), entry.raw.clone()));
    Ok(parsed
        .into_iter()
        .enumerate()
        .map(|(index, entry)| Entry {
            index,
            timestamp: entry.timestamp,
            date_utc: entry.date_utc,
            file: entry.file,
            prefix: entry.prefix,
            body: entry.body,
            raw: entry.raw,
        })
        .collect())
}

fn parse_log_line(line: &str, file: &str) -> Option<PartialEntry> {
    let (prefix, body) = line.split_once(": ")?;
    let timestamp = prefix.split_whitespace().next()?.parse::<i64>().ok()?;
    let date_utc = match Utc.timestamp_opt(timestamp, 0) {
        LocalResult::Single(dt) => format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day()),
        _ => return None,
    };
    Some(PartialEntry {
        timestamp,
        date_utc,
        file: file.to_string(),
        prefix: prefix.to_string(),
        body: body.to_string(),
        raw: line.to_string(),
    })
}

#[derive(Clone)]
struct ExtraFile {
    index: usize,
    path: String,
    content: String,
    bytes: usize,
}

fn load_extra_files(root: &Path) -> AppResult<Vec<ExtraFile>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    if !root.is_dir() {
        return Err(format!("extra data path is not a directory: {}", root.display()).into());
    }
    let root = root.canonicalize()?;
    let mut raw = Vec::<(String, String, usize)>::new();
    collect_extra_files(&root, &root, &mut raw)?;
    raw.sort_by_key(|(path, _, _)| path.clone());
    Ok(raw
        .into_iter()
        .enumerate()
        .map(|(index, (path, content, bytes))| ExtraFile { index, path, content, bytes })
        .collect())
}

fn collect_extra_files(root: &Path, dir: &Path, out: &mut Vec<(String, String, usize)>) -> AppResult<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_extra_files(root, &path, out)?;
        } else if path.is_file() {
            let bytes = entry.metadata().map(|m| usize::try_from(m.len()).unwrap_or(usize::MAX)).unwrap_or(0);
            if let Ok(content) = fs::read_to_string(&path) {
                let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                out.push((rel, content, bytes));
            }
        }
    }
    Ok(())
}

fn normalize_extra_path(path: &str) -> Option<String> {
    let path = Path::new(path);
    if path.is_absolute() {
        return None;
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

struct OpenAiClient {
    key: String,
    http: Client,
}

impl OpenAiClient {
    fn new(key: String) -> AppResult<Self> {
        Ok(Self {
            key,
            http: Client::builder().timeout(Duration::from_secs(TIMEOUT_SECS)).build()?,
        })
    }

    fn response(&self, body: &Value, log: &mut RunLog) -> AppResult<Value> {
        for attempt in 0..=RETRIES {
            let response = self.http.post("https://api.openai.com/v1/responses").bearer_auth(&self.key).json(body).send();
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    log.value(
                        "openai_transport_error",
                        &json!({"attempt": attempt + 1, "error": error.to_string(), "timeout": error.is_timeout()}),
                    );
                    if (error.is_timeout() || error.is_connect()) && attempt < RETRIES {
                        let delay = retry_delay(None, "", attempt);
                        log.line(format!("transport retry in {:.1}s", delay.as_secs_f64()));
                        thread::sleep(delay);
                        continue;
                    }
                    return Err(format!("OpenAI request failed: {error}").into());
                }
            };

            let status = response.status();
            let retry_after = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<f64>().ok())
                .map(Duration::from_secs_f64);
            let text = response.text()?;

            if status.is_success() {
                return Ok(serde_json::from_str(&text)?);
            }

            log.value(
                "openai_error",
                &json!({"status": status.as_u16(), "attempt": attempt + 1, "body": parse_json(&text)}),
            );
            if retryable(status, &text) && attempt < RETRIES {
                let delay = retry_delay(retry_after, &text, attempt);
                log.line(format!("retryable OpenAI error {status}; waiting {:.1}s", delay.as_secs_f64()));
                thread::sleep(delay);
                continue;
            }
            return Err(format!("OpenAI API error {status}: {text}").into());
        }
        Err("OpenAI retry loop failed".into())
    }
}

fn retryable(status: StatusCode, text: &str) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
        || (status == StatusCode::TOO_MANY_REQUESTS && {
            let lower = text.to_lowercase();
            lower.contains("rate_limit_exceeded")
                || lower.contains("resource_unavailable")
                || lower.contains("rate limit")
                || lower.contains("resource unavailable")
        })
}

fn retry_delay(retry_after: Option<Duration>, text: &str, attempt: usize) -> Duration {
    let mut delay = retry_after
        .or_else(|| parse_try_again(text))
        .unwrap_or_else(|| Duration::from_secs(2u64.pow(attempt.min(6) as u32)));
    if is_token_limit(text) {
        delay = delay.max(Duration::from_secs(30 * (attempt as u64 + 1)));
    }
    min(delay + Duration::from_secs(2), Duration::from_secs(MAX_BACKOFF))
}

fn parse_try_again(text: &str) -> Option<Duration> {
    let lower = text.to_lowercase();
    let start = lower.find("try again in ")? + "try again in ".len();
    let rest = text[start..].trim_start();
    let number: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    if number.is_empty() {
        return None;
    }
    let value = number.parse::<f64>().ok()?;
    let unit = rest[number.len()..].trim_start().to_lowercase();
    Some(Duration::from_secs_f64(if unit.starts_with("ms") { value / 1000.0 } else { value }))
}

fn is_token_limit(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("tokens per min") || lower.contains("tpm") || lower.contains("\"type\": \"tokens\"")
}

struct Harness {
    entries: Vec<Entry>,
    extra_files: Vec<ExtraFile>,
    cursor: Option<usize>,
    notes: String,
    summary: String,
    log: RunLog,
    input: Vec<Value>,
}

impl Harness {
    fn new(entries: Vec<Entry>, extra_files: Vec<ExtraFile>, log: RunLog) -> Self {
        Self { entries, extra_files, cursor: None, notes: String::new(), summary: String::new(), log, input: Vec::new() }
    }

    fn resume(entries: Vec<Entry>, extra_files: Vec<ExtraFile>, log: RunLog, path: &Path) -> AppResult<Self> {
        let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let mut harness = Self::new(entries, extra_files, log);
        harness.notes = value.get("notes").and_then(Value::as_str).unwrap_or("").to_string();
        harness.summary = value.get("summary").and_then(Value::as_str).unwrap_or("").to_string();
        harness.cursor = value.get("cursor").and_then(Value::as_u64).map(|value| value as usize);
        harness.input = value.get("input").and_then(Value::as_array).cloned().unwrap_or_default();
        harness.log(format!("resumed session from {} with {} input item(s)", path.display(), harness.input.len()));
        harness.drop_orphan_function_outputs();
        harness.log(format!("new resumed-session snapshot path: {}", harness.log.session_path.display()));
        Ok(harness)
    }

    fn save_session(&mut self) -> AppResult<()> {
        self.drop_orphan_function_outputs();
        let value = json!({
            "version": 1,
            "saved_at_utc": Utc::now().to_rfc3339(),
            "notes": self.notes,
            "summary": self.summary,
            "cursor": self.cursor,
            "input": self.input,
        });
        fs::write(&self.log.session_path, serde_json::to_string_pretty(&value)?)?;
        self.log(format!("saved session snapshot: {}", self.log.session_path.display()));
        Ok(())
    }

    fn log(&mut self, message: impl AsRef<str>) {
        self.log.line(message);
    }

    fn push_user(&mut self, content: String) {
        self.input.push(json!({"role": "user", "content": content}));
    }

    fn add_user_messages(&mut self, label: &str, messages: Vec<String>) {
        for message in messages {
            self.log(format!("queued user message: {}", truncate_chars(&message, 500)));
            self.push_user(format!("{label}:\n{message}"));
        }
    }

    fn run(&mut self, client: &OpenAiClient, user_input: &mut UserInput, model: &str, max_steps: usize) -> AppResult<String> {
        for step in 0..max_steps {
            self.drain_user(user_input)?;
            self.compact_history();
            self.drop_orphan_function_outputs();
            self.save_session()?;

            self.log(format!(
                "step {}/{}: model request with {} items, approx {} chars",
                step + 1,
                max_steps,
                self.input.len(),
                value_chars(&self.input)
            ));

            let body = json!({
                "model": model,
                "service_tier": SERVICE_TIER,
                "store": false,
                "input": self.input.clone(),
                "instructions": self.instructions(),
                "tools": [self.tool()],
                "parallel_tool_calls": false,
            });
            self.log.value("openai_request", &body);
            let response = client.response(&body, &mut self.log)?;
            self.log.value("openai_response", &response);

            let output = response.get("output").and_then(Value::as_array).cloned().unwrap_or_default();
            let calls: Vec<Value> = output
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
                .cloned()
                .collect();
            self.push_output_items(&output);

            if calls.is_empty() {
                let text = output_text(&response);
                if text.trim().is_empty() {
                    return Err("model produced no output".into());
                }
                self.save_session()?;
                return Ok(text);
            }

            for call in calls {
                let call_id = call.get("call_id").and_then(Value::as_str).ok_or("missing call_id")?.to_string();
                let args_text = call.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                let args: Value = serde_json::from_str(args_text).unwrap_or_else(|_| json!({}));
                let action = args.get("action").and_then(Value::as_str).unwrap_or("");
                self.log(format!("log_action: {action} {}", truncate_chars(args_text, 500)));

                if action == "finish" {
                    let answer = args.get("answer").and_then(Value::as_str).unwrap_or("").trim().to_string();
                    if answer.is_empty() {
                        return Err("finish missing answer".into());
                    }
                    self.input.push(function_output(&call_id, &json!({"ok": true})));
                    self.save_session()?;
                    return Ok(answer);
                }

                let result = if action == "wait" { self.wait(&args, user_input)? } else { self.action(&args) };
                self.log_result(action, &result);
                self.log.value("tool_result", &json!({"action": action, "result": result.clone()}));
                self.input.push(function_output(&call_id, &result));
                self.save_session()?;
            }
        }
        Err(format!("Reached --max-steps ({max_steps})").into())
    }

    fn drain_user(&mut self, user_input: &mut UserInput) -> AppResult<()> {
        let parsed = parse_controls(user_input.drain_pending());
        if parsed.quit {
            return Err("user requested exit".into());
        }
        self.add_user_messages("Additional user input", parsed.messages);
        Ok(())
    }

    fn compact_history(&mut self) {
        let chars = value_chars(&self.input);
        if chars < COMPACT_AFTER_CHARS || self.input.len() <= KEEP_RECENT_ITEMS + 2 {
            return;
        }

        let first = self.input.first().cloned();
        let keep_start = self.input.len().saturating_sub(KEEP_RECENT_ITEMS);
        let old = self.input[..keep_start].to_vec();
        let recent = self.input[keep_start..].to_vec();
        let mut addition = String::new();
        for item in old.iter().skip(1) {
            addition.push_str(&summarize_item(item));
            addition.push('\n');
        }
        if !addition.trim().is_empty() {
            if !self.summary.is_empty() {
                self.summary.push('\n');
            }
            self.summary.push_str(&addition);
            self.summary = tail_chars(&self.summary, SUMMARY_CHARS);
        }

        let mut new_input = Vec::new();
        if let Some(first) = first {
            new_input.push(first);
        }
        new_input.push(json!({
            "role": "user",
            "content": format!(
                "Compacted conversation summary so far:\n{}\n\nCurrent notes:\n{}",
                self.summary.trim(),
                self.notes.trim()
            )
        }));
        new_input.extend(recent);
        self.input = new_input;
        self.log(format!(
            "compacted history: approx {chars} chars -> {} chars, {} items",
            value_chars(&self.input),
            self.input.len()
        ));
    }

    fn drop_orphan_function_outputs(&mut self) {
        let call_ids: HashSet<String> = self
            .input
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .filter_map(|item| item.get("call_id").and_then(Value::as_str).map(ToString::to_string))
            .collect();

        let before = self.input.len();
        self.input.retain(|item| {
            if item.get("type").and_then(Value::as_str) != Some("function_call_output") {
                return true;
            }
            item.get("call_id")
                .and_then(Value::as_str)
                .map(|id| call_ids.contains(id))
                .unwrap_or(false)
        });
        let removed = before.saturating_sub(self.input.len());
        if removed > 0 {
            self.log(format!("dropped {removed} orphaned function_call_output item(s) after compaction"));
        }
    }

    fn push_output_items(&mut self, output: &[Value]) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => self.input.push(json!({
                    "type": "function_call",
                    "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                    "name": item.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": item.get("arguments").cloned().unwrap_or_else(|| json!("{}")),
                })),
                Some("message") => {
                    let text = message_text(item);
                    if !text.trim().is_empty() {
                        self.input.push(json!({"role": "assistant", "content": text}));
                    }
                }
                Some("reasoning") => self.log("skipped transient reasoning item because store=false"),
                _ => {}
            }
        }
    }

    fn instructions(&self) -> String {
        format!(
            "You analyze QuickLogger logs and optional files under extra_data. There are {} parsed log entries and {} UTF-8 extra_data files. Use one tool: log_action. Keep requests targeted and page through results. Log actions: summary, get_index, previous, next, day, between, search, around. Extra-data actions: extra_summary, read_extra_file, search_extra_files/query_extra_files. Other actions: read_notes, write_notes, wait, finish. Dates are UTC YYYY-MM-DD. Tool outputs are intentionally compact; request another page or surrounding context if needed.",
            self.entries.len(),
            self.extra_files.len()
        )
    }

    fn tool(&self) -> Value {
        json!({
            "type": "function",
            "name": "log_action",
            "description": "Inspect QuickLogger logs, query extra_data files, update notes, wait for user, or finish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["summary", "get_index", "previous", "next", "day", "between", "search", "around", "extra_summary", "read_extra_file", "search_extra_files", "query_extra_files", "read_notes", "write_notes", "wait", "finish"]},
                    "index": {"type": "integer"},
                    "from_index": {"type": "integer"},
                    "path": {"type": "string"},
                    "date": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "query": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "before": {"type": "integer"},
                    "after": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["append", "replace"]},
                    "text": {"type": "string"},
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn action(&mut self, args: &Value) -> Value {
        match args.get("action").and_then(Value::as_str).unwrap_or("") {
            "summary" => self.summary_action(),
            "get_index" => self.get_index(args),
            "previous" => self.previous(args),
            "next" => self.next(args),
            "day" => self.day(args),
            "between" => self.between(args),
            "search" => self.search(args),
            "around" => self.around(args),
            "extra_summary" => self.extra_summary(args),
            "read_extra_file" => self.read_extra_file(args),
            "search_extra_files" | "query_extra_files" => self.search_extra_files(args),
            "read_notes" => json!({"notes": self.notes}),
            "write_notes" => self.write_notes(args),
            other => json!({"error": format!("unknown action {other}")}),
        }
    }

    fn summary_action(&self) -> Value {
        let mut counts = Vec::<(String, usize)>::new();
        for entry in &self.entries {
            match counts.last_mut() {
                Some((date, count)) if *date == entry.date_utc => *count += 1,
                _ => counts.push((entry.date_utc.clone(), 1)),
            }
        }
        json!({
            "entry_count": self.entries.len(),
            "extra_file_count": self.extra_files.len(),
            "cursor_index": self.cursor,
            "first": self.entries.first().map(|entry| self.entry_json(entry)),
            "last": self.entries.last().map(|entry| self.entry_json(entry)),
            "counts_by_day": counts.into_iter().map(|(date, count)| json!({"date": date, "count": count})).collect::<Vec<_>>()
        })
    }

    fn get_index(&mut self, args: &Value) -> Value {
        let Some(index) = arg_usize(args, "index") else {
            return json!({"error": "index required"});
        };
        match self.entries.get(index).cloned() {
            Some(entry) => {
                self.cursor = Some(index);
                json!({"entry": self.entry_json(&entry), "cursor_index": self.cursor})
            }
            None => json!({"error": "index out of range", "entry_count": self.entries.len()}),
        }
    }

    fn previous(&mut self, args: &Value) -> Value {
        let base = arg_usize(args, "from_index").or(self.cursor).unwrap_or(self.entries.len());
        if base == 0 {
            return json!({"error": "already at first entry"});
        }
        let index = min(base, self.entries.len()) - 1;
        self.cursor = Some(index);
        json!({"entry": self.entry_json(&self.entries[index]), "cursor_index": self.cursor})
    }

    fn next(&mut self, args: &Value) -> Value {
        let index = arg_usize(args, "from_index").or(self.cursor).and_then(|value| value.checked_add(1)).unwrap_or(0);
        if index >= self.entries.len() {
            return json!({"error": "already at last entry"});
        }
        self.cursor = Some(index);
        json!({"entry": self.entry_json(&self.entries[index]), "cursor_index": self.cursor})
    }

    fn day(&mut self, args: &Value) -> Value {
        let Some(date) = args.get("date").and_then(Value::as_str) else {
            return json!({"error": "date required"});
        };
        let indexes = self.entries.iter().enumerate().filter_map(|(index, entry)| (entry.date_utc == date).then_some(index)).collect();
        self.page(indexes, args)
    }

    fn between(&mut self, args: &Value) -> Value {
        let Some(start) = args.get("start_date").and_then(Value::as_str) else {
            return json!({"error": "start_date required"});
        };
        let Some(end) = args.get("end_date").and_then(Value::as_str) else {
            return json!({"error": "end_date required"});
        };
        let indexes = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.date_utc.as_str() >= start && entry.date_utc.as_str() <= end).then_some(index))
            .collect();
        self.page(indexes, args)
    }

    fn search(&mut self, args: &Value) -> Value {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return json!({"error": "query required"});
        };
        let case_sensitive = args.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false);
        let needle = if case_sensitive { query.to_string() } else { query.to_lowercase() };
        let indexes = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let haystack = if case_sensitive { entry.body.clone() } else { entry.body.to_lowercase() };
                haystack.contains(&needle).then_some(index)
            })
            .collect();
        self.page(indexes, args)
    }

    fn around(&mut self, args: &Value) -> Value {
        let Some(center) = arg_usize(args, "index").or_else(|| arg_usize(args, "center_index")) else {
            return json!({"error": "index required"});
        };
        if center >= self.entries.len() {
            return json!({"error": "index out of range"});
        }
        let before = arg_usize(args, "before").unwrap_or(5).min(25);
        let after = arg_usize(args, "after").unwrap_or(5).min(25);
        let start = center.saturating_sub(before);
        let end = min(self.entries.len(), center + after + 1);
        self.cursor = Some(center);
        json!({
            "start_index": start,
            "end_index_exclusive": end,
            "entries": self.entries[start..end].iter().map(|entry| self.entry_json(entry)).collect::<Vec<_>>()
        })
    }

    fn page(&mut self, indexes: Vec<usize>, args: &Value) -> Value {
        let total = indexes.len();
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(RESULT_LIMIT).min(MAX_RESULT_LIMIT);
        let entries = indexes
            .iter()
            .skip(offset)
            .take(limit)
            .filter_map(|index| self.entries.get(*index))
            .map(|entry| self.entry_json(entry))
            .collect::<Vec<_>>();
        if let Some(first) = entries.first() {
            self.cursor = first.get("index").and_then(Value::as_u64).map(|value| value as usize);
        }
        json!({
            "total_matches": total,
            "offset": offset,
            "limit": limit,
            "returned": entries.len(),
            "cursor_index": self.cursor,
            "entries": entries,
            "hint": "Results are compact; use offset or around for more context."
        })
    }

    fn extra_summary(&self, args: &Value) -> Value {
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(25).min(100);
        let files = self
            .extra_files
            .iter()
            .skip(offset)
            .take(limit)
            .map(|file| json!({"index": file.index, "path": file.path, "bytes": file.bytes, "chars": file.content.chars().count()}))
            .collect::<Vec<_>>();
        json!({"extra_file_count": self.extra_files.len(), "offset": offset, "limit": limit, "returned": files.len(), "files": files})
    }

    fn read_extra_file(&self, args: &Value) -> Value {
        let file = if let Some(index) = arg_usize(args, "index") {
            self.extra_files.get(index)
        } else if let Some(path) = args.get("path").and_then(Value::as_str) {
            self.find_extra_file(path)
        } else {
            return json!({"error": "index or path required"});
        };
        let Some(file) = file else {
            return json!({"error": "extra_data file not found"});
        };
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(EXTRA_READ_CHARS).min(MAX_EXTRA_READ_CHARS);
        let content = slice_chars(&file.content, offset, limit);
        json!({
            "index": file.index,
            "path": file.path,
            "bytes": file.bytes,
            "chars": file.content.chars().count(),
            "offset": offset,
            "limit": limit,
            "returned_chars": content.chars().count(),
            "content": content,
            "hint": "Use offset and limit to page through this file."
        })
    }

    fn search_extra_files(&self, args: &Value) -> Value {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return json!({"error": "query required"});
        };
        if query.trim().is_empty() {
            return json!({"error": "query must not be empty"});
        }
        let case_sensitive = args.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false);
        let needle = if case_sensitive { query.to_string() } else { query.to_lowercase() };
        let mut matches = Vec::new();
        for file in &self.extra_files {
            let path_haystack = if case_sensitive { file.path.clone() } else { file.path.to_lowercase() };
            let body_haystack = if case_sensitive { file.content.clone() } else { file.content.to_lowercase() };
            if path_haystack.contains(&needle) || body_haystack.contains(&needle) {
                matches.push(file);
            }
        }
        let total = matches.len();
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(RESULT_LIMIT).min(MAX_RESULT_LIMIT);
        let files = matches
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|file| {
                json!({
                    "index": file.index,
                    "path": file.path,
                    "bytes": file.bytes,
                    "chars": file.content.chars().count(),
                    "snippet": truncate_chars(&file.content, EXTRA_SNIPPET_CHARS)
                })
            })
            .collect::<Vec<_>>();
        json!({"total_matches": total, "offset": offset, "limit": limit, "returned": files.len(), "files": files, "hint": "Use read_extra_file with index/path for full paged content."})
    }

    fn find_extra_file(&self, path: &str) -> Option<&ExtraFile> {
        let normalized = normalize_extra_path(path)?;
        self.extra_files.iter().find(|file| file.path == normalized)
    }

    fn write_notes(&mut self, args: &Value) -> Value {
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");
        if args.get("mode").and_then(Value::as_str) == Some("replace") {
            self.notes = text.to_string();
        } else {
            if !self.notes.trim().is_empty() && !text.trim().is_empty() {
                self.notes.push('\n');
            }
            self.notes.push_str(text);
        }
        self.notes = tail_chars(&self.notes, SUMMARY_CHARS);
        json!({"ok": true, "notes": self.notes})
    }

    fn wait(&mut self, args: &Value, user_input: &mut UserInput) -> AppResult<Value> {
        let question = args.get("question").and_then(Value::as_str).unwrap_or("Please provide more information.");
        self.log(format!("model is waiting for user input: {question}"));
        let parsed = parse_controls(user_input.wait_for_messages());
        if parsed.quit {
            return Err("user requested exit".into());
        }
        Ok(json!({"status": if parsed.continue_requested { "continue" } else { "received" }, "messages": parsed.messages}))
    }

    fn entry_json(&self, entry: &Entry) -> Value {
        json!({
            "index": entry.index,
            "timestamp": entry.timestamp,
            "date_utc": entry.date_utc,
            "file": entry.file,
            "prefix": entry.prefix,
            "body": truncate_chars(&entry.body, ENTRY_CHARS)
        })
    }

    fn log_result(&mut self, action: &str, result: &Value) {
        self.log(format!(
            "tool result: {action} returned={} total={} error={}",
            result.get("returned").unwrap_or(&Value::Null),
            result.get("total_matches").unwrap_or(&Value::Null),
            result.get("error").unwrap_or(&Value::Null)
        ));
    }
}

fn function_output(call_id: &str, result: &Value) -> Value {
    json!({"type": "function_call_output", "call_id": call_id, "output": result.to_string()})
}

fn arg_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).and_then(|value| usize::try_from(value).ok())
}

fn parse_json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| json!({"raw": text}))
}

fn value_chars(value: &impl serde::Serialize) -> usize {
    serde_json::to_string(value).map(|text| text.chars().count()).unwrap_or(0)
}

fn summarize_item(value: &Value) -> String {
    if let Some(role) = value.get("role").and_then(Value::as_str) {
        return format!("{role}: {}", truncate_chars(value.get("content").and_then(Value::as_str).unwrap_or(""), 700));
    }
    match value.get("type").and_then(Value::as_str) {
        Some("function_call") => format!(
            "tool_call {} {}",
            value.get("name").and_then(Value::as_str).unwrap_or(""),
            truncate_chars(value.get("arguments").and_then(Value::as_str).unwrap_or(""), 500)
        ),
        Some("function_call_output") => format!(
            "tool_output {}",
            truncate_chars(value.get("output").and_then(Value::as_str).unwrap_or(""), 900)
        ),
        _ => truncate_chars(&value.to_string(), 500),
    }
}

fn output_text(response: &Value) -> String {
    response
        .get("output")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter(|item| item.get("type").and_then(Value::as_str) == Some("message")).map(message_text).filter(|text| !text.is_empty()).collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

fn message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|content| content.get("type").and_then(Value::as_str) == Some("output_text"))
                .filter_map(|content| content.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn truncate_chars(text: &str, max: usize) -> String {
    let mut chars = text.chars();
    let output = chars.by_ref().take(max).collect::<String>();
    if chars.next().is_some() {
        format!("{output}… [truncated]")
    } else {
        output
    }
}

fn tail_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        text.to_string()
    } else {
        text.chars().skip(count - max).collect()
    }
}

fn slice_chars(text: &str, offset: usize, max: usize) -> String {
    text.chars().skip(offset).take(max).collect()
}
