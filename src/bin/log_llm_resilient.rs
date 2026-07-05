use chrono::{Datelike, LocalResult, NaiveDate, TimeZone, Utc};
use reqwest::{blocking::Client, header::RETRY_AFTER, StatusCode};
use serde_json::{json, Value};
use std::{
    cmp::min,
    env,
    error::Error,
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Duration,
};

const DEFAULT_LOGS_PATH: &str = "./logs";
const DEFAULT_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_MAX_STEPS: usize = 30;
const OPENAI_SERVICE_TIER: &str = "flex";
const OPENAI_TIMEOUT_SECS: u64 = 900;
const LLM_LOGS_PATH: &str = "./llm_logs";
const MAX_ENTRY_CHARS: usize = 1_500;
const MAX_RESULT_ENTRIES: usize = 100;
const VERBOSE_VALUE_CHARS: usize = 2_000;
const CONSOLE_ARGUMENT_CHARS: usize = 500;
const MAX_OPENAI_RETRIES: usize = 8;
const MAX_OPENAI_BACKOFF_SECS: u64 = 60;

type AppResult<T> = Result<T, Box<dyn Error>>;

fn main() -> AppResult<()> {
    let config = Config::from_args()?;
    let mut user_input = UserInput::spawn();
    let mut logger = RunLogger::new(config.verbose)?;

    logger.log(format!("run log path: {}", logger.path().display()));
    logger.log(format!("logs path: {}", config.logs_path.display()));
    logger.log(format!("model: {}", config.model));
    logger.log(format!("service tier: {OPENAI_SERVICE_TIER}"));
    logger.log(format!("HTTP timeout: {OPENAI_TIMEOUT_SECS}s"));
    logger.log(format!("max OpenAI retries: {MAX_OPENAI_RETRIES}"));
    logger.log(format!("max steps: {}", config.max_steps));
    logger.log(format!("goal: {}", config.goal));
    logger.log("type extra messages at any time; they will be added before the next model request");
    logger.log("after an answer or error, type /continue, a follow-up, or /quit");

    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(api_key) => {
            logger.log("OPENAI_API_KEY is set");
            api_key
        }
        Err(_) => {
            logger.log("OPENAI_API_KEY is missing");
            return Err("OPENAI_API_KEY must be set to use the log LLM harness".into());
        }
    };

    let entries = load_log_entries(&config.logs_path)?;
    if entries.is_empty() {
        return Err(format!("No log entries found under {}", config.logs_path.display()).into());
    }
    logger.log(format!(
        "loaded {} parsed entries, date range {} to {}",
        entries.len(),
        entries.first().map(|entry| entry.date_utc.as_str()).unwrap_or("unknown"),
        entries.last().map(|entry| entry.date_utc.as_str()).unwrap_or("unknown")
    ));

    let client = OpenAiClient::new(api_key)?;
    let mut harness = Harness::new(entries, logger);
    harness.push_initial_goal(&config.goal);

    if !run_until_answer(
        &mut harness,
        &client,
        &mut user_input,
        &config.model,
        config.max_steps,
        "initial run",
    ) {
        return Ok(());
    }

    loop {
        harness.log("waiting for follow-up input; type a message, /continue, or /quit");
        let parsed = parse_control_messages(user_input.wait_for_messages());
        if parsed.quit || parsed.is_empty() {
            harness.log("exiting follow-up loop");
            break;
        }
        if !parsed.messages.is_empty() {
            harness.add_user_messages("Follow-up user input", parsed.messages);
        }
        if !run_until_answer(
            &mut harness,
            &client,
            &mut user_input,
            &config.model,
            config.max_steps,
            "follow-up run",
        ) {
            break;
        }
    }

    Ok(())
}

fn run_until_answer(
    harness: &mut Harness,
    client: &OpenAiClient,
    user_input: &mut UserInput,
    model: &str,
    max_steps: usize,
    label: &str,
) -> bool {
    loop {
        match harness.run(client, user_input, model, max_steps) {
            Ok(answer) => {
                harness.log(format!("{label} finished successfully"));
                print_run_result(&answer, harness);
                return true;
            }
            Err(error) => {
                harness.log(format!("{label} paused after error: {error}"));
                eprintln!(
                    "\n=== Run paused after error ===\n{error}\n\nLLM run log: {}\n\nType /continue to retry from the current state, type extra context to add it before retrying, or type /quit to exit.",
                    harness.log_path().display()
                );

                let parsed = parse_control_messages(user_input.wait_for_messages());
                if parsed.quit || parsed.is_empty() {
                    harness.log("exiting after paused error");
                    return false;
                }
                if !parsed.messages.is_empty() {
                    harness.add_user_messages("User input after run error", parsed.messages);
                }
                if parsed.continue_requested {
                    harness.log("/continue received after error; retrying from current state");
                } else {
                    harness.log("user input received after error; retrying from current state");
                }
            }
        }
    }
}

fn print_run_result(answer: &str, harness: &Harness) {
    println!("\n=== Answer ===\n{answer}");
    println!("\n=== Notes ===\n{}", harness.notes.trim());
    println!("\n=== LLM Run Log ===\n{}", harness.log_path().display());
}

struct Config {
    goal: String,
    logs_path: PathBuf,
    model: String,
    max_steps: usize,
    verbose: bool,
}

impl Config {
    fn from_args() -> AppResult<Self> {
        let mut goal: Option<String> = None;
        let mut logs_path = env::var("QUICKLOGGER_LOGS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOGS_PATH));
        let mut model = env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut max_steps = DEFAULT_MAX_STEPS;
        let mut verbose = false;
        let mut positional_goal = Vec::new();

        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--goal" | "-g" => goal = Some(next_arg(&mut args, "--goal")?),
                "--logs" | "-l" => logs_path = PathBuf::from(next_arg(&mut args, "--logs")?),
                "--model" | "-m" => model = next_arg(&mut args, "--model")?,
                "--max-steps" => max_steps = next_arg(&mut args, "--max-steps")?.parse()?,
                "--verbose" | "-v" => verbose = true,
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => positional_goal.push(other.to_string()),
            }
        }

        let goal = goal
            .or_else(|| (!positional_goal.is_empty()).then(|| positional_goal.join(" ")))
            .ok_or("Pass a goal with --goal \"...\" or as positional text")?;

        Ok(Self { goal, logs_path, model, max_steps, verbose })
    }
}

fn next_arg(args: &mut std::iter::Peekable<impl Iterator<Item = String>>, flag: &str) -> AppResult<String> {
    args.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run --bin log_llm -- --goal \"summarise my logs from yesterday\" [--logs ./logs] [--model gpt-5.4-mini] [--max-steps 30] [--verbose]\n\nProcessing tier:\n  Uses service_tier=flex, a 900s timeout, and retry/backoff for flex timeouts/resource-unavailable errors.\n\nInteractive input:\n  Type messages while running. After an answer or error, type /continue, a follow-up, or /quit."
    );
}

struct UserInput {
    receiver: Receiver<String>,
    closed: bool,
}

impl UserInput {
    fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel::<String>();
        thread::spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
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
                    if let Some(message) = normalize_user_message(&line) {
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
                    if let Some(message) = normalize_user_message(&line) {
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
struct ParsedControlMessages {
    quit: bool,
    continue_requested: bool,
    messages: Vec<String>,
}

impl ParsedControlMessages {
    fn is_empty(&self) -> bool {
        !self.quit && !self.continue_requested && self.messages.is_empty()
    }
}

fn parse_control_messages(messages: Vec<String>) -> ParsedControlMessages {
    let mut parsed = ParsedControlMessages::default();
    for message in messages {
        let trimmed = message.trim();
        let lowered = trimmed.to_lowercase();
        if is_exit_input(trimmed) {
            parsed.quit = true;
        } else if lowered == "/continue" || lowered.starts_with("/continue ") {
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

struct RunLogger {
    path: PathBuf,
    file: fs::File,
    verbose: bool,
}

impl RunLogger {
    fn new(verbose: bool) -> AppResult<Self> {
        fs::create_dir_all(LLM_LOGS_PATH)?;
        let started_at = Utc::now();
        let path = Path::new(LLM_LOGS_PATH).join(format!(
            "{}_{}.log",
            started_at.format("%Y%m%dT%H%M%SZ"),
            std::process::id()
        ));
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "# QuickLogger LLM harness run")?;
        writeln!(file, "started_at_utc: {}", started_at.to_rfc3339())?;
        writeln!(file, "pid: {}", std::process::id())?;
        writeln!(file, "---")?;
        file.flush()?;
        Ok(Self { path, file, verbose })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn log(&mut self, message: impl AsRef<str>) {
        let message = message.as_ref();
        eprintln!("[log-llm] {message}");
        let _ = writeln!(self.file, "[{}] {message}", Utc::now().to_rfc3339());
        let _ = self.file.flush();
    }

    fn log_value(&mut self, label: &str, value: &Value) {
        let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        if self.verbose && matches!(label, "tool_result" | "openai_error" | "openai_transport_error") {
            eprintln!("[log-llm] {label}: {}", truncate_chars(&text, VERBOSE_VALUE_CHARS));
        }
        let _ = writeln!(self.file, "[{}] {label}:", Utc::now().to_rfc3339());
        let _ = writeln!(self.file, "{text}");
        let _ = writeln!(self.file, "---");
        let _ = self.file.flush();
    }
}

#[derive(Debug, Clone)]
struct LogEntry {
    index: usize,
    timestamp: i64,
    date_utc: String,
    file_name: String,
    prefix: String,
    body: String,
    raw: String,
}

struct PartialLogEntry {
    timestamp: i64,
    date_utc: String,
    file_name: String,
    prefix: String,
    body: String,
    raw: String,
}

fn load_log_entries(logs_root: &Path) -> AppResult<Vec<LogEntry>> {
    let mut parsed = Vec::new();
    for entry in fs::read_dir(logs_root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = entry.file_name().to_string_lossy().to_string();
        let contents = fs::read_to_string(path)?;
        let mut current: Option<PartialLogEntry> = None;
        for line in contents.lines() {
            if let Some(next) = parse_log_line(line, &file_name) {
                if let Some(previous) = current.take() {
                    parsed.push(previous);
                }
                current = Some(next);
            } else if let Some(existing) = current.as_mut() {
                existing.body.push('\n');
                existing.body.push_str(line);
                existing.raw.push('\n');
                existing.raw.push_str(line);
            }
        }
        if let Some(previous) = current.take() {
            parsed.push(previous);
        }
    }
    parsed.sort_by_key(|entry| (entry.timestamp, entry.file_name.clone(), entry.raw.clone()));
    Ok(parsed
        .into_iter()
        .enumerate()
        .map(|(index, entry)| LogEntry {
            index,
            timestamp: entry.timestamp,
            date_utc: entry.date_utc,
            file_name: entry.file_name,
            prefix: entry.prefix,
            body: entry.body,
            raw: entry.raw,
        })
        .collect())
}

fn parse_log_line(line: &str, file_name: &str) -> Option<PartialLogEntry> {
    let (prefix, body) = line.split_once(": ")?;
    let timestamp = prefix.split_whitespace().next()?.parse::<i64>().ok()?;
    let date_utc = match Utc.timestamp_opt(timestamp, 0) {
        LocalResult::Single(dt) => format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day()),
        _ => return None,
    };
    Some(PartialLogEntry {
        timestamp,
        date_utc,
        file_name: file_name.to_string(),
        prefix: prefix.to_string(),
        body: body.to_string(),
        raw: line.to_string(),
    })
}

struct OpenAiClient {
    api_key: String,
    http: Client,
}

impl OpenAiClient {
    fn new(api_key: String) -> AppResult<Self> {
        Ok(Self {
            api_key,
            http: Client::builder()
                .timeout(Duration::from_secs(OPENAI_TIMEOUT_SECS))
                .build()?,
        })
    }

    fn create_response(&self, request_body: &Value, logger: &mut RunLogger) -> AppResult<Value> {
        for attempt in 0..=MAX_OPENAI_RETRIES {
            let response = match self
                .http
                .post("https://api.openai.com/v1/responses")
                .bearer_auth(&self.api_key)
                .json(request_body)
                .send()
            {
                Ok(response) => response,
                Err(error) => {
                    logger.log_value(
                        "openai_transport_error",
                        &json!({
                            "attempt": attempt + 1,
                            "max_attempts": MAX_OPENAI_RETRIES + 1,
                            "error": error.to_string(),
                            "is_timeout": error.is_timeout(),
                            "is_connect": error.is_connect(),
                        }),
                    );
                    if (error.is_timeout() || error.is_connect()) && attempt < MAX_OPENAI_RETRIES {
                        let delay = retry_delay(None, "", attempt).0;
                        logger.log(format!(
                            "OpenAI transport error: waiting {:.3}s before retry (attempt {}/{})",
                            delay.as_secs_f64(),
                            attempt + 1,
                            MAX_OPENAI_RETRIES
                        ));
                        thread::sleep(delay);
                        continue;
                    }
                    return Err(format!("OpenAI request failed: {error}").into());
                }
            };

            let status = response.status();
            let retry_after_header = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_retry_after_header);
            let text = response.text()?;

            if status.is_success() {
                if attempt > 0 {
                    logger.log(format!("OpenAI request succeeded after {attempt} retry attempt(s)"));
                }
                return Ok(serde_json::from_str(&text)?);
            }

            logger.log_value(
                "openai_error",
                &json!({
                    "status": status.as_u16(),
                    "attempt": attempt + 1,
                    "max_attempts": MAX_OPENAI_RETRIES + 1,
                    "body": parse_error_body(&text),
                }),
            );

            if is_retryable_openai_error(status, &text) && attempt < MAX_OPENAI_RETRIES {
                let (delay, source) = retry_delay(retry_after_header, &text, attempt);
                logger.log(format!(
                    "retryable OpenAI error {status}: waiting {:.3}s before retry ({source}, attempt {}/{})",
                    delay.as_secs_f64(),
                    attempt + 1,
                    MAX_OPENAI_RETRIES
                ));
                thread::sleep(delay);
                continue;
            }

            return Err(format!("OpenAI API error {status}: {text}").into());
        }
        Err("OpenAI API request failed after retry loop".into())
    }
}

struct Harness {
    entries: Vec<LogEntry>,
    cursor_index: Option<usize>,
    notes: String,
    logger: RunLogger,
    input_items: Vec<Value>,
}

impl Harness {
    fn new(entries: Vec<LogEntry>, logger: RunLogger) -> Self {
        Self { entries, cursor_index: None, notes: String::new(), logger, input_items: Vec::new() }
    }

    fn log_path(&self) -> &Path { self.logger.path() }
    fn log(&mut self, message: impl AsRef<str>) { self.logger.log(message); }

    fn push_initial_goal(&mut self, goal: &str) {
        self.input_items.push(user_message_item(format!(
            "Goal: {goal}\n\nUse the available log-reading actions to inspect the QuickLogger log data. Keep useful intermediate findings in the notes object. Do not claim a fact from the logs unless you saw it in an action result. If the goal is unclear, call wait_for_user_input. When done, call finish."
        )));
    }

    fn add_user_messages(&mut self, label: &str, messages: Vec<String>) {
        for message in messages {
            self.log(format!("queued user message for next model call: {}", truncate_chars(&message, CONSOLE_ARGUMENT_CHARS)));
            self.input_items.push(user_message_item(format!("{label}:\n{message}")));
        }
    }

    fn drain_queued_user_messages(&mut self, user_input: &mut UserInput) -> AppResult<()> {
        let parsed = parse_control_messages(user_input.drain_pending());
        if parsed.quit { return Err("user requested exit".into()); }
        if parsed.continue_requested { self.log("/continue received while already running; continuing current run"); }
        self.add_user_messages("Additional user input while the harness was running", parsed.messages);
        Ok(())
    }

    fn run(&mut self, client: &OpenAiClient, user_input: &mut UserInput, model: &str, max_steps: usize) -> AppResult<String> {
        for step in 0..max_steps {
            self.drain_queued_user_messages(user_input)?;
            self.log(format!("step {}/{}: asking model what to do next ({} conversation item(s))", step + 1, max_steps, self.input_items.len()));
            let request_body = json!({
                "model": model,
                "service_tier": OPENAI_SERVICE_TIER,
                "input": self.input_items.clone(),
                "instructions": self.instructions(),
                "tools": self.tools(),
                "parallel_tool_calls": false,
                "store": false,
            });
            self.logger.log_value("openai_request", &request_body);

            let response = client.create_response(&request_body, &mut self.logger)?;
            self.logger.log_value("openai_response", &response);
            let output = response.get("output").and_then(Value::as_array).cloned().unwrap_or_default();
            let tool_calls = output
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
                .cloned()
                .collect::<Vec<_>>();
            self.log(format!("step {}/{}: model returned {} tool call(s)", step + 1, max_steps, tool_calls.len()));
            self.push_model_output_items(&output);

            if tool_calls.is_empty() {
                let text = extract_output_text(&response);
                self.log(format!("step {}/{}: model produced final text directly ({} chars)", step + 1, max_steps, text.chars().count()));
                if text.trim().is_empty() { return Err(format!("Model stopped without output after step {step}").into()); }
                return Ok(text);
            }

            for call in tool_calls {
                let name = call.get("name").and_then(Value::as_str).ok_or("Function call missing name")?;
                let call_id = call.get("call_id").and_then(Value::as_str).ok_or("Function call missing call_id")?;
                let arguments = call.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                self.log(format!("tool: {name} {}", truncate_chars(arguments, CONSOLE_ARGUMENT_CHARS)));

                if name == "finish" {
                    let answer = args.get("answer").and_then(Value::as_str).unwrap_or("").trim().to_string();
                    if answer.is_empty() { return Err("finish called without an answer".into()); }
                    self.log(format!("finish called with {} answer chars", answer.chars().count()));
                    self.input_items.push(function_output_item(call_id, &json!({ "ok": true, "answer": answer.clone() })));
                    return Ok(answer);
                }

                let result = if name == "wait_for_user_input" {
                    self.wait_for_user_input(&args, user_input)?
                } else {
                    self.call_tool(name, &args)
                };
                self.log_tool_summary(name, &result);
                self.logger.log_value("tool_result", &json!({ "tool": name, "arguments": args, "result": result.clone() }));
                self.input_items.push(function_output_item(call_id, &result));
            }
        }
        Err(format!("Reached --max-steps ({max_steps}) before the model called finish").into())
    }

    fn push_model_output_items(&mut self, output: &[Value]) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => self.input_items.push(json!({
                    "type": "function_call",
                    "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                    "name": item.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": item.get("arguments").cloned().unwrap_or_else(|| json!("{}")),
                })),
                Some("message") => {
                    let text = extract_message_item_text(item);
                    if !text.trim().is_empty() { self.input_items.push(assistant_message_item(text)); }
                }
                Some("reasoning") => self.log("skipped transient reasoning item in conversation history because store=false"),
                Some(other) => self.log(format!("skipped unsupported model output item type in conversation history: {other}")),
                None => self.log("skipped model output item without a type"),
            }
        }
    }

    fn wait_for_user_input(&mut self, args: &Value, user_input: &mut UserInput) -> AppResult<Value> {
        let question = args.get("question").and_then(Value::as_str).unwrap_or("Please provide more information.");
        let reason = args.get("reason").and_then(Value::as_str).unwrap_or("");
        if reason.trim().is_empty() {
            self.log(format!("model is waiting for user input: {question}"));
        } else {
            self.log(format!("model is waiting for user input: {question} (reason: {reason})"));
        }
        let parsed = parse_control_messages(user_input.wait_for_messages());
        if parsed.quit { return Err("user requested exit".into()); }
        Ok(json!({
            "status": if parsed.continue_requested { "continue" } else { "received" },
            "messages": parsed.messages,
        }))
    }

    fn instructions(&self) -> String {
        format!(
            "You are an analysis harness for a personal QuickLogger log corpus. There are {} parsed entries. Prefer targeted tool calls over dumping everything. Use get_log_summary first unless the goal already names an exact date or search term. Maintain concise notes. If the user's goal is too vague or you need clarification, call wait_for_user_input. Finish with a direct answer and mention uncertainty.",
            self.entries.len()
        )
    }

    fn tools(&self) -> Vec<Value> {
        vec![
            tool("get_log_summary", "Return corpus metadata.", json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false })),
            tool("get_log_entry_by_index", "Read one log entry by zero-based index.", json!({ "type": "object", "properties": { "index": { "type": "integer", "minimum": 0 } }, "required": ["index"], "additionalProperties": false })),
            tool("get_previous_log_entry", "Read the previous log entry.", json!({ "type": "object", "properties": { "from_index": { "type": "integer", "minimum": 0 } }, "additionalProperties": false })),
            tool("get_next_log_entry", "Read the next log entry.", json!({ "type": "object", "properties": { "from_index": { "type": "integer", "minimum": 0 } }, "additionalProperties": false })),
            tool("get_log_entries_for_day", "Return entries for a UTC day.", json!({ "type": "object", "properties": { "date": { "type": "string" }, "limit": { "type": "integer", "minimum": 1, "maximum": 100 }, "offset": { "type": "integer", "minimum": 0 } }, "required": ["date"], "additionalProperties": false })),
            tool("get_log_entries_between", "Return entries for an inclusive UTC date range.", json!({ "type": "object", "properties": { "start_date": { "type": "string" }, "end_date": { "type": "string" }, "limit": { "type": "integer", "minimum": 1, "maximum": 100 }, "offset": { "type": "integer", "minimum": 0 } }, "required": ["start_date", "end_date"], "additionalProperties": false })),
            tool("search_log_entries", "Substring search over log bodies.", json!({ "type": "object", "properties": { "query": { "type": "string" }, "case_sensitive": { "type": "boolean" }, "limit": { "type": "integer", "minimum": 1, "maximum": 100 }, "offset": { "type": "integer", "minimum": 0 } }, "required": ["query"], "additionalProperties": false })),
            tool("get_log_entries_around", "Return entries around a center index.", json!({ "type": "object", "properties": { "center_index": { "type": "integer", "minimum": 0 }, "before": { "type": "integer", "minimum": 0, "maximum": 50 }, "after": { "type": "integer", "minimum": 0, "maximum": 50 } }, "required": ["center_index"], "additionalProperties": false })),
            tool("read_notes", "Read current notes.", json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false })),
            tool("write_notes", "Append to or replace notes.", json!({ "type": "object", "properties": { "mode": { "type": "string", "enum": ["append", "replace"] }, "text": { "type": "string" } }, "required": ["mode", "text"], "additionalProperties": false })),
            tool("wait_for_user_input", "Pause and wait for user input.", json!({ "type": "object", "properties": { "question": { "type": "string" }, "reason": { "type": "string" } }, "required": ["question"], "additionalProperties": false })),
            tool("finish", "Finish the current turn.", json!({ "type": "object", "properties": { "answer": { "type": "string" } }, "required": ["answer"], "additionalProperties": false })),
        ]
    }

    fn call_tool(&mut self, name: &str, args: &Value) -> Value {
        match name {
            "get_log_summary" => self.get_log_summary(),
            "get_log_entry_by_index" => self.get_log_entry_by_index(args),
            "get_previous_log_entry" => self.get_previous_log_entry(args),
            "get_next_log_entry" => self.get_next_log_entry(args),
            "get_log_entries_for_day" => self.get_log_entries_for_day(args),
            "get_log_entries_between" => self.get_log_entries_between(args),
            "search_log_entries" => self.search_log_entries(args),
            "get_log_entries_around" => self.get_log_entries_around(args),
            "read_notes" => json!({ "notes": self.notes.clone() }),
            "write_notes" => self.write_notes(args),
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }

    fn get_log_summary(&self) -> Value {
        let mut counts_by_day = Vec::<(String, usize)>::new();
        for entry in &self.entries {
            match counts_by_day.last_mut() {
                Some((date, count)) if *date == entry.date_utc => *count += 1,
                _ => counts_by_day.push((entry.date_utc.clone(), 1)),
            }
        }
        json!({
            "entry_count": self.entries.len(),
            "cursor_index": self.cursor_index,
            "first_entry": self.entries.first().map(|entry| self.entry_json(entry)),
            "last_entry": self.entries.last().map(|entry| self.entry_json(entry)),
            "counts_by_day": counts_by_day.into_iter().map(|(date, count)| json!({ "date": date, "count": count })).collect::<Vec<_>>()
        })
    }

    fn get_log_entry_by_index(&mut self, args: &Value) -> Value {
        let Some(index) = arg_usize(args, "index") else { return json!({ "error": "index is required" }); };
        match self.entries.get(index).cloned() {
            Some(entry) => { self.cursor_index = Some(index); json!({ "entry": self.entry_json(&entry), "cursor_index": self.cursor_index }) }
            None => json!({ "error": "index out of range", "entry_count": self.entries.len() }),
        }
    }

    fn get_previous_log_entry(&mut self, args: &Value) -> Value {
        if self.entries.is_empty() { return json!({ "error": "no entries loaded" }); }
        let base = arg_usize(args, "from_index").or(self.cursor_index).unwrap_or(self.entries.len());
        if base == 0 { return json!({ "error": "already at first entry", "cursor_index": self.cursor_index }); }
        let index = min(base, self.entries.len()) - 1;
        self.cursor_index = Some(index);
        json!({ "entry": self.entry_json(&self.entries[index]), "cursor_index": self.cursor_index })
    }

    fn get_next_log_entry(&mut self, args: &Value) -> Value {
        if self.entries.is_empty() { return json!({ "error": "no entries loaded" }); }
        let index = arg_usize(args, "from_index").or(self.cursor_index).and_then(|value| value.checked_add(1)).unwrap_or(0);
        if index >= self.entries.len() { return json!({ "error": "already at last entry", "cursor_index": self.cursor_index }); }
        self.cursor_index = Some(index);
        json!({ "entry": self.entry_json(&self.entries[index]), "cursor_index": self.cursor_index })
    }

    fn get_log_entries_for_day(&mut self, args: &Value) -> Value {
        let Some(date) = args.get("date").and_then(Value::as_str) else { return json!({ "error": "date is required" }); };
        if let Err(error) = validate_date(date) { return json!({ "error": error }); }
        let indices = self.entries.iter().enumerate().filter_map(|(index, entry)| (entry.date_utc == date).then_some(index)).collect::<Vec<_>>();
        self.page_entries(indices, args)
    }

    fn get_log_entries_between(&mut self, args: &Value) -> Value {
        let Some(start_date) = args.get("start_date").and_then(Value::as_str) else { return json!({ "error": "start_date is required" }); };
        let Some(end_date) = args.get("end_date").and_then(Value::as_str) else { return json!({ "error": "end_date is required" }); };
        if validate_date(start_date).is_err() || validate_date(end_date).is_err() || start_date > end_date {
            return json!({ "error": "invalid date range" });
        }
        let indices = self.entries.iter().enumerate().filter_map(|(index, entry)| {
            (entry.date_utc.as_str() >= start_date && entry.date_utc.as_str() <= end_date).then_some(index)
        }).collect::<Vec<_>>();
        self.page_entries(indices, args)
    }

    fn search_log_entries(&mut self, args: &Value) -> Value {
        let Some(query) = args.get("query").and_then(Value::as_str) else { return json!({ "error": "query is required" }); };
        if query.trim().is_empty() { return json!({ "error": "query must not be empty" }); }
        let case_sensitive = args.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false);
        let query_normalized = if case_sensitive { query.to_string() } else { query.to_lowercase() };
        let indices = self.entries.iter().enumerate().filter_map(|(index, entry)| {
            let haystack = if case_sensitive { entry.body.clone() } else { entry.body.to_lowercase() };
            haystack.contains(&query_normalized).then_some(index)
        }).collect::<Vec<_>>();
        self.page_entries(indices, args)
    }

    fn get_log_entries_around(&mut self, args: &Value) -> Value {
        let Some(center_index) = arg_usize(args, "center_index") else { return json!({ "error": "center_index is required" }); };
        if center_index >= self.entries.len() { return json!({ "error": "center_index out of range", "entry_count": self.entries.len() }); }
        let before = arg_usize(args, "before").unwrap_or(5).min(50);
        let after = arg_usize(args, "after").unwrap_or(5).min(50);
        let start = center_index.saturating_sub(before);
        let end = min(self.entries.len(), center_index + after + 1);
        self.cursor_index = Some(center_index);
        json!({
            "start_index": start,
            "end_index_exclusive": end,
            "cursor_index": self.cursor_index,
            "entries": self.entries[start..end].iter().map(|entry| self.entry_json(entry)).collect::<Vec<_>>()
        })
    }

    fn page_entries(&mut self, indices: Vec<usize>, args: &Value) -> Value {
        let total = indices.len();
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(25).min(MAX_RESULT_ENTRIES);
        let entries = indices.iter().skip(offset).take(limit).filter_map(|index| self.entries.get(*index)).map(|entry| self.entry_json(entry)).collect::<Vec<_>>();
        if let Some(first) = entries.first() {
            self.cursor_index = first.get("index").and_then(Value::as_u64).map(|value| value as usize);
        }
        json!({ "total_matches": total, "offset": offset, "limit": limit, "returned": entries.len(), "cursor_index": self.cursor_index, "entries": entries })
    }

    fn write_notes(&mut self, args: &Value) -> Value {
        let mode = args.get("mode").and_then(Value::as_str).unwrap_or("append");
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");
        match mode {
            "replace" => self.notes = text.to_string(),
            "append" => {
                if !self.notes.trim().is_empty() && !text.trim().is_empty() { self.notes.push('\n'); }
                self.notes.push_str(text);
            }
            _ => return json!({ "error": "mode must be append or replace" }),
        }
        json!({ "ok": true, "notes": self.notes.clone() })
    }

    fn entry_json(&self, entry: &LogEntry) -> Value {
        let mut value = json!({
            "index": entry.index,
            "timestamp": entry.timestamp,
            "date_utc": entry.date_utc.clone(),
            "file": entry.file_name.clone(),
            "prefix": entry.prefix.clone(),
            "body": truncate_chars(&entry.body, MAX_ENTRY_CHARS),
            "raw": truncate_chars(&entry.raw, MAX_ENTRY_CHARS),
        });
        if let Some(fields) = parse_media_fields(&entry.body) { value["media"] = fields; }
        value
    }

    fn log_tool_summary(&mut self, name: &str, result: &Value) {
        let mut parts = Vec::new();
        for key in ["status", "total_matches", "returned", "cursor_index", "entry_count", "offset", "limit"] {
            if let Some(value) = result.get(key) { parts.push(format!("{key}={value}")); }
        }
        if let Some(entries) = result.get("entries").and_then(Value::as_array) { parts.push(format!("entries={}", entries.len())); }
        if let Some(messages) = result.get("messages").and_then(Value::as_array) { parts.push(format!("messages={}", messages.len())); }
        if let Some(error) = result.get("error") { parts.push(format!("error={error}")); }
        if parts.is_empty() { self.log(format!("tool result: {name} completed")); } else { self.log(format!("tool result: {name} {}", parts.join(", "))); }
    }
}

fn tool(name: &str, description: &str, parameters: Value) -> Value {
    json!({ "type": "function", "name": name, "description": description, "parameters": parameters })
}

fn user_message_item(content: String) -> Value { json!({ "role": "user", "content": content }) }
fn assistant_message_item(content: String) -> Value { json!({ "role": "assistant", "content": content }) }
fn function_output_item(call_id: &str, result: &Value) -> Value { json!({ "type": "function_call_output", "call_id": call_id, "output": result.to_string() }) }

fn normalize_user_message(line: &str) -> Option<String> {
    let trimmed = line.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn is_exit_input(message: &str) -> bool {
    matches!(message.trim().to_lowercase().as_str(), "/quit" | "/exit" | ":q" | "quit" | "exit")
}

fn is_retryable_openai_error(status: StatusCode, response_text: &str) -> bool {
    status == StatusCode::REQUEST_TIMEOUT || status.is_server_error() || is_retryable_flex_or_rate_limit(status, response_text)
}

fn is_retryable_flex_or_rate_limit(status: StatusCode, response_text: &str) -> bool {
    if status != StatusCode::TOO_MANY_REQUESTS { return false; }
    let code = extract_openai_error_code(response_text).unwrap_or_default().to_lowercase();
    let lower = response_text.to_lowercase();
    code == "rate_limit_exceeded" || code == "resource_unavailable" || lower.contains("rate limit reached") || lower.contains("resource unavailable") || lower.contains("insufficient resources")
}

fn parse_retry_after_header(value: &str) -> Option<Duration> {
    seconds_to_duration(value.trim().parse::<f64>().ok()?)
}

fn retry_delay(retry_after_header: Option<Duration>, response_text: &str, attempt: usize) -> (Duration, &'static str) {
    if let Some(delay) = retry_after_header {
        return (cap_retry_delay(delay + Duration::from_millis(250)), "Retry-After header");
    }
    if let Some(delay) = parse_retry_delay_from_error_message(response_text) {
        return (cap_retry_delay(delay + Duration::from_millis(250)), "OpenAI error message");
    }
    let fallback = Duration::from_secs(2_u64.pow(attempt.min(6) as u32));
    (cap_retry_delay(fallback), "exponential fallback")
}

fn parse_retry_delay_from_error_message(response_text: &str) -> Option<Duration> {
    let lower = response_text.to_lowercase();
    let marker = "try again in ";
    let marker_start = lower.find(marker)? + marker.len();
    let rest = response_text[marker_start..].trim_start();
    let number = rest.chars().take_while(|ch| ch.is_ascii_digit() || *ch == '.').collect::<String>();
    if number.is_empty() { return None; }
    let value = number.parse::<f64>().ok()?;
    let unit_rest = rest[number.len()..].trim_start().to_lowercase();
    if unit_rest.starts_with("ms") || unit_rest.starts_with("millisecond") { seconds_to_duration(value / 1_000.0) } else { seconds_to_duration(value) }
}

fn seconds_to_duration(seconds: f64) -> Option<Duration> {
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
}

fn cap_retry_delay(delay: Duration) -> Duration {
    min(delay, Duration::from_secs(MAX_OPENAI_BACKOFF_SECS))
}

fn extract_openai_error_code(response_text: &str) -> Option<String> {
    serde_json::from_str::<Value>(response_text).ok()?.get("error")?.get("code")?.as_str().map(ToString::to_string)
}

fn parse_error_body(response_text: &str) -> Value {
    serde_json::from_str::<Value>(response_text).unwrap_or_else(|_| json!({ "raw": response_text }))
}

fn extract_output_text(response: &Value) -> String {
    response
        .get("output")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter(|item| item.get("type").and_then(Value::as_str) == Some("message")).map(extract_message_item_text).filter(|text| !text.is_empty()).collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

fn extract_message_item_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter(|content| content.get("type").and_then(Value::as_str) == Some("output_text")).filter_map(|content| content.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

fn arg_usize(args: &Value, key: &str) -> Option<usize> {
    args.get(key).and_then(Value::as_u64).and_then(|value| usize::try_from(value).ok())
}

fn validate_date(date: &str) -> Result<(), String> {
    NaiveDate::parse_from_str(date, "%Y-%m-%d").map(|_| ()).map_err(|_| format!("invalid date {date:?}; expected YYYY-MM-DD"))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut iter = value.chars();
    let truncated = iter.by_ref().take(max_chars).collect::<String>();
    if iter.next().is_some() { format!("{truncated}… [truncated]") } else { truncated }
}

fn parse_media_fields(body: &str) -> Option<Value> {
    if !body.starts_with("media ") { return None; }
    let mut fields = serde_json::Map::new();
    for key in ["path", "timestamp", "mime", "caption", "tags"] {
        if let Some(value) = extract_quoted_field(body, key) {
            fields.insert(key.to_string(), json!(value));
        }
    }
    Some(Value::Object(fields))
}

fn extract_quoted_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = body.find(&needle)? + needle.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in body[start..].chars() {
        if escaped { out.push(ch); escaped = false; continue; }
        match ch { '\\' => escaped = true, '"' => return Some(out), _ => out.push(ch) }
    }
    None
}
