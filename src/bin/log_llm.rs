use chrono::{Datelike, LocalResult, NaiveDate, TimeZone, Utc};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    cmp::min,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const DEFAULT_LOGS_PATH: &str = "./logs";
const DEFAULT_MODEL: &str = "gpt-5.5";
const DEFAULT_MAX_STEPS: usize = 30;
const MAX_ENTRY_CHARS: usize = 1_500;
const MAX_RESULT_ENTRIES: usize = 100;

type AppResult<T> = Result<T, Box<dyn Error>>;

fn main() -> AppResult<()> {
    let config = Config::from_args()?;
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY must be set to use the log LLM harness")?;

    let entries = load_log_entries(&config.logs_path)?;
    if entries.is_empty() {
        return Err(format!(
            "No log entries found under {}",
            config.logs_path.display()
        )
        .into());
    }

    let client = OpenAiClient::new(api_key)?;
    let mut harness = Harness::new(entries);
    let answer = harness.run(&client, &config.goal, &config.model, config.max_steps)?;

    println!("\n=== Answer ===\n{answer}");
    println!("\n=== Notes ===\n{}", harness.notes.trim());

    Ok(())
}

struct Config {
    goal: String,
    logs_path: PathBuf,
    model: String,
    max_steps: usize,
}

impl Config {
    fn from_args() -> AppResult<Self> {
        let mut goal: Option<String> = None;
        let mut logs_path = env::var("QUICKLOGGER_LOGS_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOGS_PATH));
        let mut model = env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut max_steps = DEFAULT_MAX_STEPS;
        let mut positional_goal = Vec::new();

        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--goal" | "-g" => goal = Some(next_arg(&mut args, "--goal")?),
                "--logs" | "-l" => logs_path = PathBuf::from(next_arg(&mut args, "--logs")?),
                "--model" | "-m" => model = next_arg(&mut args, "--model")?,
                "--max-steps" => {
                    max_steps = next_arg(&mut args, "--max-steps")?.parse()?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => positional_goal.push(other.to_string()),
            }
        }

        let goal = goal
            .or_else(|| {
                if positional_goal.is_empty() {
                    None
                } else {
                    Some(positional_goal.join(" "))
                }
            })
            .ok_or("Pass a goal with --goal \"...\" or as positional text")?;

        Ok(Self {
            goal,
            logs_path,
            model,
            max_steps,
        })
    }
}

fn next_arg(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> AppResult<String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run --bin log_llm -- --goal \"summarise my logs from yesterday\" [--logs ./logs] [--model gpt-5.5] [--max-steps 30]\n\nEnvironment:\n  OPENAI_API_KEY          required\n  OPENAI_MODEL            optional default model\n  QUICKLOGGER_LOGS_PATH   optional default log directory"
    );
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

fn load_log_entries(logs_root: &Path) -> AppResult<Vec<LogEntry>> {
    let mut parsed = Vec::new();

    for entry in fs::read_dir(logs_root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let file_name = entry.file_name().to_string_lossy().to_string();
        let contents = fs::read_to_string(&path)?;
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

#[derive(Debug)]
struct PartialLogEntry {
    timestamp: i64,
    date_utc: String,
    file_name: String,
    prefix: String,
    body: String,
    raw: String,
}

fn parse_log_line(line: &str, file_name: &str) -> Option<PartialLogEntry> {
    let (prefix, body) = line.split_once(": ")?;
    let timestamp = prefix.split_whitespace().next()?.parse::<i64>().ok()?;
    let date_utc = date_from_timestamp(timestamp)?;

    Some(PartialLogEntry {
        timestamp,
        date_utc,
        file_name: file_name.to_string(),
        prefix: prefix.to_string(),
        body: body.to_string(),
        raw: line.to_string(),
    })
}

fn date_from_timestamp(timestamp: i64) -> Option<String> {
    match Utc.timestamp_opt(timestamp, 0) {
        LocalResult::Single(dt) => Some(format!(
            "{:04}-{:02}-{:02}",
            dt.year(),
            dt.month(),
            dt.day()
        )),
        _ => None,
    }
}

struct OpenAiClient {
    api_key: String,
    http: Client,
}

impl OpenAiClient {
    fn new(api_key: String) -> AppResult<Self> {
        Ok(Self {
            api_key,
            http: Client::builder().timeout(Duration::from_secs(120)).build()?,
        })
    }

    fn create_response(&self, request_body: Value) -> AppResult<Value> {
        let response = self
            .http
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()?;

        let status = response.status();
        let text = response.text()?;
        if !status.is_success() {
            return Err(format!("OpenAI API error {status}: {text}").into());
        }

        Ok(serde_json::from_str(&text)?)
    }
}

struct Harness {
    entries: Vec<LogEntry>,
    cursor_index: Option<usize>,
    notes: String,
}

impl Harness {
    fn new(entries: Vec<LogEntry>) -> Self {
        Self {
            entries,
            cursor_index: None,
            notes: String::new(),
        }
    }

    fn run(
        &mut self,
        client: &OpenAiClient,
        goal: &str,
        model: &str,
        max_steps: usize,
    ) -> AppResult<String> {
        let mut input_items = vec![json!({
            "role": "user",
            "content": format!(
                "Goal: {goal}\n\nUse the available log-reading actions to inspect the QuickLogger log data. Keep useful intermediate findings in the notes object. Do not claim a fact from the logs unless you saw it in an action result. When done, call finish with the final answer. Dates in tool arguments are UTC dates because QuickLogger writes log timestamps with Utc::now()."
            )
        })];

        for step in 0..max_steps {
            let request_body = json!({
                "model": model,
                "input": input_items,
                "instructions": self.instructions(),
                "tools": self.tools(),
                "parallel_tool_calls": false,
                "store": false,
            });

            let response = client.create_response(request_body)?;
            let output = response
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            let mut tool_calls = Vec::new();
            for item in &output {
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    tool_calls.push(item.clone());
                }
            }

            for item in output {
                input_items.push(item);
            }

            if tool_calls.is_empty() {
                let text = extract_output_text(&response);
                if text.trim().is_empty() {
                    return Err(format!("Model stopped without output after step {step}").into());
                }
                return Ok(text);
            }

            for call in tool_calls {
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("Function call missing name")?;
                let call_id = call
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("Function call missing call_id")?;
                let arguments = call
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));

                if name == "finish" {
                    let answer = args
                        .get("answer")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if answer.is_empty() {
                        return Err("finish called without an answer".into());
                    }
                    return Ok(answer);
                }

                let result = self.call_tool(name, &args);
                input_items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result.to_string(),
                }));
            }
        }

        Err(format!("Reached --max-steps ({max_steps}) before the model called finish").into())
    }

    fn instructions(&self) -> String {
        format!(
            "You are an analysis harness for a personal QuickLogger log corpus. There are {} parsed entries. Prefer targeted tool calls over asking to dump everything. Use get_log_summary first unless the goal already names an exact date or search term. Maintain concise notes as you learn facts. Use get_log_entries_for_day for daily questions, search_log_entries for keywords, get_log_entries_around for context, and previous/next actions for local navigation. Finish with a direct answer and mention uncertainty or missing evidence when relevant.",
            self.entries.len()
        )
    }

    fn tools(&self) -> Vec<Value> {
        vec![
            json!({
                "type": "function",
                "name": "get_log_summary",
                "description": "Return corpus metadata: entry count, first/last entry, date range, and counts per UTC day.",
                "parameters": { "type": "object", "properties": {}, "required": [], "additionalProperties": false }
            }),
            json!({
                "type": "function",
                "name": "get_log_entry_by_index",
                "description": "Read one log entry by its zero-based global index. Sets the navigation cursor to that entry.",
                "parameters": {
                    "type": "object",
                    "properties": { "index": { "type": "integer", "minimum": 0 } },
                    "required": ["index"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "get_previous_log_entry",
                "description": "Read the log entry immediately before from_index, or before the current cursor if from_index is omitted. With no cursor, returns the last entry.",
                "parameters": {
                    "type": "object",
                    "properties": { "from_index": { "type": "integer", "minimum": 0 } },
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "get_next_log_entry",
                "description": "Read the log entry immediately after from_index, or after the current cursor if from_index is omitted. With no cursor, returns the first entry.",
                "parameters": {
                    "type": "object",
                    "properties": { "from_index": { "type": "integer", "minimum": 0 } },
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "get_log_entries_for_day",
                "description": "Return log entries for a UTC day. Use limit and offset for large days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": { "type": "string", "description": "UTC date in YYYY-MM-DD format" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                        "offset": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["date"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "get_log_entries_between",
                "description": "Return log entries whose UTC date lies in the inclusive date range. Use limit and offset for large ranges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": { "type": "string", "description": "UTC date in YYYY-MM-DD format" },
                        "end_date": { "type": "string", "description": "UTC date in YYYY-MM-DD format" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                        "offset": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "search_log_entries",
                "description": "Substring search over log bodies. Returns matching entries in chronological order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "case_sensitive": { "type": "boolean" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                        "offset": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "get_log_entries_around",
                "description": "Return entries around a center index for chronological context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "center_index": { "type": "integer", "minimum": 0 },
                        "before": { "type": "integer", "minimum": 0, "maximum": 50 },
                        "after": { "type": "integer", "minimum": 0, "maximum": 50 }
                    },
                    "required": ["center_index"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "read_notes",
                "description": "Read the current notes object.",
                "parameters": { "type": "object", "properties": {}, "required": [], "additionalProperties": false }
            }),
            json!({
                "type": "function",
                "name": "write_notes",
                "description": "Append to or replace the notes object with concise findings and hypotheses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": { "type": "string", "enum": ["append", "replace"] },
                        "text": { "type": "string" }
                    },
                    "required": ["mode", "text"],
                    "additionalProperties": false
                }
            }),
            json!({
                "type": "function",
                "name": "finish",
                "description": "Finish the run with a final answer to the original goal.",
                "parameters": {
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }),
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
            "read_notes" => json!({ "notes": self.notes }),
            "write_notes" => self.write_notes(args),
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }

    fn get_log_summary(&self) -> Value {
        let first = self.entries.first().map(|entry| self.entry_json(entry));
        let last = self.entries.last().map(|entry| self.entry_json(entry));
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
            "first_entry": first,
            "last_entry": last,
            "counts_by_day": counts_by_day.into_iter().map(|(date, count)| json!({ "date": date, "count": count })).collect::<Vec<_>>()
        })
    }

    fn get_log_entry_by_index(&mut self, args: &Value) -> Value {
        let Some(index) = arg_usize(args, "index") else {
            return json!({ "error": "index is required" });
        };

        match self.entries.get(index).cloned() {
            Some(entry) => {
                self.cursor_index = Some(index);
                json!({ "entry": self.entry_json(&entry), "cursor_index": self.cursor_index })
            }
            None => json!({ "error": "index out of range", "entry_count": self.entries.len() }),
        }
    }

    fn get_previous_log_entry(&mut self, args: &Value) -> Value {
        if self.entries.is_empty() {
            return json!({ "error": "no entries loaded" });
        }

        let base = arg_usize(args, "from_index")
            .or(self.cursor_index)
            .unwrap_or(self.entries.len());
        if base == 0 {
            return json!({ "error": "already at first entry", "cursor_index": self.cursor_index });
        }

        let index = min(base, self.entries.len()) - 1;
        self.cursor_index = Some(index);
        json!({ "entry": self.entry_json(&self.entries[index]), "cursor_index": self.cursor_index })
    }

    fn get_next_log_entry(&mut self, args: &Value) -> Value {
        if self.entries.is_empty() {
            return json!({ "error": "no entries loaded" });
        }

        let base = arg_usize(args, "from_index")
            .or(self.cursor_index)
            .and_then(|value| value.checked_add(1))
            .unwrap_or(0);

        if base >= self.entries.len() {
            return json!({ "error": "already at last entry", "cursor_index": self.cursor_index });
        }

        self.cursor_index = Some(base);
        json!({ "entry": self.entry_json(&self.entries[base]), "cursor_index": self.cursor_index })
    }

    fn get_log_entries_for_day(&mut self, args: &Value) -> Value {
        let Some(date) = args.get("date").and_then(Value::as_str) else {
            return json!({ "error": "date is required" });
        };
        if let Err(error) = validate_date(date) {
            return json!({ "error": error });
        }

        let indices = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.date_utc == date).then_some(index))
            .collect::<Vec<_>>();
        self.page_entries(indices, args)
    }

    fn get_log_entries_between(&mut self, args: &Value) -> Value {
        let Some(start_date) = args.get("start_date").and_then(Value::as_str) else {
            return json!({ "error": "start_date is required" });
        };
        let Some(end_date) = args.get("end_date").and_then(Value::as_str) else {
            return json!({ "error": "end_date is required" });
        };
        if let Err(error) = validate_date(start_date) {
            return json!({ "error": error });
        }
        if let Err(error) = validate_date(end_date) {
            return json!({ "error": error });
        }
        if start_date > end_date {
            return json!({ "error": "start_date must be <= end_date" });
        }

        let indices = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                (entry.date_utc.as_str() >= start_date && entry.date_utc.as_str() <= end_date)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        self.page_entries(indices, args)
    }

    fn search_log_entries(&mut self, args: &Value) -> Value {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return json!({ "error": "query is required" });
        };
        if query.trim().is_empty() {
            return json!({ "error": "query must not be empty" });
        }

        let case_sensitive = args
            .get("case_sensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let query_normalized = if case_sensitive {
            query.to_string()
        } else {
            query.to_lowercase()
        };

        let indices = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let haystack = if case_sensitive {
                    entry.body.clone()
                } else {
                    entry.body.to_lowercase()
                };
                haystack.contains(&query_normalized).then_some(index)
            })
            .collect::<Vec<_>>();

        self.page_entries(indices, args)
    }

    fn get_log_entries_around(&mut self, args: &Value) -> Value {
        let Some(center_index) = arg_usize(args, "center_index") else {
            return json!({ "error": "center_index is required" });
        };
        if center_index >= self.entries.len() {
            return json!({ "error": "center_index out of range", "entry_count": self.entries.len() });
        }

        let before = arg_usize(args, "before").unwrap_or(5).min(50);
        let after = arg_usize(args, "after").unwrap_or(5).min(50);
        let start = center_index.saturating_sub(before);
        let end = min(self.entries.len(), center_index + after + 1);
        self.cursor_index = Some(center_index);

        json!({
            "start_index": start,
            "end_index_exclusive": end,
            "cursor_index": self.cursor_index,
            "entries": self.entries[start..end]
                .iter()
                .map(|entry| self.entry_json(entry))
                .collect::<Vec<_>>()
        })
    }

    fn page_entries(&mut self, indices: Vec<usize>, args: &Value) -> Value {
        let total = indices.len();
        let offset = arg_usize(args, "offset").unwrap_or(0);
        let limit = arg_usize(args, "limit").unwrap_or(25).min(MAX_RESULT_ENTRIES);

        let entries = indices
            .iter()
            .skip(offset)
            .take(limit)
            .filter_map(|index| self.entries.get(*index))
            .map(|entry| self.entry_json(entry))
            .collect::<Vec<_>>();

        if let Some(first) = entries.first() {
            self.cursor_index = first.get("index").and_then(Value::as_u64).map(|v| v as usize);
        }

        json!({
            "total_matches": total,
            "offset": offset,
            "limit": limit,
            "returned": entries.len(),
            "cursor_index": self.cursor_index,
            "entries": entries
        })
    }

    fn write_notes(&mut self, args: &Value) -> Value {
        let mode = args
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("append");
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");

        match mode {
            "replace" => self.notes = text.to_string(),
            "append" => {
                if !self.notes.trim().is_empty() && !text.trim().is_empty() {
                    self.notes.push('\n');
                }
                self.notes.push_str(text);
            }
            _ => return json!({ "error": "mode must be append or replace" }),
        }

        json!({ "ok": true, "notes": self.notes })
    }

    fn entry_json(&self, entry: &LogEntry) -> Value {
        let mut value = json!({
            "index": entry.index,
            "timestamp": entry.timestamp,
            "date_utc": entry.date_utc,
            "file": entry.file_name,
            "prefix": entry.prefix,
            "body": truncate_chars(&entry.body, MAX_ENTRY_CHARS),
            "raw": truncate_chars(&entry.raw, MAX_ENTRY_CHARS),
        });

        if let Some(fields) = parse_media_fields(&entry.body) {
            value["media"] = fields;
        }

        value
    }
}

fn extract_output_text(response: &Value) -> String {
    let mut text = String::new();
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return text;
    };

    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for content_item in content {
            if content_item.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(part) = content_item.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(part);
                }
            }
        }
    }

    text
}

fn arg_usize(args: &Value, key: &str) -> Option<usize> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn validate_date(date: &str) -> Result<(), String> {
    NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map(|_| ())
        .map_err(|_| format!("invalid date {date:?}; expected YYYY-MM-DD"))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut iter = value.chars();
    let truncated = iter.by_ref().take(max_chars).collect::<String>();
    if iter.next().is_some() {
        format!("{truncated}… [truncated]")
    } else {
        truncated
    }
}

fn parse_media_fields(body: &str) -> Option<Value> {
    if !body.starts_with("media ") {
        return None;
    }

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
    let rest = &body[start..];
    let mut out = String::new();
    let mut escaped = false;

    for ch in rest.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some(out),
            _ => out.push(ch),
        }
    }

    None
}
