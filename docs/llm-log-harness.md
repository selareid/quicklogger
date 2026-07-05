# LLM log harness

This branch adds a separate Rust binary for asking an OpenAI model questions about the local QuickLogger log files.

It does not change the existing web server entry point. The server still runs with:

```bash
cargo run
```

Run the LLM harness with the best low-latency general-purpose model:

```bash
OPENAI_API_KEY=sk-... cargo run --bin log_llm -- --model gpt-5.4-mini --goal "summarise what I logged yesterday"
```

Every run writes a full debug log under `./llm_logs`, even without `--verbose`. The log file includes run metadata, parsed log count, each OpenAI request payload, each OpenAI response payload, tool calls, tool arguments, and tool results.

The log files can contain private QuickLogger entries and model responses. The `llm_logs/.gitignore` file keeps generated `.log` files out of git.

Rate-limit handling:

- OpenAI `rate_limit_exceeded` errors are retried automatically.
- The harness first uses the `Retry-After` header when OpenAI sends one.
- If there is no header, it parses messages like `Please try again in 4.991s`.
- If neither is available, it uses exponential backoff, capped at 60 seconds.
- Retries are logged into `./llm_logs` with the chosen delay and source.
- `insufficient_quota` is not retried, because waiting will not fix missing quota or billing limits.

To also see progress in the terminal while it runs, add `--verbose` or `-v`:

```bash
OPENAI_API_KEY=sk-... cargo run --bin log_llm -- \
  --verbose \
  --goal "summarise what I logged yesterday"
```

Verbose mode prints progress to stderr, including the selected model, parsed log count, each model step, tool calls, parsed arguments, truncated tool results, and rate-limit retry messages. The file in `llm_logs` still contains the full request/response/error payloads.

Optional flags:

```bash
cargo run --bin log_llm -- \
  --logs ./logs \
  --model gpt-5.4-mini \
  --max-steps 30 \
  --verbose \
  --goal "find any entries about bike tyres and summarise the context"
```

Environment variables:

- `OPENAI_API_KEY`: required.
- `OPENAI_MODEL`: optional model override. Defaults to `gpt-5.4-mini`.
- `QUICKLOGGER_LOGS_PATH`: optional logs directory override. Defaults to `./logs`.

The harness parses the existing `./logs` files, then lets the model inspect them through explicit actions rather than dumping the whole log corpus into one prompt.

Available actions include:

- `get_log_summary`
- `get_log_entry_by_index`
- `get_previous_log_entry`
- `get_next_log_entry`
- `get_log_entries_for_day`
- `get_log_entries_between`
- `search_log_entries`
- `get_log_entries_around`
- `read_notes`
- `write_notes`
- `finish`

Dates passed to actions are UTC dates because QuickLogger writes entries using `Utc::now()`.
