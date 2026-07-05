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

To see what the harness is doing while it runs, add `--verbose` or `-v`:

```bash
OPENAI_API_KEY=sk-... cargo run --bin log_llm -- \
  --verbose \
  --goal "summarise what I logged yesterday"
```

Verbose mode prints progress to stderr, including the selected model, parsed log count, each model step, tool calls, parsed arguments, and truncated tool results.

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
