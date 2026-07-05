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

Processing tier:

- The harness sends `service_tier: "flex"` on every OpenAI Responses API request.
- This prioritizes cheaper flex processing over low latency.
- The HTTP client timeout is 900 seconds, matching OpenAI's Flex guidance for longer-running work.
- The selected tier, timeout, and retry count are printed in startup progress output and included in full request/run logs.

Normal runs print lightweight progress to stderr while they run. You will see setup details, parsed log count, step `N/max`, tool names, compact tool-result counts, retry waits, and the final log path. Full API payloads are not printed to the terminal.

Interactive input:

- You can type extra messages at any time while the harness is running.
- Messages typed during API/model/tool work are queued and inserted before the next model request.
- After each answer, the harness stays open and waits for follow-up input.
- If a run errors after retries, the harness pauses instead of exiting.
- Type `/continue` to retry from the current conversation/tool state.
- Type `/continue more context here` to add context and retry.
- Type a normal follow-up and press Enter to continue the same conversation.
- Type `/quit`, `/exit`, `:q`, `quit`, or `exit` to stop.
- Blank lines are ignored.

The model also has a `wait_for_user_input` action. It can call this when the goal is vague, ambiguous, or missing needed context. When that happens, the harness pauses, shows the model's question in the terminal, waits for your typed reply, and sends that reply back as the action result.

Conversation history is kept locally while still sending `store: false` to OpenAI. The harness does not re-send transient response item IDs from prior calls, because those items are not persisted when `store` is false. It keeps portable conversation items instead: user messages, assistant text, function calls, and function-call outputs.

Every run writes a full debug log under `./llm_logs`, even without `--verbose`. The log file includes run metadata, parsed log count, each OpenAI request payload, each OpenAI response payload, tool calls, tool arguments, tool results, follow-up messages, and wait-for-user-input replies.

The log files can contain private QuickLogger entries and model responses. The `llm_logs/.gitignore` file keeps generated `.log` files out of git.

Retry handling:

- Flex transport timeouts and connection errors are retried automatically.
- HTTP `408 Request Timeout` errors are retried automatically.
- OpenAI `rate_limit_exceeded` errors are retried automatically.
- Flex `429 Resource Unavailable` / insufficient-resource style errors are retried automatically.
- Transient server errors are retried automatically.
- The harness first uses the `Retry-After` header when OpenAI sends one.
- If there is no header, it parses messages like `Please try again in 4.991s`.
- If neither is available, it uses exponential backoff, capped at 60 seconds.
- Retries are logged into `./llm_logs` and shown in the terminal with the chosen delay and source.
- `insufficient_quota` is not retried, because waiting will not fix missing quota or billing limits.

To see compact tool-result/error snippets in the terminal too, add `--verbose` or `-v`:

```bash
OPENAI_API_KEY=sk-... cargo run --bin log_llm -- \
  --verbose \
  --goal "summarise what I logged yesterday"
```

Verbose mode still does not print full OpenAI request/response payloads to the terminal. The file in `llm_logs` remains the place for full request/response/error payloads.

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
- `wait_for_user_input`
- `finish`

Dates passed to actions are UTC dates because QuickLogger writes entries using `Utc::now()`.
