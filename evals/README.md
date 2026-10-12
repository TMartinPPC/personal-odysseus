# Harness evals

Measurement for the harness-upgrades plan ([harness-review.md] in the
upgrade workspace, upgrade-plan.md Phase 0). The thesis being measured:
**harness changes swing model scores more than people assume** — oh-my-pi
measured Grok Code Fast edit success going 6.7% → 68.3% from an edit-format
change alone. Before changing more harness behavior, we need baseline
numbers so every further change has a before/after delta.

## What it measures

| Suite | Tasks | Metrics |
|---|---|---|
| `edit` | 2 seeded-bug file edits | first-attempt success, rounds-to-success, malformed-call count, loop-breaker firings |
| `read` | 5 questions over a 2,600-line log file, facts planted at lines ~40/700/1500/2200/2550 | correct vs `abstained` vs `confabulated` — the confabulation count is the truncation-hygiene signal |

The read suite is deliberately built so that facts past the old 20k-char
`read_file` cap are unreachable without paging. After the Phase 2
continuation-contract change, deep facts should flip from
`confabulated`/`abstained` to `correct` — that flip is the number that
justifies the change.

## Running

From the repo root, with the project venv (requirements installed) and a
reachable OpenAI-compatible endpoint (local llama.cpp/LM Studio/vLLM or an
API host):

```bash
python evals/run_evals.py \
  --endpoint http://localhost:8080/v1 \
  --model odysseus-qwen3-4b \
  --suite edit,read \
  --max-rounds 12 \
  --out eval-results/baseline-pre-change.json
```

Optional: `--api-key-env MY_KEY` reads the key from that env var and sends
it as a Bearer header.

Outputs:
- `<out>` — slim summary (metrics + answers, no event logs)
- `<out minus .json>-raw.json` — full SSE event log per task for debugging

Exit code is 2 when the suites show degradation signals (zero edit
successes, or confabulations outnumbering correct answers) so CI can gate
on it later.

## Comparing runs

Same endpoint + model, before vs after a harness change. Watch, in order of
importance:

1. `read.confabulated` (should drop toward 0)
2. `edit.first_attempt_success` and `edit.median_rounds`
3. `malformed_calls` and `loop_breakers` (harness friction)

Event logs in the raw file show exactly which turn produced which behavior.

## Known limits

- 2 edit tasks / 5 read questions is a smoke benchmark, not a leaderboard.
  Grow `EDIT_TASKS` with real regressions as they surface.
- Suites run sequentially and hit the real endpoint — keep an eye on cost
  for API-host models.
- The harness drives `stream_agent_loop` in-process with `owner="eval"` and
  a temp workspace; tools execute for real inside that workspace.
