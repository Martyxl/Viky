# tools/

Tool implementations invoked by the brain via tool calling (**M5**).

- `trigger_n8n_workflow`, `get_trading_stats`, `send_email`,
  `list_agents` / `run_agent`, `get_time`.
- Every tool honours `VIKY_DRY_RUN=true` (default in dev): it logs the intended
  call instead of executing it.
- `mock_stats.py` — local mock of the tradezer.app stats backend for dev.
