<h1 align="center">🍪 Examples & Recipes</h1>

<h3 align="center">Customize your EvoScientist — harness it, make it yours.</h3>

| Example                                                     | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Survey literature](https://github.com/EvoScientist/EvoScientist/tree/main/docs/examples/survey-literature#literature-survey)   | Run EvoScientist with the `paper-navigator` skill to produce a bilingual, conference-grade literature survey |


| Recipe                                                     | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [macOS 24/7 Deployment](https://github.com/EvoScientist/EvoScientist/blob/main/docs/recipes/deployment-macos-24h.md#running-evoscientist-247-on-macos-telegram-bot--stt--ccproxy)   | Run EvoScientist as an always-on service on macOS with OAuth + Telegram + STT   |
| [Subscription sign-in (Claude OAuth + ChatGPT)](https://github.com/EvoScientist/EvoScientist/blob/main/docs/recipes/subscription-oauth.md#using-evoscientist-with-your-ai-subscriptions-claude-oauth--chatgpt-sign-in)   | Use Claude and GPT models through your existing subscriptions — no API keys; Claude via ccproxy, ChatGPT by direct sign-in   |


| Guide                                                      | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`stream-json` output protocol](https://github.com/EvoScientist/EvoScientist/blob/main/docs/guides/stream-json.md#stream-json-output-protocol)   | Line-delimited JSON event stream (`--output-format stream-json`) for driving EvoScientist headlessly from SDK / programmatic clients |
| [Grading a run against a rubric](https://github.com/EvoScientist/EvoScientist/blob/main/docs/guides/rubric.md#grading-a-run-against-a-rubric)   | Acceptance criteria for single-shot runs (`--rubric`) and for threads in the TUI / interactive CLI (`/rubric`): what the grader checks, when grading is deferred, exit codes and limits |
| [Retrying transient model failures](https://github.com/EvoScientist/EvoScientist/blob/main/docs/guides/model-retry.md#retrying-transient-model-failures) | What a transient provider failure is, the per-call wait budget for interactive and unattended runs (`model_retry_wait_seconds`, `auto_mode_retry_wait_seconds`), and what each surface shows |

## Contributing a Recipe

See the [Contributing Guide](../CONTRIBUTING.md) for general guidelines. When adding a new recipe:

- **Use `EvoSci` CLI** — recipes should work with `EvoSci serve`, `EvoSci config`, or `EvoSci onboard`
- **Pin dependencies** — specify EvoScientist extras (e.g., `pip install -e ".[telegram,stt]"`)
- **Include a README** with clear setup and usage instructions
- **Keep it focused** — each recipe should demonstrate one deployment or integration scenario
- **Add to the table** above so others can discover it
