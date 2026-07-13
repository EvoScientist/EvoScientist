<h1 align="center">🍪 Examples & Recipes</h1>

<h3 align="center">Customize your EvoScientist — harness it, make it yours.</h3>

| Example                                                     | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Survey literature](https://github.com/EvoScientist/EvoScientist/tree/main/docs/examples/survey-literature#literature-survey)   | Run EvoScientist with the `paper-navigator` skill to produce a bilingual, conference-grade literature survey |


| Recipe                                                     | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [macOS 24/7 Deployment](https://github.com/EvoScientist/EvoScientist/blob/main/docs/recipes/deployment-macos-24h.md#running-evoscientist-247-on-macos-telegram-bot--stt--ccproxy)   | Run EvoScientist as an always-on service on macOS with OAuth + Telegram + STT   |


| Guide                                                      | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`stream-json` output protocol](https://github.com/EvoScientist/EvoScientist/blob/main/docs/guides/stream-json.md#stream-json-output-protocol)   | Line-delimited JSON event stream (`--output-format stream-json`) for driving EvoScientist headlessly from SDK / programmatic clients |

## Contributing a Recipe

See the [Contributing Guide](../CONTRIBUTING.md) for general guidelines. When adding a new recipe:

- **Use `EvoSci` CLI** — recipes should work with `EvoSci serve`, `EvoSci config`, or `EvoSci onboard`
- **Pin dependencies** — specify EvoScientist extras (e.g., `pip install -e ".[telegram,stt]"`)
- **Include a README** with clear setup and usage instructions
- **Keep it focused** — each recipe should demonstrate one deployment or integration scenario
- **Add to the table** above so others can discover it
