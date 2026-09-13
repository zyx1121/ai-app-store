# {{name}}

{{description}}

Scaffolded from the AI App Store `fastapi` template. The manifest is
`aias.yaml`: the platform reads it, downloads the model it declares, creates a
Postgres database, applies `db/migrations` and starts the app on a loopback
port it chooses.

## What the platform injects

| Variable | Value |
|---|---|
| `PORT`, `HOST`, `HOSTNAME` | The loopback address and port the app must bind |
| `AIAS_MODEL_CHAT_URL` | Base URL of the model instance, OpenAI compatible |
| `AIAS_MODEL_CHAT_ID` | Model name to put in the request body |
| `AIAS_MODEL_CHAT_KEY` | Bearer token of that instance |
| `DATABASE_URL` | DSN of this app's own database |

Nothing else reaches the app, and the app never starts a model itself.

`start` is an argv array, so `$PORT` cannot appear in it: `main.py` reads the
variable and hands it to uvicorn.

## Run it

```
aias apps build <this directory>
aias apps run <this directory>
```
