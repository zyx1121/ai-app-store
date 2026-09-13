"""One prompt in, the streamed answer out, both rows kept in Postgres."""

import os

import psycopg
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

SYSTEM_PROMPT = "You are a helpful assistant running locally on the user's machine. Answer briefly."

app = FastAPI(title="{{name}}")


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. The platform injects it when the app declares services.postgres.")
    return url


def chat_model() -> tuple[OpenAI, str]:
    """The platform injects one group of variables per declared alias."""
    base_url = os.environ.get("AIAS_MODEL_CHAT_URL")
    model = os.environ.get("AIAS_MODEL_CHAT_ID")
    if not base_url or not model:
        raise RuntimeError("AIAS_MODEL_CHAT_URL and AIAS_MODEL_CHAT_ID must be set by the platform.")
    # The instance is started with an api key, so a request without the bearer
    # token is a 401. Loopback is not a permission.
    return OpenAI(base_url=base_url, api_key=os.environ.get("AIAS_MODEL_CHAT_KEY") or "none"), model


class Ask(BaseModel):
    prompt: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    """The platform polls this until it answers 2xx, then calls the app started."""
    try:
        with psycopg.connect(database_url()) as connection:
            connection.execute("SELECT 1")
        return JSONResponse({"ok": True})
    except Exception as error:  # noqa: BLE001 - the platform only needs the reason
        return JSONResponse({"ok": False, "error": str(error)}, status_code=503)


@app.post("/api/ask")
def ask(body: Ask) -> StreamingResponse:
    prompt = body.prompt.strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    client, model = chat_model()
    completion = client.chat.completions.create(
        model=model,
        # Sampling belongs in the request body, never in the manifest.
        temperature=0.7,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    def stream():
        answer = ""
        try:
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    answer += delta
                    yield delta
        except Exception as error:  # noqa: BLE001 - shown to the user as text
            yield f"\n[model error: {error}]"
        finally:
            with psycopg.connect(database_url()) as connection:
                connection.execute(
                    "INSERT INTO prompts (prompt, answer) VALUES (%s, %s)",
                    (prompt, answer),
                )

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
