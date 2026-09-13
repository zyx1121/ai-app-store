"use client";

import { useState } from "react";

// Written by `aias apps init` from the manifest.
const NAME = "{{name}}";
const DESCRIPTION = "{{description}}";

// One prompt, one streamed answer. The key never reaches the browser: the
// request goes to /api/ask, which is the only place that talks to the model.
export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setAnswer("");

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!response.body) throw new Error(`the app answered ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        setAnswer((current) => current + decoder.decode(value, { stream: true }));
      }
    } catch (error) {
      setAnswer(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>{NAME}</h1>
      <p className="muted">{DESCRIPTION}</p>
      <form onSubmit={ask}>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Ask the local model something"
          rows={4}
        />
        <button type="submit" disabled={busy || prompt.trim().length === 0}>
          {busy ? "Answering" : "Ask"}
        </button>
      </form>
      <pre>{answer}</pre>
    </main>
  );
}
