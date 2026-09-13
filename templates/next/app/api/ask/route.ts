import { db } from "@/lib/db";
import { chatModel } from "@/lib/model";

export const dynamic = "force-dynamic";

const SYSTEM_PROMPT = "You are a helpful assistant running locally on the user's machine. Answer briefly.";

type AskRequest = { prompt?: string };

// One prompt in, the streamed answer out, both rows kept in Postgres.
export async function POST(request: Request) {
  const body = (await request.json()) as AskRequest;
  const prompt = body.prompt?.trim();
  if (!prompt) {
    return Response.json({ error: "prompt is required" }, { status: 400 });
  }

  const sql = db();
  const { client, model } = chatModel();

  const completion = await client.chat.completions.create({
    model,
    // Sampling belongs in the request body, never in the manifest.
    temperature: 0.7,
    stream: true,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
  });

  const encoder = new TextEncoder();
  let answer = "";

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of completion) {
          const delta = chunk.choices[0]?.delta?.content ?? "";
          if (delta) {
            answer += delta;
            controller.enqueue(encoder.encode(delta));
          }
        }
      } catch (error) {
        const note = error instanceof Error ? error.message : String(error);
        controller.enqueue(encoder.encode(`\n[model error: ${note}]`));
      } finally {
        controller.close();
        await sql`INSERT INTO prompts (prompt, answer) VALUES (${prompt}, ${answer})`;
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
  });
}
