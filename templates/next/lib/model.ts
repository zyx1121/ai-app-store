import OpenAI from "openai";

// The platform injects one group of variables per alias declared in aias.yaml.
// This app declares a single alias: chat.
const baseURL = process.env.AIAS_MODEL_CHAT_URL;
const model = process.env.AIAS_MODEL_CHAT_ID;
const apiKey = process.env.AIAS_MODEL_CHAT_KEY;

export function chatModel() {
  if (!baseURL || !model) {
    throw new Error("AIAS_MODEL_CHAT_URL and AIAS_MODEL_CHAT_ID must be set by the platform.");
  }

  // The instance listens on loopback, which every process on the machine can
  // reach, so it is started with an api key and answers 401 without the bearer
  // token. The platform hands that token over as AIAS_MODEL_CHAT_KEY. The
  // OpenAI SDK insists on a non empty string, so "none" stands in for one when
  // the server is unguarded.
  const client = new OpenAI({ baseURL, apiKey: apiKey || "none" });
  return { client, model };
}
