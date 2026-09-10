"use node";

/**
 * Deployment health check — reports which AI/database env vars are set in the
 * Convex deployment environment (booleans only, never values) and, when a
 * provider is configured, runs a bounded live probe so we can tell whether
 * AI is actually working end-to-end.
 *
 * Temporary diagnostic — safe to delete once deployment is verified.
 */

import { action } from "./_generated/server";

const ENV_KEYS = [
  "GROQ_API_KEY",
  "GROQ_API_KEY_1",
  "GROQ_API_KEY_2",
  "GROQ_MODEL",
  "GEMINI_API_KEY",
  "GEMINI_API_KEY_1",
  "GEMINI_API_KEY_2",
  "GEMINI_API_KEY_3",
  "GEMINI_MODEL",
  "OPENROUTER_API_KEY",
  "OPENROUTER_API_KEY_1",
  "OPENROUTER_API_KEY_2",
  "OPENROUTER_MODEL",
  "VLY_INTEGRATION_KEY",
  "VLY_MODEL",
  "SUPABASE_URL",
  "SUPABASE_ANON_KEY",
  "SITE_URL",
  "JWKS",
] as const;

export const envCheck = action({
  args: {},
  handler: async () => {
    const env: Record<string, boolean> = {};
    for (const key of ENV_KEYS) env[key] = Boolean(process.env[key]);
    return { env };
  },
});

/** Probes Groq with a tiny request (bounded, 25s). */
export const probeGroq = action({
  args: {},
  handler: async () => {
    const key = process.env.GROQ_API_KEY ?? process.env.GROQ_API_KEY_1;
    if (!key) return { configured: false };
    const model = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
    try {
      const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${key}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "Say OK" }],
          temperature: 0,
          max_tokens: 10,
        }),
        signal: AbortSignal.timeout(25_000),
      });
      const body = await res.text();
      return {
        configured: true,
        model,
        status: res.status,
        body: body.slice(0, 300).replace(/\s+/g, " "),
      };
    } catch (err) {
      return {
        configured: true,
        model,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  },
});

/** Probes Gemini with a tiny request (bounded, 25s). */
export const 

[FILE_TOO_LARGE]: The combined read_files output exceeded the 100,000 character hard limit. This file was truncated after 2,233 characters. Read it separately or use code_search for the relevant section.