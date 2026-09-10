"use node";

/**
 * TEMPORARY diagnostic action — deleted after verification.
 * Isolates which aiProvider request parameter breaks gemini-3.6-flash.
 */
import { action } from "./_generated/server";

const ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent";

async function callGemini(key: string, body: unknown): Promise<string> {
  try {
    const res = await fetch(`${ENDPOINT}?key=${encodeURIComponent(key)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(40_000),
    });
    const text = await res.text();
    return `HTTP ${res.status}: ${text.slice(0, 220).replace(/\s+/g, " ")}`;
  } catch (err) {
    return `ERR: ${err instanceof Error ? err.message : String(err)}`;
  }
}

export const diag = action({
  args: {},
  handler: async () => {
    const key = process.env.GEMINI_API_KEY ?? "";
    const base = {
      contents: [{ role: "user", parts: [{ text: "Say OK" }] }],
    };
    return {
      a_probe_basic: await callGemini(key, {
        ...base,
        generationConfig: { maxOutputTokens: 5 },
      }),
      b_system_instruction: await callGemini(key, {
        system_instruction: { parts: [{ text: "Reply with exactly one word." }] },
        ...base,
        generationConfig: { maxOutputTokens: 5 },
      }),
      c_temp: await callGemini(key, {
        ...base,
        generationConfig: { temperature: 0, maxOutputTokens: 5 },
      }),
      d_json_mode: await callGemini(key, {
        ...base,
        generationConfig: { maxOutputTokens: 5, responseMimeType: "application/json" },
      }),
      e_full_provider: await callGemini(key, {
        system_instruction: { parts: [{ text: "Reply with exactly one word." }] },
        ...base,
        generationConfig: { temperature: 0, maxOutputTokens: 5, responseMimeType: "application/json" },
      }),
    };
  },
});
