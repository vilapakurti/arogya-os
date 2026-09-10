"use node";

/**
 * Shared AI provider gateway with automatic fallback (ArogyaOS).
 *
 * Every AI feature (report analysis, Health Baseline briefing, Doctor Copilot
 * brief, Ask Doctor Copilot chat) routes its LLM call through this ONE module:
 *
 *   AI Request
 *        ↓
 *   ArogyaOS AI Gateway
 *        ↓
 *   1. Groq        (primary — GROQ_API_KEY / GROQ_API_KEY_1..N)
 *        ↓ on retryable/key failure
 *   2. Gemini      (secondary — GEMINI_API_KEY / GEMINI_API_KEY_1..N)
 *        ↓ on retryable/key failure
 *   3. OpenRouter  (tertiary — OPENROUTER_API_KEY / OPENROUTER_API_KEY_1..N)
 *        ↓ on retryable failure
 *   4. Platform AI gateway (VLY — VLY_INTEGRATION_KEY, auto-billed)
 *        ↓
 *   normalized response to the application
 *
 * Multiple keys per provider are supported (GROQ_API_KEY_1, GROQ_API_KEY_2,
 * GEMINI_API_KEY_1..3, OPENROUTER_API_KEY_1..2). Key rotation is ONLY a
 * fallback for key/provider-specific errors (invalid/revoked key 401/403,
 * rate limit 429) — it does NOT try to multiply provider quota. Provider-level
 * failures (5xx, timeouts, network, model errors) skip straight to the next
 * provider. The bare `*_API_KEY` name is still honored for backward
 * compatibility; numbered keys are tried in ascending order after it.
 *
 * Retry policy: at most MAX_RETRIES (2) retries per key attempt with
 * exponential backoff + jitter, and ONLY for transient errors (429, 5xx).
 * Errors that indicate invalid configuration or malformed requests (401/403,
 * 400 model errors) are never retried as-is.
 *
 * All providers are normalized into the exact same shape before the caller
 * sees them, so the frontend never knows (or cares) which provider produced
 * the response — and no provider-specific logic leaks into the actions.
 *
 * Security: API keys are read from process.env ONLY — they live on the Convex
 * server and are never sent to the browser or included in any result.
 *
 * Environment variables (Keys tab / Convex env):
 *   GROQ_API_KEY            — Groq API key (primary). May be split across
 *                             GROQ_API_KEY_1, GROQ_API_KEY_2, ... for rotation.
 *   GROQ_MODEL              — optional, default "openai/gpt-oss-20b"
 *   GEMINI_API_KEY          — Google AI Studio API key (secondary). May be
 *                             split across GEMINI_API_KEY_1..N.
 *   GEMINI_MODEL            — optional, default "gemini-3.6-flash"
 *   OPENROUTER_API_KEY      — OpenRouter API key (tertiary). May be split
 *                             across OPENROUTER_API_KEY_1..N.
 *   OPENROUTER_MODEL        — optional, default "inclusionai/ling-3.0-flash:free"
 *   VLY_INTEGRATION_KEY     — platform AI gateway key (final fallback, `sk_*`)
 *   VLY_MODEL               — optional, default "gpt-4o-mini"
 */

export type AiProvider = "groq" | "gemini" | "openrouter" | "vly" | "cache";

/** Everything an AI action needs to make one LLM call (provider-agnostic). */
export interface AiProviderRequest {
  systemPrompt: string;
  userMessage: string;
  temperature?: number;
  maxOutputTokens?: number;
  /** Ask the provider for strict JSON output. Groq/Gemini/OpenRouter support it. */
  jsonMode?: boolean;
  timeoutMs?: number;
}

export type AiProviderErrorCode =
  | "not_configured"
  | "rate_limited"
  | "no_credits"
  | "timeout"
  | "network"
  | "server"
  | "model_error";

export interface AiProviderSuccess {
  ok: true;
  /** Raw text response — identical schema regardless of provider. */
  text: string;
  provider: AiProvider;
  /** Effective model identifier actually used (for model_used / logs). */
  model: string;
  /** Provider finish reason (e.g. "MAX_TOKENS" / "length"). */
  finishReason?: string;
}

export interface AiProviderFailure {
  ok: false;
  code: AiProviderErrorCode;
  message: string;
  /** Which provider produced this failure (undefined when none configured). */
  provider?: AiProvider;
}

const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b";
const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_MODEL = "gemini-3.6-flash";
const OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
// Default to a live `:free` model: accounts with no credits (402) fall back
// through the chain below, so starting with a free endpoint avoids the
// guaranteed credit check every time. Paid keys can override via
// OPENROUTER_MODEL.
const DEFAULT_OPENROUTER_MODEL = "inclusionai/ling-3.0-flash:free";
const DEFAULT_VLY_MODEL = "gpt-4o-mini";

/**
 * Extra OpenRouter models tried automatically when the configured model is
 * unavailable (no credits / model not found / per-model rate limit). Free
 * (`:free`) models work even on accounts that have never purchased credits.
 * The configured model is always tried first; these are only fallbacks.
 *
 * NOTE (Aug 2026): the OpenRouter free catalog changes constantly — retired
 * slugs just fail with 404 and the chain moves to the next entry.
 */
const OPENROUTER_FALLBACK_MODELS = [
  "inclusionai/ling-3.0-flash:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "google/gemma-4-31b-it:free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
  "openai/gpt-oss-20b:free",
  "poolside/laguna-s-2.1:free",
  "cohere/north-mini-code:free",
  "nvidia/nemotron-nano-9b-v2:free",
];

const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_RETRIES = 2;
const MAX_RETRY_DELAY_MS = 12_000;

/** Honest explanation for quota exhaustion (the recurring 429 root cause). */
const QUOTA_EXHAUSTED_MESSAGE =
  "The AI service's free-tier request quota is used up for today (Gemini free keys allow ~20 requests/day). " +
  "It resets daily — try again later, or add a Gemini API key with billing enabled in the Keys tab.";

/** Clear explanation for OpenRouter accounts that have never purchased credits. */
const NO_CREDITS_MESSAGE =
  "The OpenRouter account has no credits left, so paid models cannot run. " +
  "The free `:free` models below were tried automatically; if they also failed, add credits at " +
  "openrouter.ai or switch to the platform AI gateway by setting VLY_INTEGRATION_KEY in the Keys tab.";

const GROQ_RATE_LIMIT_MESSAGE =
  "Groq is rate-limited right now (free-tier limits are per-minute). " +
  "The gateway automatically tries the next key or provider.";

function fail(
  code: AiProviderErrorCode,
  message: string,
  provider?: AiProvider,
): AiProviderFailure {
  return { ok: false, code, message, provider };
}

function isTimeout(err: unknown): boolean {
  const name = err instanceof Error ? err.name : "";
  return name === "TimeoutError" || name === "AbortError" || /timeout/i.test(String(err));
}

/**
 * Collects API keys for one provider: the bare `*_API_KEY` name first (legacy
 * deployments), then the numbered variants `*_API_KEY_1..9`, deduped.
 */
function collectKeys(base: string): string[] {
  const keys: string[] = [];
  const push = (key: string | undefined): void => {
    if (key && !keys.includes(key)) keys.push(key);
  };
  push(process.env[base]);
  for (let i = 1; i <= 9; i += 1) {
    push(process.env[`${base}_${i}`]);
  }
  return keys;
}

/** True when at least one AI provider is configured. Used by every action. */
export function hasAiProvidersConfigured(): boolean {
  return (
    collectKeys("GROQ_API_KEY").length > 0 ||
    collectKeys("GEMINI_API_KEY").length > 0 ||
    collectKeys("OPENROUTER_API_KEY").length > 0 ||
    Boolean(process.env.VLY_INTEGRATION_KEY)
  );
}

/**
 * Waits using the server's own "retry in Xs" hint when present; otherwise
 * exponential backoff with jitter (capped). Never blocks forever.
 */
async function backoff(res: Response, attempt: number): Promise<void> {
  const body = await res.clone().text().catch(() => "");
  const match = body.match(/retry in ([0-9.]+)s/i);
  const serverDelay = match ? Math.round(parseFloat(match[1]) * 1000) : 0;
  const delay =
    serverDelay > 0 ? serverDelay : 1_000 * 2 ** attempt + Math.floor(Math.random() * 500);
  await new Promise((resolve) => setTimeout(resolve, Math.min(delay, MAX_RETRY_DELAY_MS)));
}

/** Key-specific failures that a different key for the same provider may fix. */
function isKeyRotatable(code: AiProviderErrorCode): boolean {
  return code === "not_configured" || code === "rate_limited";
}

/* ------------------------------------------------------------------ */
/* Groq (PRIMARY — OpenAI-compatible API)                              */
/* ------------------------------------------------------------------ */

/** Tries one Groq key with bounded retries; returns the normalized result. */
async function attemptGroqKey(
  req: AiProviderRequest,
  key: string,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const model = process.env.GROQ_MODEL || DEFAULT_GROQ_MODEL;
  const baseBody: Record<string, unknown> = {
    model,
    messages: [
      { role: "system", content: req.systemPrompt },
      { role: "user", content: req.userMessage },
    ],
    temperature: req.temperature ?? 0.3,
    max_tokens: req.maxOutputTokens ?? 2048,
  };
  const withJson = req.jsonMode
    ? { ...baseBody, response_format: { type: "json_object" } }
    : baseBody;

  // Some models reject response_format — retry the same request without it.
  const attempts: Array<Record<string, unknown>> = req.jsonMode
    ? [withJson, baseBody]
    : [baseBody];

  for (const attemptBody of attempts) {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
      let res: Response;
      try {
        res = await fetch(GROQ_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${key}`,
          },
          body: JSON.stringify(attemptBody),
          signal: AbortSignal.timeout(req.timeoutMs ?? DEFAULT_TIMEOUT_MS),
        });
      } catch (err) {
        if (isTimeout(err)) {
          return fail(
            "timeout",
            "The AI took too long to respond (Groq timed out).",
            "groq",
          );
        }
        return fail(
          "network",
          "Could not reach the AI service (Groq network error).",
          "groq",
        );
      }

      if (res.status === 429) {
        if (attempt < MAX_RETRIES) {
          await backoff(res, attempt);
          continue;
        }
        return fail("rate_limited", GROQ_RATE_LIMIT_MESSAGE, "groq");
      }
      if (res.status === 401 || res.status === 403) {
        return fail(
          "not_configured",
          "The Groq API key was rejected. Check GROQ_API_KEY (or GROQ_API_KEY_1..N) in the Keys tab.",
          "groq",
        );
      }
      if (res.status === 500 || res.status === 502 || res.status === 503 || res.status === 504) {
        if (attempt < MAX_RETRIES) {
          await backoff(res, attempt);
          continue;
        }
        return fail(
          "server",
          `The AI service returned an error (Groq HTTP ${res.status}).`,
          "groq",
        );
      }
      if (res.status === 400) {
        const errText = await res.text().catch(() => "");
        // The model may not support JSON mode — drop response_format and retry.
        if (/response_format|json.?object/i.test(errText) && attemptBody !== baseBody) {
          break;
        }
        if (/model|not found|does not exist|not supported|invalid/i.test(errText)) {
          return fail(
            "model_error",
            `The Groq model "${model}" was rejected (HTTP 400): ${errText.slice(0, 200)}`,
            "groq",
          );
        }
        return fail(
          "server",
          `Groq rejected the request (HTTP 400): ${errText.slice(0, 200)}`,
          "groq",
        );
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        return fail(
          "server",
          `Groq returned an error (HTTP ${res.status})${
            errText ? `: ${errText.slice(0, 200)}` : ""
          }.`,
          "groq",
        );
      }

      const parsed = (await res.json()) as {
        choices?: Array<{
          message?: { content?: string };
          finish_reason?: string;
        }>;
      };
      const text = parsed.choices?.[0]?.message?.content ?? "";
      if (!text.trim()) {
        return fail("server", "Groq returned an empty response.", "groq");
      }
      return {
        ok: true,
        text,
        provider: "groq",
        model,
        finishReason: parsed.choices?.[0]?.finish_reason ?? undefined,
      };
    }
  }
  return fail("server", "The AI service is temporarily unavailable (Groq).", "groq");
}

/** Tries every configured Groq key, rotating on key-specific failures only. */
async function attemptGroq(req: AiProviderRequest): Promise<AiProviderSuccess | AiProviderFailure> {
  const keys = collectKeys("GROQ_API_KEY");
  if (keys.length === 0) {
    return fail(
      "not_configured",
      "Groq is not configured. Add GROQ_API_KEY (or GROQ_API_KEY_1, GROQ_API_KEY_2, ...) in the Keys tab.",
      "groq",
    );
  }

  let lastFailure: AiProviderFailure | null = null;
  for (const key of keys) {
    const result = await attemptGroqKey(req, key);
    if (result.ok) return result;
    lastFailure = result;
    console.warn(
      `[ai-provider] Groq key attempt failed (code=${result.code}, remainingKeys=${keys.length - (keys.indexOf(key) + 1)})` +
        (isKeyRotatable(result.code) ? " — trying next Groq key." : " — moving to next provider."),
    );
    // Provider-level / request-level failures won't be fixed by another key.
    if (!isKeyRotatable(result.code)) return result;
  }
  return (
    lastFailure ??
    fail("server", "The AI service is temporarily unavailable (Groq).", "groq")
  );
}

/* ------------------------------------------------------------------ */
/* Gemini (SECONDARY)                                                  */
/* ------------------------------------------------------------------ */

/** Tries one Gemini key with bounded retries; returns the normalized result. */
async function attemptGeminiKey(
  req: AiProviderRequest,
  key: string,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const model = process.env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL;
  const url = `${GEMINI_ENDPOINT}/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`;
  const body = {
    system_instruction: { parts: [{ text: req.systemPrompt }] },
    contents: [{ role: "user", parts: [{ text: req.userMessage }] }],
    generationConfig: {
      temperature: req.temperature ?? 0.3,
      maxOutputTokens: req.maxOutputTokens ?? 2048,
      ...(req.jsonMode ? { responseMimeType: "application/json" } : {}),
    },
  };

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(req.timeoutMs ?? DEFAULT_TIMEOUT_MS),
      });
    } catch (err) {
      if (isTimeout(err)) {
        return fail(
          "timeout",
          "The AI took too long to respond (Gemini timed out).",
          "gemini",
        );
      }
      return fail(
        "network",
        "Could not reach the AI service (Gemini network error).",
        "gemini",
      );
    }

    // Transient capacity / throttling — retry with the server's own delay.
    if (res.status === 429 || res.status === 500 || res.status === 502 || res.status === 503 || res.status === 504) {
      const errText = await res.clone().text().catch(() => "");
      // Daily-quota 429s (RESOURCE_EXHAUSTED) do NOT recover within seconds —
      // skip the same-key backoff entirely so the next key/provider is tried
      // immediately instead of leaving users staring at a spinner.
      if (
        res.status === 429 &&
        /RESOURCE_EXHAUSTED|quota|rate limit|reached.*limit/i.test(errText)
      ) {
        return fail("rate_limited", QUOTA_EXHAUSTED_MESSAGE, "gemini");
      }
      if (attempt < MAX_RETRIES) {
        await backoff(res, attempt);
        continue;
      }
      if (res.status === 429) {
        return fail("rate_limited", QUOTA_EXHAUSTED_MESSAGE, "gemini");
      }
      return fail(
        "server",
        `The AI service returned an error (Gemini HTTP ${res.status}).`,
        "gemini",
      );
    }

    // Invalid / expired key → fall through to the next key/provider.
    if (res.status === 401 || res.status === 403) {
      return fail(
        "not_configured",
        "The Gemini API key was rejected. Check GEMINI_API_KEY in the Keys tab.",
        "gemini",
      );
    }

    // Missing / unsupported model → fall through.
    if (res.status === 404) {
      return fail(
        "model_error",
        `The Gemini model "${model}" was not found (HTTP 404). Set GEMINI_MODEL in the Keys tab to a model your key can use.`,
        "gemini",
      );
    }

    if (res.status === 400) {
      const errText = await res.text().catch(() => "");
      if (/not found|not supported|does not exist|not available|invalid model/i.test(errText)) {
        return fail(
          "model_error",
          `The Gemini model "${model}" was rejected (HTTP 400): ${errText.slice(0, 200)}`,
          "gemini",
        );
      }
      return fail(
        "server",
        `The AI service rejected the request (Gemini HTTP 400): ${errText.slice(0, 200)}`,
        "gemini",
      );
    }

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return fail(
        "server",
        `The AI service returned an error (Gemini HTTP ${res.status})${
          errText ? `: ${errText.slice(0, 200)}` : ""
        }.`,
        "gemini",
      );
    }

    // Success — extract every part (thinking models emit a thought part first).
    const parsed = (await res.json()) as {
      candidates?: Array<{
        content?: { parts?: Array<{ text?: string }> };
        finishReason?: string;
      }>;
      promptFeedback?: { blockReason?: string };
    };
    const candidates = parsed.candidates;
    if (!Array.isArray(candidates) || candidates.length === 0) {
      const blockReason = parsed.promptFeedback?.blockReason;
      return fail(
        "server",
        `The AI returned no content${
          blockReason ? ` (blocked by safety filter: ${blockReason})` : ""
        }.`,
        "gemini",
      );
    }
    const parts = candidates[0]?.content?.parts ?? [];
    const text = parts
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .join("\n");
    if (!text.trim()) {
      return fail("server", "The AI returned an empty response.", "gemini");
    }
    return {
      ok: true,
      text,
      provider: "gemini",
      model,
      finishReason: candidates[0]?.finishReason ?? undefined,
    };
  }
  return fail("server", "The AI service is temporarily unavailable (Gemini).", "gemini");
}

/** Tries every configured Gemini key, rotating on key-specific failures only. */
async function attemptGemini(
  req: AiProviderRequest,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const keys = collectKeys("GEMINI_API_KEY");
  if (keys.length === 0) {
    return fail(
      "not_configured",
      "Gemini is not configured. Add GEMINI_API_KEY (or GEMINI_API_KEY_1..N) in the Keys tab.",
      "gemini",
    );
  }

  let lastFailure: AiProviderFailure | null = null;
  for (const key of keys) {
    const result = await attemptGeminiKey(req, key);
    if (result.ok) return result;
    lastFailure = result;
    console.warn(
      `[ai-provider] Gemini key attempt failed (code=${result.code}, remainingKeys=${keys.length - (keys.indexOf(key) + 1)})` +
        (isKeyRotatable(result.code) ? " — trying next Gemini key." : " — moving to next provider."),
    );
    if (!isKeyRotatable(result.code)) return result;
  }
  return (
    lastFailure ??
    fail("server", "The AI service is temporarily unavailable (Gemini).", "gemini")
  );
}

/* ------------------------------------------------------------------ */
/* OpenRouter (TERTIARY)                                               */
/* ------------------------------------------------------------------ */

/** Tries one OpenRouter model; returns the failure so callers can retry others. */
async function attemptOpenRouterModel(
  req: AiProviderRequest,
  key: string,
  model: string,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const baseBody: Record<string, unknown> = {
    model,
    messages: [
      { role: "system", content: req.systemPrompt },
      { role: "user", content: req.userMessage },
    ],
    temperature: req.temperature ?? 0.3,
    max_tokens: req.maxOutputTokens ?? 2048,
  };
  const withJson = req.jsonMode ? { ...baseBody, response_format: { type: "json_object" } } : baseBody;

  // Some models reject response_format — retry the same request without it.
  const attempts: Array<Record<string, unknown>> = req.jsonMode
    ? [withJson, baseBody]
    : [baseBody];

  for (const attemptBody of attempts) {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
      let res: Response;
      try {
        res = await fetch(OPENROUTER_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${key}`,
            "HTTP-Referer": "https://arogyaos.app",
            "X-Title": "ArogyaOS",
          },
          body: JSON.stringify(attemptBody),
          signal: AbortSignal.timeout(req.timeoutMs ?? DEFAULT_TIMEOUT_MS),
        });
      } catch (err) {
        if (isTimeout(err)) {
          return fail(
            "timeout",
            "The AI took too long to respond (OpenRouter timed out).",
            "openrouter",
          );
        }
        return fail(
          "network",
          "Could not reach the AI service (OpenRouter network error).",
          "openrouter",
        );
      }

      if (res.status === 429) {
        if (attempt < MAX_RETRIES) {
          await backoff(res, attempt);
          continue;
        }
        return fail(
          "rate_limited",
          `OpenRouter is rate-limited for "${model}" right now. Trying the next model/key automatically.`,
          "openrouter",
        );
      }
      if (res.status === 500 || res.status === 502 || res.status === 503 || res.status === 504) {
        if (attempt < MAX_RETRIES) {
          await backoff(res, attempt);
          continue;
        }
        return fail(
          "server",
          `The AI service returned an error (OpenRouter HTTP ${res.status}).`,
          "openrouter",
        );
      }

      if (res.status === 401 || res.status === 403) {
        return fail(
          "not_configured",
          "The OpenRouter API key was rejected. Check OPENROUTER_API_KEY in the Keys tab.",
          "openrouter",
        );
      }
      // No credits (402) — a *free* model may still work, so let the caller
      // try the next model in the chain before giving up.
      if (res.status === 402) {
        return fail("no_credits", NO_CREDITS_MESSAGE, "openrouter");
      }
      if (res.status === 404) {
        return fail(
          "model_error",
          `The OpenRouter model "${model}" was not found (HTTP 404).`,
          "openrouter",
        );
      }
      if (res.status === 400) {
        const errText = await res.text().catch(() => "");
        // The model may not support JSON mode — drop response_format and retry.
        if (/response_format|json.?object/i.test(errText) && attemptBody !== baseBody) {
          break;
        }
        if (/model|not found|not supported|does not exist/i.test(errText)) {
          return fail(
            "model_error",
            `The OpenRouter model "${model}" was rejected (HTTP 400): ${errText.slice(0, 200)}`,
            "openrouter",
          );
        }
        return fail(
          "server",
          `OpenRouter rejected the request (HTTP 400): ${errText.slice(0, 200)}`,
          "openrouter",
        );
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        return fail(
          "server",
          `OpenRouter returned an error (HTTP ${res.status})${
            errText ? `: ${errText.slice(0, 200)}` : ""
          }.`,
          "openrouter",
        );
      }

      const parsed = (await res.json()) as {
        choices?: Array<{
          message?: { content?: string | Array<{ text?: string }> };
          finish_reason?: string;
        }>;
      };
      const content = parsed.choices?.[0]?.message?.content;
      let text = "";
      if (typeof content === "string") text = content;
      else if (Array.isArray(content)) {
        text = content
          .map((part) => (typeof part?.text === "string" ? part.text : ""))
          .join("");
      }
      if (!text.trim()) {
        return fail("server", "OpenRouter returned an empty response.", "openrouter");
      }
      return {
        ok: true,
        text,
        provider: "openrouter",
        model,
        finishReason: parsed.choices?.[0]?.finish_reason ?? undefined,
      };
    }
  }
  return fail("server", "The AI service is temporarily unavailable (OpenRouter).", "openrouter");
}

/**
 * Tries the configured OpenRouter model first, then a chain of free models for
 * ONE key, returning the first success. Keeps AI working even when the account
 * has no credits (paid models return 402) or a model is retired/renamed.
 */
async function attemptOpenRouterModels(
  req: AiProviderRequest,
  key: string,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const configured = process.env.OPENROUTER_MODEL || DEFAULT_OPENROUTER_MODEL;
  const models = [configured, ...OPENROUTER_FALLBACK_MODELS].filter(
    (model, index, all) => all.indexOf(model) === index,
  );

  let lastFailure: AiProviderFailure | null = null;
  for (const model of models) {
    const result = await attemptOpenRouterModel(req, key, model);
    if (result.ok) return result;
    lastFailure = result;
    console.warn(
      `[ai-provider] OpenRouter model "${model}" failed (code=${result.code}) — trying next model.`,
    );
    // Fatal errors won't be fixed by another model — stop the chain.
    if (result.code === "not_configured" || result.code === "network" || result.code === "timeout") {
      return result;
    }
  }
  return (
    lastFailure ??
    fail("server", "The AI service is temporarily unavailable (OpenRouter).", "openrouter")
  );
}

/** Tries every configured OpenRouter key, rotating on key-specific failures. */
async function attemptOpenRouter(
  req: AiProviderRequest,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const keys = collectKeys("OPENROUTER_API_KEY");
  if (keys.length === 0) {
    return fail(
      "not_configured",
      "OpenRouter is not configured. Add OPENROUTER_API_KEY (or OPENROUTER_API_KEY_1..N) in the Keys tab.",
      "openrouter",
    );
  }

  let lastFailure: AiProviderFailure | null = null;
  for (const key of keys) {
    const result = await attemptOpenRouterModels(req, key);
    if (result.ok) return result;
    lastFailure = result;
    console.warn(
      `[ai-provider] OpenRouter key attempt failed (code=${result.code}, remainingKeys=${keys.length - (keys.indexOf(key) + 1)})` +
        (isKeyRotatable(result.code) ? " — trying next OpenRouter key." : " — moving to next provider."),
    );
    if (!isKeyRotatable(result.code)) return result;
  }
  return (
    lastFailure ??
    fail("server", "The AI service is temporarily unavailable (OpenRouter).", "openrouter")
  );
}

/* ------------------------------------------------------------------ */
/* Platform AI gateway (VLY — final fallback)                          */
/* ------------------------------------------------------------------ */

/**
 * Freebuff's built-in AI gateway (auto-billed, no external account needed).
 * Activated when VLY_INTEGRATION_KEY is set in the Convex environment. The
 * gateway speaks the OpenAI-compatible API, so chat messages map 1:1.
 */
async function attemptVly(
  req: AiProviderRequest,
  key: string,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const { createVlyIntegrations } = await import("@vly-ai/integrations");
  const model = process.env.VLY_MODEL || DEFAULT_VLY_MODEL;
  const vly = createVlyIntegrations({ deploymentToken: key, debug: false });

  try {
    const result = await Promise.race([
      vly.ai.completion({
        model,
        messages: [
          { role: "system", content: req.systemPrompt },
          { role: "user", content: req.userMessage },
        ],
        temperature: req.temperature ?? 0.3,
        maxTokens: req.maxOutputTokens ?? 2048,
      }),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error("VLY_TIMEOUT")),
          req.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        ),
      ),
    ]);

    if (!result.success) {
      return fail(
        "server",
        `The AI gateway returned an error: ${result.error ?? "unknown"}`,
        "vly",
      );
    }
    const text = result.data?.choices?.[0]?.message?.content ?? "";
    if (!text.trim()) {
      return fail("server", "The AI gateway returned an empty response.", "vly");
    }
    return {
      ok: true,
      text,
      provider: "vly",
      model,
      finishReason: result.data?.choices?.[0]?.finishReason ?? "stop",
    };
  } catch (err) {
    if (isTimeout(err) || /VLY_TIMEOUT/i.test(String(err))) {
      return fail(
        "timeout",
        "The AI took too long to respond (gateway timed out).",
        "vly",
      );
    }
    return fail(
      "server",
      `The AI gateway could not be reached: ${err instanceof Error ? err.message : String(err)}`,
      "vly",
    );
  }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                  */
/* ------------------------------------------------------------------ */

/**
 * Runs the request against Groq first, then Gemini, then OpenRouter, then the
 * platform gateway — moving to the next provider on retryable/key-specific
 * failures. Returns the normalized result — the caller cannot tell which
 * provider produced it from the schema alone.
 */
export async function generateWithProviderFallback(
  req: AiProviderRequest,
): Promise<AiProviderSuccess | AiProviderFailure> {
  const startedAt = Date.now();
  let lastFailure: AiProviderFailure | null = null;

  // 1. Groq — primary provider.
  const groqKeys = collectKeys("GROQ_API_KEY");
  if (groqKeys.length > 0) {
    const result = await attemptGroq(req);
    if (result.ok) {
      console.log(
        `[ai-provider] Success: groq (model=${result.model}, keys=${groqKeys.length}, textBytes=${Buffer.byteLength(
          result.text,
          "utf8",
        )}, elapsedMs=${Date.now() - startedAt})`,
      );
      return result;
    }
    lastFailure = result;
    console.warn(`[ai-provider] Groq failed (code=${result.code}) — trying next provider.`);
  } else {
    console.log("[ai-provider] GROQ_API_KEY not set — skipping Groq.");
  }

  // 2. Gemini — secondary provider.
  const geminiKeys = collectKeys("GEMINI_API_KEY");
  if (geminiKeys.length > 0) {
    const result = await attemptGemini(req);
    if (result.ok) {
      console.log(
        `[ai-provider] Success: gemini (model=${result.model}, keys=${geminiKeys.length}, textBytes=${Buffer.byteLength(
          result.text,
          "utf8",
        )}, elapsedMs=${Date.now() - startedAt})`,
      );
      return result;
    }
    lastFailure = result;
    console.warn(`[ai-provider] Gemini failed (code=${result.code}) — trying next provider.`);
  } else {
    console.log("[ai-provider] GEMINI_API_KEY not set — skipping Gemini.");
  }

  // 3. OpenRouter — tertiary provider.
  const openRouterKeys = collectKeys("OPENROUTER_API_KEY");
  if (openRouterKeys.length > 0) {
    const result = await attemptOpenRouter(req);
    if (result.ok) {
      console.log(
        `[ai-provider] Success: openrouter (model=${result.model}, keys=${openRouterKeys.length}, textBytes=${Buffer.byteLength(
          result.text,
          "utf8",
        )}, elapsedMs=${Date.now() - startedAt})`,
      );
      return result;
    }
    lastFailure = result;
    console.warn(`[ai-provider] OpenRouter failed (code=${result.code}) — trying next provider.`);
  } else {
    console.log("[ai-provider] OPENROUTER_API_KEY not set — skipping OpenRouter.");
  }

  // 4. Platform AI gateway — final fallback (no external account needed).
  const vlyKey = process.env.VLY_INTEGRATION_KEY;
  if (vlyKey) {
    const result = await attemptVly(req, vlyKey);
    if (result.ok) {
      console.log(
        `[ai-provider] Success: vly (model=${result.model}, textBytes=${Buffer.byteLength(
          result.text,
          "utf8",
        )}, elapsedMs=${Date.now() - startedAt})`,
      );
      return result;
    }
    lastFailure = result;
    console.warn(
      `[ai-provider] Platform gateway failed (code=${result.code}) — all providers exhausted.`,
    );
  } else {
    console.log("[ai-provider] VLY_INTEGRATION_KEY not set — skipping platform gateway.");
  }

  if (lastFailure) return lastFailure;

  return fail(
    "not_configured",
    "No AI provider is configured. Add GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or VLY_INTEGRATION_KEY in the Keys tab (Convex env).",
  );
}

/**
 * Maps a provider failure onto the action-level error codes the frontend
 * already understands, so all actions handle failures identically.
 */
export function mapProviderFailure(
  failure: AiProviderFailure,
): { code: AiProviderErrorCode; message: string } {
  switch (failure.code) {
    case "rate_limited":
      return { code: "rate_limited", message: failure.message };
    case "no_credits":
      return { code: "no_credits", message: failure.message };
    case "timeout":
      return { code: "timeout", message: "The AI took too long to respond. Please try again." };
    case "network":
      return { code: "network", message: "Could not reach the AI service. Check your connection." };
    case "model_error":
      return { code: "model_error", message: failure.message };
    case "server":
      return { code: "server", message: failure.message };
    case "not_configured":
    default:
      return { code: "not_configured", message: failure.message };
  }
}
