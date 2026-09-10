"use node";

/**
 * APBE AI Enhancement (Feature: Adaptive Personal Baseline Engine) — secure
 * AI action.
 *
 * This action is separate from `insights:generate` (the report analysis) and
 * does NOT affect it. It takes the deterministic APBE statistics computed by
 * the client engine (historical values, personal baseline, current values,
 * trend statistics) and asks the AI for a plain-language briefing:
 * overall trend, improving/declining metrics, important changes, recommended
 * actions, monitoring advice, and a confidence score.
 *
 * The LLM call goes through the shared `aiProvider` module
 * (src/convex/aiProvider.ts), same as every other AI feature: Groq primary,
 * Gemini automatic fallback on quota (429 / RESOURCE_EXHAUSTED), 5xx,
 * timeout, network, or invalid key/model, OpenRouter tertiary, and the
 * platform AI gateway (VLY) as the final fallback. Only the
 * already-computed numeric statistics are sent — never the raw report files
 * or OCR text, and never the full raw chronological value arrays. If EVERY
 * provider fails, the user's most recent successful briefing is re-served
 * from the aiCache table (Convex) so the UI still shows a result instead of
 * an error.
 *
 * Token budget: at most 30 metrics are sent, each with only the compact
 * statistics (rolling mean, std dev, z-score, latest value, direction) plus
 * a bounded recent-value excerpt; the model is limited to 700 output tokens
 * (the briefing schema needs ~400–700).
 *
 * Security model (mirrors insights.ts):
 *  - The caller's Supabase access token is verified server-side against the
 *    Supabase Auth endpoint.
 *  - The AI keys live only in process.env on the Convex server.
 *  - The aiCache fallback is keyed by the verified Supabase user id, so a
 *    user can only read/write their own cache entry.
 *
 * Environment variables (Keys tab / Convex env):
 *   GROQ_API_KEY         — Groq API key (primary provider), may use _1..N
 *   GROQ_MODEL           — optional, default "llama-3.3-70b-versatile"
 *   GEMINI_API_KEY       — Google AI Studio API key (secondary provider)
 *   GEMINI_MODEL         — optional, default "gemini-3.6-flash"
 *   OPENROUTER_API_KEY   — OpenRouter API key (tertiary fallback provider)
 *   OPENROUTER_MODEL     — optional, default "openai/gpt-4o-mini"
 *   VLY_INTEGRATION_KEY  — platform AI gateway key (final fallback, `sk_*`)
 *   VLY_MODEL            — optional, default "gpt-4o-mini"
 *   SUPABASE_URL         — e.g. https://<project>.supabase.co (required)
 *   SUPABASE_ANON_KEY    — publishable anon key (required, used with the user JWT)
 */

import { action } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import {
  generateWithProviderFallback,
  hasAiProvidersConfigured,
  mapProviderFailure,
  type AiProvider,
} from "./aiProvider";

/** Token budget: keep the baseline briefing prompt compact (400–700 out). */
const MAX_METRICS_SENT = 30;
const MAX_VALUES_PER_METRIC = 20;
const MAX_OUTPUT_TOKENS = 700;

const SYSTEM_PROMPT = `You are a preventive-health analyst helping a patient understand how their lab values compare to their OWN historical baseline (not population ranges).

The user provides, per metric: the chronological values, the personal rolling average, standard deviation, latest z-score, latest value, percentage difference from personal average, and the recent trend direction.

Rules:
- Write plainly and reassuringly; never diagnose disease.
- Always frame findings as observations and suggest confirming with a doctor when anything deviates more than 1 standard deviation from the personal baseline.
- Respond with ONLY a single JSON object. No markdown, no code fences, no commentary.

The JSON must match EXACTLY this schema:
{
  "overallTrend": "string — 1 to 3 sentence summary of the overall picture across all metrics",
  "improvingMetrics": ["string — metric name that is improving"],
  "decliningMetrics": ["string — metric name that is worsening"],
  "importantChanges": ["string — notable changes worth watching"],
  "recommendedActions": ["string — practical actions, at most 5"],
  "monitoringAdvice": ["string — how often/what to monitor, at most 3"],
  "confidence": "integer 0-100"
}`;

export type BaselineAiErrorCode =
  | "not_configured"
  | "model_error"
  | "unauthorized"
  | "empty_input"
  | "rate_limited"
  | "no_credits"
  | "timeout"
  | "network"
  | "invalid_json"
  | "server";

export interface BaselineAiBrief {
  overallTrend: string;
  improvingMetrics: string[];
  decliningMetrics: string[];
  importantChanges: string[];
  recommendedActions: string[];
  monitoringAdvice: string[];
  confidence: number;
}

export type BaselineAiOutcome =
  | {
      ok: true;
      brief: BaselineAiBrief;
      raw: string;
      model: string;
      provider: AiProvider;
      processingTimeMs: number;
      /** True when re-served from the user's last successful briefing. */
      cached?: boolean;
    }
  | { ok: false; code: BaselineAiErrorCode; message: string };

export const enhance = action({
  args: {
    accessToken: v.string(),
    metrics: v.array(
      v.object({
        metricName: v.string(),
        label: v.string(),
        unit: v.optional(v.union(v.string(), v.null())),
        values: v.array(v.number()),
        rollingMean: v.optional(v.union(v.number(), v.null())),
        stdDev: v.optional(v.union(v.number(), v.null())),
        latestZScore: v.optional(v.union(v.number(), v.null())),
        latestValue: v.optional(v.union(v.number(), v.null())),
        percentageDifference: v.optional(v.union(v.number(), v.null())),
        direction: v.string(),
        personalStatus: v.string(),
      }),
    ),
  },
  handler: async (ctx, args): Promise<BaselineAiOutcome> => {
    const supabaseUrl = process.env.SUPABASE_URL?.replace(/\/+$/, "");
    const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

    const fail = (
      code: BaselineAiErrorCode,
      message: string,
    ): BaselineAiOutcome => ({ ok: false, code, message });

    if (!hasAiProvidersConfigured()) {
      return fail(
        "not_configured",
        "No AI provider is configured. Add GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or VLY_INTEGRATION_KEY in the Keys tab (Convex env).",
      );
    }
    if (!supabaseUrl || !supabaseAnonKey) {
      return fail(
        "not_configured",
        "SUPABASE_URL / SUPABASE_ANON_KEY are not set in the Convex environment.",
      );
    }

    // 1. Verify the caller's Supabase session server-side and capture the user
    //    id — used to scope the aiCache fallback to this user only.
    let userId: string | undefined;
    try {
      const userRes = await fetch(`${supabaseUrl}/auth/v1/user`, {
        headers: {
          apikey: supabaseAnonKey,
          Authorization: `Bearer ${args.accessToken}`,
        },
      });
      if (!userRes.ok) {
        return fail("unauthorized", "Could not verify your session. Please sign in again.");
      }
      const userData = (await userRes.json()) as { id?: string };
      userId = userData.id;
    } catch {
      return fail("network", "Could not reach the authentication service.");
    }
    if (!userId) {
      return fail("unauthorized", "Could not verify your session. Please sign in again.");
    }

    // 2. Empty input guard.
    if (args.metrics.length === 0) {
      return fail(
        "empty_input",
        "No metric data was provided. Upload at least three reports to build your baseline first.",
      );
    }

    // 3. Call the AI provider layer (Groq → Gemini → OpenRouter → gateway)
    //    with the deterministic APBE statistics. Only the compact statistics
    //    are sent — the full raw chronological `values` arrays are replaced
    //    with a bounded recent excerpt (the rolling mean / std dev / z-score /
    //    latest value already encode the trend) to keep the prompt small and
    //    within provider input limits.
    const metricsForAi = args.metrics.slice(0, MAX_METRICS_SENT).map((m) => ({
      metricName: m.metricName,
      label: m.label,
      unit: m.unit ?? null,
      readingCount: (m.values ?? []).length,
      recentValues: (m.values ?? []).slice(-MAX_VALUES_PER_METRIC),
      rollingMean: m.rollingMean ?? null,
      stdDev: m.stdDev ?? null,
      latestZScore: m.latestZScore ?? null,
      latestValue: m.latestValue ?? null,
      percentageDifference: m.percentageDifference ?? null,
      direction: m.direction,
      personalStatus: m.personalStatus,
    }));

    const userMessage = [
      "Here is the user's personal baseline analysis. Please provide the plain-language briefing.",
      "",
      "=== METRIC STATISTICS ===",
      JSON.stringify(metricsForAi, null, 1),
      "",
      "Return the JSON briefing exactly as instructed.",
    ].join("\n");

    const startedAt = Date.now();
    const result = await generateWithProviderFallback({
      systemPrompt: SYSTEM_PROMPT,
      userMessage,
      temperature: 0.4,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      jsonMode: true,
    });

    if (!result.ok) {
      // Cached fallback: re-serve the user's most recent successful briefing.
      try {
        const cached = await ctx.runQuery(internal.aiCache.get, {
          userId,
          feature: "baseline",
        });
        if (cached?.payload) {
          const brief = normalizeBrief(JSON.parse(cached.payload) as Record<string, unknown>);
          if (brief.overallTrend) {
            console.warn(
              `[baselines] All AI providers failed (code=${result.code}) — serving cached baseline briefing.`,
            );
            return {
              ok: true,
              brief,
              raw: cached.payload,
              model: cached.model ?? "cached",
              provider: (cached.provider ?? "cache") as AiProvider,
              processingTimeMs: Date.now() - startedAt,
              cached: true,
            };
          }
        }
      } catch (err) {
        console.warn(
          `[baselines] aiCache read failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
      const mapped = mapProviderFailure(result);
      return fail(mapped.code, mapped.message);
    }

    const rawContent = result.text;
    const processingTimeMs = Date.now() - startedAt;

    // 4. Parse + validate the JSON briefing.
    const parsed = extractJsonFromParts(rawContent.split("\n"));
    if (!parsed) {
      const snippet = rawContent.trim().slice(0, 300);
      return fail(
        "invalid_json",
        `The AI returned an unreadable response${
          snippet ? ` — body: ${snippet}` : ""
        }. Please try again.`,
      );
    }
    const brief = normalizeBrief(parsed);

    // 5. Persist the successful briefing as the user's cache for fallback.
    try {
      await ctx.runMutation(internal.aiCache.put, {
        userId,
        feature: "baseline",
        payload: JSON.stringify(brief),
        provider: result.provider,
        model: result.model,
      });
    } catch (err) {
      console.warn(
        `[baselines] aiCache write failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    return {
      ok: true,
      brief,
      raw: rawContent,
      model: result.model,
      provider: result.provider,
      processingTimeMs,
    };
  },
});

/** Pulls the first balanced JSON object out of a possibly-noisy string. */
function extractJson(content: string): Record<string, unknown> | null {
  if (!content) return null;
  const cleaned = content.replace(/```(?:json)?/gi, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const parsed: unknown = JSON.parse(cleaned.slice(start, end + 1));
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** Tries each response part (the JSON part is usually last), then the join. */
function extractJsonFromParts(parts: string[]): Record<string, unknown> | null {
  for (let i = parts.length - 1; i >= 0; i--) {
    const parsed = extractJson(parts[i]);
    if (parsed) return parsed;
  }
  return extractJson(parts.join("\n"));
}

function stringArray(value: unknown, max = 10): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .slice(0, max)
    .map((item) => item.trim());
}

function normalizeBrief(raw: Record<string, unknown>): BaselineAiBrief {
  const confidenceRaw = Number(raw.confidence);
  const confidence = Number.isFinite(confidenceRaw)
    ? Math.max(0, Math.min(100, Math.round(confidenceRaw)))
    : 0;

  return {
    overallTrend:
      typeof raw.overallTrend === "string" && raw.overallTrend.trim()
        ? raw.overallTrend.trim()
        : "",
    improvingMetrics: stringArray(raw.improvingMetrics),
    decliningMetrics: stringArray(raw.decliningMetrics),
    importantChanges: stringArray(raw.importantChanges),
    recommendedActions: stringArray(raw.recommendedActions, 5),
    monitoringAdvice: stringArray(raw.monitoringAdvice, 3),
    confidence,
  };
}
