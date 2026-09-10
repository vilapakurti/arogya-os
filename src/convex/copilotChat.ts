"use node";

/**
 * Doctor Copilot — "Ask Doctor Copilot" (conversational AI assistant).
 *
 * Secure AI action for the interactive chat panel on the Doctor Copilot page.
 * It answers a patient's questions using their OWN health data: uploaded
 * reports, extracted OCR text, parsed health metrics, previous AI analyses,
 * personal baselines, the latest-report comparison, and a Clinical Decision
 * Support (CDSS) highlights block computed by src/lib/clinical.
 *
 * The LLM call goes through the shared `aiProvider` module
 * (src/convex/aiProvider.ts): Groq primary, Gemini automatic fallback on
 * quota (429 / RESOURCE_EXHAUSTED), 5xx, timeout, network, or invalid
 * key/model, OpenRouter tertiary, and the platform AI gateway (VLY) as the
 * final fallback. Chat replies are plain text, so the response schema is the
 * same regardless of provider. (Chat is conversational — no cached-result
 * fallback; each reply is a fresh question.)
 *
 * Token budget: history is capped at 10 turns, the OCR excerpt at 3,000
 * chars, previous AI summaries at 3, and the reply at 1024 output tokens.
 *
 * Security model (mirrors insights.ts / baselines.ts / copilot.ts):
 *  - The caller's Supabase access token is verified server-side against the
 *    Supabase Auth endpoint — the action never trusts the client identity.
 *  - The health snapshot is assembled client-side exclusively from existing
 *    RLS-scoped queries (the signed-in user's own reports/metrics/insights/
 *    baselines), so it can only ever contain that user's data.
 *  - Conversation history is session-scoped and never persisted anywhere.
 *  - The AI keys live only in process.env on the Convex server.
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
import { v } from "convex/values";
import {
  generateWithProviderFallback,
  hasAiProvidersConfigured,
  mapProviderFailure,
  type AiProvider,
} from "./aiProvider";

const MAX_HISTORY_TURNS = 10;
const MAX_OCR_CHARS = 3_000;
const MAX_SUMMARIES_SENT = 3;

const CHAT_SYSTEM_PROMPT = `You are "Arogya Copilot", a supportive preventive-health assistant inside ArogyaOS. You help a patient understand their own medical history and prepare for a doctor visit.

The patient provides a snapshot of their OWN health data: number of reports, latest report details, tracked metrics, improving/worsening metrics, an overall risk level (already computed deterministically), abnormal findings, milestones, per-metric comparisons (latest vs previous vs personal baseline vs population reference range), an excerpt of the latest report's extracted OCR text, summaries of previous AI analyses, and a Clinical Decision Support (CDSS) block with per-metric clinical meanings, combined findings, risk levels, and any emergency flags.

Rules you must follow:
- Answer ONLY from the provided health snapshot. Never invent values, tests, medications, or findings that are not in the snapshot.
- If the snapshot lacks what they ask about, say so plainly, e.g. "I don't have data on that in your uploaded reports."
- Use the CDSS clinical meanings and risk levels to frame answers, but never diagnose disease or prescribe treatment. Frame everything as observations and always encourage confirming with a doctor.
- Be concise and reassuring: 2-6 short paragraphs or a few bullets.
- If the question is unrelated to their health data, gently bring the conversation back to their reports.`;

export type CopilotChatErrorCode =
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

export type CopilotChatOutcome =
  | {
      ok: true;
      reply: string;
      model: string;
      provider: AiProvider;
      processingTimeMs: number;
    }
  | { ok: false; code: CopilotChatErrorCode; message: string };

export const chat = action({
  args: {
    accessToken: v.string(),
    question: v.string(),
    history: v.array(
      v.object({
        role: v.string(),
        content: v.string(),
      }),
    ),
    context: v.object({
      reportCount: v.number(),
      latestReportTitle: v.optional(v.union(v.string(), v.null())),
      latestReportDate: v.optional(v.union(v.string(), v.null())),
      metricsTracked: v.array(v.string()),
      improvingMetrics: v.array(v.string()),
      worseningMetrics: v.array(v.string()),
      overallRiskLevel: v.string(),
      abnormalFindings: v.array(v.string()),
      milestones: v.array(v.string()),
      metricComparisons: v.array(
        v.object({
          label: v.string(),
          unit: v.optional(v.union(v.string(), v.null())),
          latestValue: v.optional(v.union(v.number(), v.null())),
          previousValue: v.optional(v.union(v.number(), v.null())),
          personalBaseline: v.optional(v.union(v.number(), v.null())),
          populationMin: v.optional(v.union(v.number(), v.null())),
          populationMax: v.optional(v.union(v.number(), v.null())),
        }),
      ),
      latestOcrExcerpt: v.optional(v.union(v.string(), v.null())),
      previousSummaries: v.array(v.string()),
      /* ---- CDSS highlights (optional — older clients keep working) ---- */
      clinicalHighlights: v.optional(v.string()),
    }),
  },
  handler: async (_ctx, args): Promise<CopilotChatOutcome> => {
    const supabaseUrl = process.env.SUPABASE_URL?.replace(/\/+$/, "");
    const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

    const fail = (code: CopilotChatErrorCode, message: string): CopilotChatOutcome => ({
      ok: false,
      code,
      message,
    });

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

    // 1. Verify the caller's Supabase session server-side.
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
    } catch {
      return fail("network", "Could not reach the authentication service.");
    }

    // 2. Empty question guard.
    const question = args.question.trim();
    if (!question) {
      return fail("empty_input", "Ask a question about your health data first.");
    }

    // 3. Call the AI provider layer (Groq → Gemini → OpenRouter → gateway)
    //    with the health snapshot + conversation history, trimmed server-side.
    const historyBlock = args.history
      .slice(-MAX_HISTORY_TURNS)
      .map((m) => `${m.role === "user" ? "Patient" : "Arogya Copilot"}: ${m.content}`)
      .join("\n\n");

    const userMessage = [
      "The patient asks a question about their health. Answer using ONLY the snapshot below and the conversation so far.",
      "",
      "=== PATIENT'S HEALTH SNAPSHOT (from their own ArogyaOS records) ===",
      JSON.stringify(
        {
          ...args.context,
          latestOcrExcerpt: (args.context.latestOcrExcerpt ?? "").slice(0, MAX_OCR_CHARS),
          previousSummaries: args.context.previousSummaries.slice(0, MAX_SUMMARIES_SENT),
        },
        null,
        1,
      ),
      "",
      "=== CONVERSATION SO FAR ===",
      historyBlock || "(no previous messages)",
      "",
      "=== PATIENT'S QUESTION ===",
      question,
    ].join("\n");

    const startedAt = Date.now();
    const result = await generateWithProviderFallback({
      systemPrompt: CHAT_SYSTEM_PROMPT,
      userMessage,
      temperature: 0.5,
      maxOutputTokens: 1024,
      jsonMode: false,
    });

    if (!result.ok) {
      const mapped = mapProviderFailure(result);
      return fail(mapped.code, mapped.message);
    }

    const processingTimeMs = Date.now() - startedAt;
    const reply = result.text.trim();
    if (!reply) {
      return fail("invalid_json", "The AI returned an empty response. Please try again.");
    }

    return {
      ok: true,
      reply,
      model: result.model,
      provider: result.provider,
      processingTimeMs,
    };
  },
});
