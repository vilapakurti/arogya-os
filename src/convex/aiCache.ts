/**
 * aiCache — last successful AI result per user + feature.
 *
 * Used by the Health Baseline and Doctor Copilot actions as a graceful
 * fallback: when every AI provider fails (quota, rate limit, outage), the
 * action re-serves the user's most recent successful brief from this table
 * instead of returning a bare error. Keyed by (userId, feature), one row per
 * user/feature, upserted on every successful AI generation.
 *
 * Security: rows are scoped by `userId` (the Supabase user id verified
 * server-side in the calling action) — a user can only read/write their own
 * cache entry.
 */

import { internalMutation, internalQuery } from "./_generated/server";
import { v } from "convex/values";

export const get = internalQuery({
  args: {
    userId: v.string(),
    feature: v.string(),
  },
  handler: async (ctx, args) => {
    const doc = await ctx.db
      .query("aiCache")
      .withIndex("by_user_feature", (q) =>
        q.eq("userId", args.userId).eq("feature", args.feature),
      )
      .first();
    return doc ?? null;
  },
});

export const put = internalMutation({
  args: {
    userId: v.string(),
    feature: v.string(),
    payload: v.string(),
    provider: v.optional(v.string()),
    model: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("aiCache")
      .withIndex("by_user_feature", (q) =>
        q.eq("userId", args.userId).eq("feature", args.feature),
      )
      .first();
    const fields = {
      payload: args.payload,
      provider: args.provider,
      model: args.model,
      createdAt: Date.now(),
    };
    if (existing) {
      await ctx.db.patch(existing._id, fields);
    } else {
      await ctx.db.insert("aiCache", {
        userId: args.userId,
        feature: args.feature,
        ...fields,
      });
    }
  },
});
