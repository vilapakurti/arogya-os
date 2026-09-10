/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as aiCache from "../aiCache.js";
import type * as aiDiag from "../aiDiag.js";
import type * as aiProvider from "../aiProvider.js";
import type * as auth from "../auth.js";
import type * as auth_emailOtp from "../auth/emailOtp.js";
import type * as baselines from "../baselines.js";
import type * as copilot from "../copilot.js";
import type * as copilotChat from "../copilotChat.js";
import type * as envCheck from "../envCheck.js";
import type * as http from "../http.js";
import type * as insights from "../insights.js";
import type * as users from "../users.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  aiCache: typeof aiCache;
  aiDiag: typeof aiDiag;
  aiProvider: typeof aiProvider;
  auth: typeof auth;
  "auth/emailOtp": typeof auth_emailOtp;
  baselines: typeof baselines;
  copilot: typeof copilot;
  copilotChat: typeof copilotChat;
  envCheck: typeof envCheck;
  http: typeof http;
  insights: typeof insights;
  users: typeof users;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
