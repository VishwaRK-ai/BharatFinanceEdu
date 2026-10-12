import type { Locale } from "@/i18n/routing";
import { getServerEnv } from "@/lib/env";
import { BackendError } from "./errors";
import {
  mockAssistant,
  mockQuizStart,
  mockQuizSubmit,
} from "./mock";
import {
  assistantResponseSchema,
  normalizeQuizStart,
  normalizeQuizSubmit,
  type AssistantResponse,
  type ChatTurn,
  type QuizStartResponse,
  type QuizSubmitResult,
  type ScanSummaryPayload,
} from "./schemas";

/**
 * Server-only client for the FastAPI LLM backend.
 *
 * The `server-only` marker package is intentionally not installed — this
 * window guard provides the same protection without a new dependency:
 * importing this module in client code throws immediately.
 */
if (typeof window !== "undefined") {
  throw new Error("src/lib/backend/client.ts is server-only — never import it from client code.");
}

const ASSISTANT_TIMEOUT_MS = 90_000;

/**
 * The backend has no `language` field — the model is multilingual and follows
 * the prompt language. This one-line instruction (in the target language) is
 * prepended to the message we forward, so replies come back in the UI locale
 * even when the user typed in English. Only the user's original text is ever
 * stored or displayed.
 */
const REPLY_INSTRUCTION: Record<Locale, string> = {
  en: "(Reply in English.)",
  hi: "(उत्तर हिंदी में दें।)",
  mr: "(उत्तर मराठीत द्या.)",
};

export function withLanguageInstruction(message: string, locale: Locale): string {
  return `${REPLY_INSTRUCTION[locale]}\n${message}`;
}

export interface AssistantInput {
  /** The user's clean message (no instruction prefix). */
  message: string;
  /** Anonymous visitor id — used as session_id. */
  sessionId: string;
  chatHistory: ChatTurn[];
  locale: Locale;
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

const LANGUAGE_TIMEOUT_MS = 10_000;
/** Remember which (session, locale) pairs we've already told the backend. */
const languageSet = new Set<string>();

/**
 * Tell the backend which language this session should use (`POST /language`).
 * Backend v2 stores the session language, so chat + quiz generation follow it.
 * Best-effort and cached per (session, locale).
 */
async function ensureLanguage(sessionId: string, locale: Locale): Promise<void> {
  const key = `${sessionId}:${locale}`;
  if (languageSet.has(key)) return;
  languageSet.add(key);

  const env = getServerEnv();
  const baseUrl = env.LLM_BACKEND_URL;
  if (!baseUrl) return;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LANGUAGE_TIMEOUT_MS);
  try {
    await fetch(`${baseUrl.replace(/\/+$/, "")}/language`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(env.LLM_BACKEND_API_KEY
          ? { "x-api-key": env.LLM_BACKEND_API_KEY }
          : {}),
      },
      body: JSON.stringify({ session_id: sessionId, language: locale }),
      signal: controller.signal,
      cache: "no-store",
    });
  } catch {
    languageSet.delete(key); // allow a retry next time
  } finally {
    clearTimeout(timer);
  }
}

export async function assistant(input: AssistantInput): Promise<AssistantResponse> {
  const env = getServerEnv();

  console.log(
    `[Vercel->Render] assistant chat invoked: sessionId=${input.sessionId}, locale=${input.locale}, USE_MOCK_BACKEND=${env.USE_MOCK_BACKEND}, LLM_BACKEND_URL=${env.LLM_BACKEND_URL || "(not set)"}`
  );

  if (env.USE_MOCK_BACKEND === "true") {
    console.warn(
      "[Vercel->Render] USE_MOCK_BACKEND is 'true' — bypassing Render and returning mock chat reply."
    );
    return mockAssistant(input);
  }

  const baseUrl = env.LLM_BACKEND_URL;
  if (!baseUrl) {
    console.error(
      "[Vercel->Render] FATAL: LLM_BACKEND_URL is not configured in Vercel environment variables!"
    );
    throw new BackendError("unreachable", "LLM_BACKEND_URL is not set");
  }

  await ensureLanguage(input.sessionId, input.locale);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ASSISTANT_TIMEOUT_MS);
  const cleanBase = baseUrl.replace(/\/+$/, "");
  const targetUrl = `${cleanBase}/chat`;

  console.log(`[Vercel->Render] Dispatching chat fetch() to exact URL: ${targetUrl}`);

  try {
    let response: Response;
    try {
      response = await fetch(targetUrl, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(env.LLM_BACKEND_API_KEY
            ? {
                "x-api-key": env.LLM_BACKEND_API_KEY,
                authorization: `Bearer ${env.LLM_BACKEND_API_KEY}`,
              }
            : {}),
        },
        body: JSON.stringify({
          message: withLanguageInstruction(input.message, input.locale),
          session_id: input.sessionId,
          chat_history: input.chatHistory,
          language: input.locale,
        }),
        signal: controller.signal,
        cache: "no-store",
      });
    } catch (err) {
      console.error(`[Vercel->Render] Network failure calling ${targetUrl}:`, err);
      if (isAbortError(err)) throw new BackendError("timeout", `Timeout connecting to ${targetUrl}`);
      throw new BackendError("unreachable", `Could not connect to ${targetUrl}`);
    }

    // Fallback: If /chat returned 404, retry with /assistant endpoint
    if (response.status === 404) {
      const altUrl = `${cleanBase}/assistant`;
      console.warn(`[Vercel->Render] /chat returned 404. Retrying with exact URL: ${altUrl}`);
      try {
        response = await fetch(altUrl, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...(env.LLM_BACKEND_API_KEY
              ? {
                  "x-api-key": env.LLM_BACKEND_API_KEY,
                  authorization: `Bearer ${env.LLM_BACKEND_API_KEY}`,
                }
              : {}),
          },
          body: JSON.stringify({
            message: withLanguageInstruction(input.message, input.locale),
            session_id: input.sessionId,
            chat_history: input.chatHistory,
            language: input.locale,
          }),
          signal: controller.signal,
          cache: "no-store",
        });
      } catch (err) {
        console.error(`[Vercel->Render] Network failure calling fallback ${altUrl}:`, err);
        if (isAbortError(err)) throw new BackendError("timeout", `Timeout connecting to ${altUrl}`);
        throw new BackendError("unreachable", `Could not connect to ${altUrl}`);
      }
    }

    if (response.status === 401 || response.status === 403) {
      const errText = await response.text().catch(() => "");
      console.error(`[Vercel->Render] Unauthorized (status ${response.status}) on ${targetUrl}: ${errText}`);
      throw new BackendError("unauthorized", undefined, response.status);
    }
    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      console.error(`[Vercel->Render] Upstream error status=${response.status} from ${targetUrl}: ${errText}`);
      throw new BackendError("upstream", undefined, response.status);
    }

    let data: unknown;
    try {
      data = await response.json();
      console.log(`[Vercel->Render] Successfully received 200 OK from Render chat endpoint (${targetUrl})`);
    } catch {
      console.error(`[Vercel->Render] Failed to parse JSON response from ${targetUrl}`);
      throw new BackendError("bad_response", undefined, response.status);
    }

    const parsed = assistantResponseSchema.safeParse(data);
    if (!parsed.success) {
      if (typeof data === "object" && data !== null && "reply" in data) {
        return {
          reply: String((data as { reply: unknown }).reply),
          rag_sources_used: Boolean((data as { rag_sources_used?: unknown }).rag_sources_used),
          live_data_used: Boolean((data as { live_data_used?: unknown }).live_data_used),
        };
      }
      throw new BackendError("bad_response", undefined, response.status);
    }
    return parsed.data;
  } finally {
    clearTimeout(timer);
  }
}

/* ------------------------------------------------------------------ *
 * Safety quiz — backend POST /quiz/generate, POST /quiz/submit
 * (session-keyed; see schemas.ts for the captured contract). The
 * backend's `feedback` / welcome `message` strings never reach our UI.
 * There is no supported-stock list endpoint anymore — the chips on the
 * unsupported card come from the static table in lib/safety.ts.
 * ------------------------------------------------------------------ */

const QUIZ_START_TIMEOUT_MS = 120_000;
const QUIZ_SUBMIT_TIMEOUT_MS = 90_000;

async function callBackend(
  path: string,
  body: unknown | undefined,
  timeoutMs: number
): Promise<unknown> {
  const env = getServerEnv();

  const baseUrl = env.LLM_BACKEND_URL;
  if (!baseUrl) {
    console.error("[Vercel->Render] FATAL: LLM_BACKEND_URL is not set in environment variables!");
    throw new BackendError("unreachable", "LLM_BACKEND_URL is not set");
  }

  const cleanBase = baseUrl.replace(/\/+$/, "");
  const targetUrl = `${cleanBase}${path}`;
  console.log(`[Vercel->Render] Dispatching fetch() to exact Render URL: ${targetUrl} (method: ${body === undefined ? "GET" : "POST"})`);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let response: Response;
    try {
      response = await fetch(targetUrl, {
        method: body === undefined ? "GET" : "POST",
        headers: {
          ...(body !== undefined
            ? { "content-type": "application/json" }
            : {}),
          ...(env.LLM_BACKEND_API_KEY
            ? {
                "x-api-key": env.LLM_BACKEND_API_KEY,
                authorization: `Bearer ${env.LLM_BACKEND_API_KEY}`,
              }
            : {}),
        },
        ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
        signal: controller.signal,
        cache: "no-store",
      });
    } catch (err) {
      console.error(`[Vercel->Render] Network failure connecting to ${targetUrl}:`, err);
      if (isAbortError(err)) throw new BackendError("timeout", `Timeout after ${timeoutMs}ms reaching ${targetUrl}`);
      throw new BackendError("unreachable", `Could not connect to backend at ${targetUrl}`);
    }

    // Fallback: If /quiz/generate returned 404, retry with legacy /quiz endpoint
    if (response.status === 404 && path === "/quiz/generate") {
      const fallbackUrl = `${cleanBase}/quiz`;
      console.warn(`[Vercel->Render] /quiz/generate returned 404. Retrying with exact URL: ${fallbackUrl}`);
      return callBackend("/quiz", body, timeoutMs);
    }

    if (response.status === 401 || response.status === 403) {
      const errText = await response.text().catch(() => "");
      console.error(
        `[Vercel->Render] Unauthorized (status ${response.status}) on ${targetUrl}. Response: ${errText}`
      );
      throw new BackendError("unauthorized", `Unauthorized on ${path} (status ${response.status}): ${errText}`, response.status);
    }

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      console.error(
        `[Vercel->Render] Upstream HTTP error on ${targetUrl} -> Status: ${response.status} ${response.statusText}\nResponse: ${errText}`
      );
      throw new BackendError("upstream", `Upstream error on ${path} (status ${response.status}): ${errText}`, response.status);
    }

    try {
      const json = await response.json();
      console.log(`[Vercel->Render] Successfully received 200 OK from Render URL: ${targetUrl}`);
      return json;
    } catch (parseErr) {
      console.error(`[Vercel->Render] Failed to parse JSON response from ${targetUrl}:`, parseErr);
      throw new BackendError("bad_response", `Invalid JSON returned by ${path}`, response.status);
    }
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Generate the safety quiz for one stock (POST /quiz/generate).
 * The backend keys the quiz by `session_id` (our sg_vid visitor id); the
 * README documents a `quiz_id` in the response, which we capture when
 * present. `language` is undeclared in OpenAPI (the model may ignore it);
 * we still send it, along with the PRD B5 personalisation extras
 * (`tip_source` / `scan_summary` / `fomo_score` / `fomo_band`) — FastAPI
 * ignores unknown body fields unless the backend opts in (backend-contract).
 */
export async function safetyQuizStart(input: {
  sessionId: string;
  targetStock: string;
  locale: Locale;
  tipSource?: string | null;
  scanSummary?: ScanSummaryPayload | null;
  fomo?: { score: number; band: string } | null;
}): Promise<QuizStartResponse> {
  const env = getServerEnv();

  console.log(
    `[Vercel->Render] safetyQuizStart invoked: targetStock=${input.targetStock}, locale=${input.locale}, USE_MOCK_BACKEND=${env.USE_MOCK_BACKEND}, LLM_BACKEND_URL=${env.LLM_BACKEND_URL || "(not set)"}`
  );

  if (env.USE_MOCK_BACKEND === "true") {
    console.warn(
      "[Vercel->Render] USE_MOCK_BACKEND is 'true' — bypassing Render and returning mock quiz."
    );
    return mockQuizStart(input);
  }

  await ensureLanguage(input.sessionId, input.locale);

  const payload = {
    session_id: input.sessionId,
    target_stock: input.targetStock,
    language: input.locale,
    ...(input.tipSource ? { tip_source: input.tipSource } : {}),
    ...(input.scanSummary ? { scan_summary: input.scanSummary } : {}),
    ...(input.fomo
      ? { fomo_score: input.fomo.score, fomo_band: input.fomo.band }
      : {}),
  };

  const data = await callBackend("/quiz/generate", payload, QUIZ_START_TIMEOUT_MS);
  const normalized = normalizeQuizStart(data);
  if (!normalized) {
    console.error(`[Vercel->Render] normalizeQuizStart failed! Raw backend payload was:`, JSON.stringify(data));
    throw new BackendError("bad_response", "Failed to normalize quiz payload from backend");
  }
  return normalized;
}

/** Grade the quiz (POST /quiz/submit). `feedback` is never returned. */
export async function safetyQuizSubmit(input: {
  sessionId: string;
  answers: string[];
  locale: Locale;
  quizId?: string | null;
  tipSource?: string | null;
  scanSummary?: ScanSummaryPayload | null;
  fomo?: { score: number; band: string } | null;
}): Promise<QuizSubmitResult> {
  const env = getServerEnv();
  if (env.USE_MOCK_BACKEND === "true") {
    return mockQuizSubmit(input);
  }

  await ensureLanguage(input.sessionId, input.locale);

  let data: unknown;
  try {
    data = await callBackend(
      "/quiz/submit",
      {
        session_id: input.sessionId,
        language: input.locale,
        answers: input.answers.map((letter, index) => ({
          question_index: index,
          selected_option: letter,
        })),
        ...(input.quizId ? { quiz_id: input.quizId } : {}),
        ...(input.tipSource ? { tip_source: input.tipSource } : {}),
        ...(input.scanSummary ? { scan_summary: input.scanSummary } : {}),
        ...(input.fomo ? { fomo_score: input.fomo.score } : {}),
      },
      QUIZ_SUBMIT_TIMEOUT_MS
    );
  } catch (err) {
    // 404 = "Quiz session not found." — the session expired or was reset.
    if (
      err instanceof BackendError &&
      err.kind === "upstream" &&
      err.status === 404
    ) {
      throw new BackendError("quiz_expired", undefined, 404);
    }
    throw err;
  }
  const normalized = normalizeQuizSubmit(data, input.answers.length);
  if (!normalized) throw new BackendError("bad_response");
  return normalized;
}
