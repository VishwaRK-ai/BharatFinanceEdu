import { NextResponse, type NextRequest } from "next/server";
import { db } from "@/db";
import { safetyAttempts } from "@/db/schema";
import { quizSubmitRequestSchema } from "@/lib/backend/schemas";
import { recomputeFomo } from "@/lib/fomo-recompute";
import { rateLimit } from "@/lib/rate-limit";
import { getSavedQuiz, gradeQuiz } from "@/lib/safety";
import { apiRequireVisitor } from "@/lib/visitor";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

/**
 * Grade the quiz against the backend, compute our verdict (PRD B5 rules —
 * a HIGH-risk scan caps it at SPECULATIVE), and store the attempt:
 * answers as letters + {eligible, correct_answers} plus the tip source and
 * the scan summary (risk level + flag TITLES only — no images, no OCR text,
 * no AI strings). Returns the result and the attempt id for the result URL.
 * The backend's `feedback` string is never returned to the client.
 */
export async function POST(request: NextRequest) {
  const auth = apiRequireVisitor(request);
  if ("response" in auth) return auth.response;
  const { visitorId } = auth;

  const limit = rateLimit(`safety-submit:${visitorId}`, 10, 60_000);
  if (!limit.ok) {
    return NextResponse.json(
      { error: "rate_limited", message: "Too many submissions. Wait a minute." },
      {
        status: 429,
        headers: { "Retry-After": String(limit.retryAfterSeconds) },
      }
    );
  }

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return NextResponse.json(
      { error: "invalid_body", message: "Request body must be valid JSON." },
      { status: 400 }
    );
  }
  const parsed = quizSubmitRequestSchema.safeParse(raw);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "invalid_body",
        message: "Five A–D answers and a ticker are required.",
      },
      { status: 400 }
    );
  }

  const { answers, ticker, locale, quizId, tipSource, scanSummary } = parsed.data;

  try {
    // Fetch the stored quiz from Neon PostgreSQL (completely stateless from Python backend)
    const savedQuiz = await getSavedQuiz({
      quizId: quizId ?? null,
      visitorId,
      ticker,
    });

    if (!savedQuiz) {
      return NextResponse.json(
        {
          error: "quiz_expired",
          message: "Quiz session not found or expired. Please generate a new quiz.",
        },
        { status: 404 }
      );
    }

    // Grade submission locally against the database record
    const graded = gradeQuiz(
      savedQuiz.questions,
      answers,
      savedQuiz.stockName || ticker,
      scanSummary?.risk_level ?? null
    );

    const inserted = await db
      .insert(safetyAttempts)
      .values({
        visitorId,
        ticker,
        tipSource: tipSource ?? null,
        scanRiskLevel: scanSummary?.risk_level ?? null,
        scanFlags: scanSummary?.flags ?? null,
        quizId: quizId ?? savedQuiz.quizId,
        answers,
        score: graded.score,
        total: graded.total,
        verdictCode: graded.level,
        result: {
          level: graded.level,
          verdict: graded.verdictText,
          strengths: graded.strengths,
          gaps: graded.gaps,
          feedback: graded.feedback,
          percentage: graded.percentage,
        },
      })
      .returning({ id: safetyAttempts.id });

    // Completing a safety quiz is one of the FOMO recompute events.
    try {
      await recomputeFomo(visitorId);
    } catch {
      // best-effort — never fail the attempt because a recompute could not run
    }

    return NextResponse.json(
      {
        attempt_id: inserted[0].id,
        score: graded.score,
        total: graded.total,
        eligible: graded.eligible,
        correct_answers: graded.correctAnswers,
        verdict: graded.verdict,
      },
      { headers: { "cache-control": "no-store" } }
    );
  } catch (err) {
    console.error("[/api/safety-quiz/submit] Failed to evaluate quiz submission:", err);
    return NextResponse.json(
      { error: "submission_failed", message: "Failed to grade quiz submission." },
      { status: 500 }
    );
  }
}
