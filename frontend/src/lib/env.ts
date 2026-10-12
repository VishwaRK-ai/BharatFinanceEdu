import { z } from "zod";

/**
 * Server-only environment validation.
 *
 * Validation runs lazily on first access of `getServerEnv()` (not at import),
 * so `next build` never breaks just because this module is loaded.
 * When USE_MOCK_BACKEND === "true", the LLM backend variables are optional.
 */

// Normalize LLM_BACKEND_URL: remove whitespace, outer quotes, trailing slashes
function sanitizeUrl(url?: string): string | undefined {
  if (!url) return undefined;
  const trimmed = url.trim().replace(/^["']|["']$/g, "").trim();
  return trimmed ? trimmed.replace(/\/+$/, "") : undefined;
}

const envSchema = z
  .object({
    DATABASE_URL: z.string().min(1, "DATABASE_URL is required (Neon pooled string)"),
    DIRECT_URL: z.string().min(1, "DIRECT_URL is required (Neon direct string)"),
    LLM_BACKEND_URL: z.string().optional().transform(sanitizeUrl),
    NEXT_PUBLIC_BACKEND_URL: z.string().optional().transform(sanitizeUrl),
    BACKEND_URL: z.string().optional().transform(sanitizeUrl),
    LLM_BACKEND_API_KEY: z.string().optional(),
    USE_MOCK_BACKEND: z.enum(["true", "false"]).optional(),
    NEXT_PUBLIC_APP_URL: z.string().default("http://localhost:3000"),
  })
  .transform((val) => {
    const rawBackendUrl =
      val.LLM_BACKEND_URL ||
      val.NEXT_PUBLIC_BACKEND_URL ||
      val.BACKEND_URL ||
      sanitizeUrl(process.env.NEXT_PUBLIC_BACKEND_URL) ||
      sanitizeUrl(process.env.BACKEND_URL) ||
      sanitizeUrl(process.env.LLM_BACKEND_URL);

    // If USE_MOCK_BACKEND is explicitly set, respect it.
    // If not set, but any backend URL is provided, default to "false" (send requests to Render).
    // Otherwise, default to "true" (local mock backend).
    let effectiveUseMock = val.USE_MOCK_BACKEND;
    if (!effectiveUseMock) {
      effectiveUseMock = rawBackendUrl ? "false" : "true";
    }
    return {
      ...val,
      LLM_BACKEND_URL: rawBackendUrl,
      USE_MOCK_BACKEND: effectiveUseMock,
    };
  })
  .superRefine((val, ctx) => {
    if (val.USE_MOCK_BACKEND === "false" && !val.LLM_BACKEND_URL) {
      ctx.addIssue({
        code: "custom",
        path: ["LLM_BACKEND_URL"],
        message:
          "LLM_BACKEND_URL or NEXT_PUBLIC_BACKEND_URL is required when USE_MOCK_BACKEND is \"false\". Set NEXT_PUBLIC_BACKEND_URL or LLM_BACKEND_URL to your Render backend URL in Vercel.",
      });
    }
  });

export type ServerEnv = z.infer<typeof envSchema>;

let cached: ServerEnv | null = null;

export function getServerEnv(): ServerEnv {
  if (cached) return cached;

  if (typeof window !== "undefined") {
    throw new Error(
      "getServerEnv() is server-only — never call it from client components."
    );
  }

  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const lines = parsed.error.issues.map(
      (issue) => `  - ${issue.path.join(".") || "(root)"}: ${issue.message}`
    );
    throw new Error(
      `Invalid environment configuration:\n${lines.join("\n")}\n` +
        `Copy .env.example to .env.local and fill in the values.`
    );
  }

  cached = parsed.data;
  return cached;
}
