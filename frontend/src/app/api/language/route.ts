import { NextResponse, type NextRequest } from "next/server";
import { isLocale, type Locale } from "@/i18n/routing";
import { LANG_COOKIE, INTL_LOCALE_COOKIE } from "@/lib/lang-cookie";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const YEAR = 60 * 60 * 24 * 365;

export async function POST(request: NextRequest) {
  let locale: Locale = "en";
  try {
    const body = await request.json();
    const candidate = body.locale || body.language;
    if (isLocale(candidate)) {
      locale = candidate;
    }
  } catch {
    // default to en
  }

  const response = NextResponse.json({ ok: true, locale });
  const isProd = process.env.NODE_ENV === "production";

  response.cookies.set(LANG_COOKIE, locale, {
    path: "/",
    maxAge: YEAR,
    sameSite: "lax",
    secure: isProd,
    httpOnly: false,
  });

  response.cookies.set(INTL_LOCALE_COOKIE, locale, {
    path: "/",
    maxAge: YEAR,
    sameSite: "lax",
    secure: isProd,
    httpOnly: false,
  });

  return response;
}
