import { getTranslations, setRequestLocale } from "next-intl/server";
import { PlaceholderPage } from "@/components/features/shell/PlaceholderPage";

export default async function DashboardPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);

  const t = await getTranslations("dashboard");
  const tc = await getTranslations("common");

  return (
    <PlaceholderPage
      title={t("title")}
      subtitle={t("subtitle")}
      badge={tc("comingSoon")}
    >
      <p className="text-sm leading-relaxed text-white/60">{t("body")}</p>
    </PlaceholderPage>
  );
}
