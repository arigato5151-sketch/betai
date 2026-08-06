import { useMemo } from "react";
import { ArcElement, Chart as ChartJS, Tooltip } from "chart.js";
import { Doughnut } from "react-chartjs-2";

import {
  eligibilityReasonLabel,
  matchLabel,
  mlSafetyLabel,
  mlSafetyTone,
  modelNameLabel,
  predictionLabel,
  resultLabel,
} from "../localization.js";

ChartJS.register(ArcElement, Tooltip);

const asProbability = (value) => {
  const probability = Number(value);
  return Number.isFinite(probability) && probability >= 0 && probability <= 100
    ? probability
    : null;
};

const secondaryMarket = (analysis, market) =>
  Array.isArray(analysis?.secondary_markets)
    ? analysis.secondary_markets.find((item) => item?.market === market)
    : null;

export function buildAlternativeResults(analysis) {
  const results = [];
  const probabilities = analysis?.all_probabilities ?? {};
  const home = asProbability(probabilities.HOME_WIN);
  const draw = asProbability(probabilities.DRAW);
  const away = asProbability(probabilities.AWAY_WIN);

  if (home !== null && draw !== null && away !== null) {
    const total = home + draw + away;
    if (total > 0) {
      const normalized = {
        home: (home / total) * 100,
        draw: (draw / total) * 100,
        away: (away / total) * 100,
      };
      const doubleChances = [
        { label: "1-X", probability: normalized.home + normalized.draw },
        { label: "X-2", probability: normalized.draw + normalized.away },
        { label: "1-2", probability: normalized.home + normalized.away },
      ].sort((left, right) => right.probability - left.probability);
      results.push({
        key: "double_chance",
        title: "Çifte Şans",
        value: doubleChances[0].label,
        probability: Number(doubleChances[0].probability.toFixed(2)),
      });
    }
  }

  const over25 = secondaryMarket(analysis, "OVER_2_5");
  if (over25 && asProbability(over25.probability) !== null) {
    results.push({
      key: "over_2_5",
      title: "Toplam 2.5 Gol",
      value: over25.pick === "UST" ? "Üst" : "Alt",
      probability: asProbability(over25.probability),
    });
  }

  const btts = secondaryMarket(analysis, "BTTS");
  if (btts && asProbability(btts.probability) !== null) {
    results.push({
      key: "btts",
      title: "Karşılıklı Gol",
      value: btts.pick === "VAR" ? "Var" : "Yok",
      probability: asProbability(btts.probability),
    });
  }

  const over15 = secondaryMarket(analysis, "OVER_1_5");
  if (over15 && asProbability(over15.probability) !== null) {
    results.push({
      key: "over_1_5",
      title: "1.5 Gol Üst",
      value: "Üst",
      probability: asProbability(over15.probability),
    });
  }

  const score = analysis?.expected_score;
  if (
    Number.isInteger(score?.home) &&
    Number.isInteger(score?.away) &&
    asProbability(score?.probability) !== null
  ) {
    results.push({
      key: "score",
      title: "En Olası Skor",
      value: `${score.home}-${score.away}`,
      probability: asProbability(score.probability),
    });
  }

  return results;
}

function AnalysisReport({ canUpdateResult, match, onSubmitActualResult }) {
  const safetyTone = mlSafetyTone(match.ml_safety_trigger);
  const eligibility = match.data_quality?.prediction_eligibility;
  const abstained = eligibility?.status === "abstain";
  const scenario = match.provenance?.analysis_origin === "scenario";
  const automaticAbstain =
    abstained && match.provenance?.analysis_origin === "automatic";
  const limitedAnalysis = abstained && !automaticAbstain && !scenario;
  const alternativeResults = useMemo(
    () => buildAlternativeResults(match.analysis),
    [match.analysis],
  );
  const chartData = useMemo(
    () => ({
      labels: ["Ev Sahibi", "Deplasman", "Beraberlik"],
      datasets: [
        {
          data: [
            match.analysis.all_probabilities.HOME_WIN,
            match.analysis.all_probabilities.AWAY_WIN,
            match.analysis.all_probabilities.DRAW,
          ],
          backgroundColor: ["#fbbf24", "#ef4444", "#4b5563"],
          borderWidth: 0,
        },
      ],
    }),
    [match],
  );

  return (
    <div className="grid grid-cols-1 gap-6 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl md:grid-cols-2">
      <div>
        <span className="text-xs font-bold uppercase tracking-widest text-emerald-400">
          Aktif Analiz Raporu
        </span>
        <h2 className="mb-4 mt-1 text-xl font-black text-white">
          {matchLabel(match.match)}
        </h2>

        {(automaticAbstain || limitedAnalysis || scenario) && (
          <div className="mb-4 rounded border border-amber-700 bg-amber-950/40 p-3 text-sm text-amber-200">
            <strong className="block">
              {scenario
                ? "Senaryo analizi"
                : automaticAbstain
                  ? "Otomatik tahmin verilmedi (ABSTAIN)"
                  : "Sınırlı veriyle istatistik analizi"}
            </strong>
            <span className="mt-1 block text-xs text-amber-300/80">
              {scenario
                ? "Manuel değişiklik içerir; eğitim ve performans hesaplarına katılmaz."
                : automaticAbstain
                  ? "Otomatik karar için gerekli veri kalitesi sağlanmadı; kayıt oluşturulmadı."
                  : "Temel maç tahmini üretildi, ancak eksik oran/geçmiş verisi nedeniyle finansal değer hesabı kullanılmamalıdır."}
            </span>
            {abstained && eligibility.reasons?.length > 0 && (
              <span className="mt-1 block text-xs text-slate-400">
                Nedenler: {eligibility.reasons.map(eligibilityReasonLabel).join(", ")}
              </span>
            )}
          </div>
        )}

        <div className="space-y-3">
          <div>
            <span className="text-xs text-slate-400">Yapay Zekâ Tahmini</span>
            <p className="text-lg font-bold text-amber-400">
              {predictionLabel(match.analysis.prediction)} (%
              {match.analysis.probability})
            </p>
          </div>
          {match.data_quality && (
            <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs">
              <span className="text-slate-500">Analiz veri skoru</span>
              <strong className="ml-2 text-emerald-400">
                {match.data_quality.score}/100
              </strong>
              {match.provenance?.model_name && (
                <p className="mt-2 text-slate-400">
                  Model: {modelNameLabel(match.provenance.model_name)}
                  {match.provenance.model_artifact_version
                    ? ` · ${match.provenance.model_artifact_version}`
                    : ""}
                </p>
              )}
            </div>
          )}
          <div>
            <span className="text-xs text-slate-400">
              Finansal Değer / Değerli Bahis
            </span>
            <p
              className={`text-sm font-bold ${
                match.value_assessment.value_bet
                  ? "text-emerald-400"
                  : "text-slate-400"
              }`}
            >
              {abstained
                ? "VERİ YETERSİZ — DEĞER HESABI KULLANILMAMALI"
                : match.value_assessment.value_bet
                ? `DEĞERLİ ORAN BULUNDU (+%${match.value_assessment.edge})`
                : `Değerli Oran Bulunamadı (%${match.value_assessment.edge})`}
            </p>
          </div>
          <div>
            <span className="text-xs text-slate-400">
              Model Katmanı
            </span>
            <p className="mt-1 text-xs">
              <span
                className={`rounded px-2 py-1 font-bold ${
                  safetyTone === "positive"
                    ? "bg-emerald-950 text-emerald-400"
                    : safetyTone === "neutral"
                      ? "bg-slate-800 text-slate-400"
                      : safetyTone === "warning"
                        ? "bg-amber-950 text-amber-400"
                        : "bg-red-950 text-red-400"
                }`}
              >
                {mlSafetyLabel(match.ml_safety_trigger)}
              </span>
              {match.ml_samples !== undefined && (
                <span className="ml-2 text-slate-500">
                  ({match.ml_samples}/{match.ml_min_samples ?? 200} model eğitim örneği)
                </span>
              )}
            </p>
            {!match.ml_ready && (
              <p className="mt-2 rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] leading-relaxed text-slate-400">
                ML modeli henüz aktif değil. Bu maçın analiz edilemediği anlamına gelmez;
                yukarıdaki tahmin zaman ağırlıklı Poisson / Dixon-Coles istatistik motorundan gelir.
              </p>
            )}
            {match.ml_safety_details?.ml_confidence !== undefined && (
              <p className="mt-2 text-[11px] text-slate-500">
                ML %{match.ml_safety_details.ml_confidence} · olasılık farkı %
                {match.ml_safety_details.confidence_gap}
              </p>
            )}
          </div>
          {canUpdateResult && match.record_id && !match.actual_result && (
            <div className="flex flex-wrap gap-2 pt-2">
              <span className="w-full text-xs text-slate-500">
                Gerçek sonucu girin:
              </span>
              {["HOME_WIN", "DRAW", "AWAY_WIN"].map((result) => (
                <button
                  key={result}
                  type="button"
                  onClick={() => onSubmitActualResult(match.record_id, result)}
                  className="rounded border border-slate-700 px-2 py-1 text-xs hover:border-emerald-500"
                >
                  {resultLabel(result)}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-col items-center justify-center border-t border-slate-800 pt-4 md:border-l md:border-t-0 md:pt-0">
        <div className="h-40 w-40">
          <Doughnut
            data={chartData}
            options={{
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
            }}
          />
        </div>
        <div className="mt-4 flex gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-amber-400" /> Ev
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-red-400" /> Deplasman
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-gray-500" /> Beraberlik
          </span>
        </div>
      </div>
      {(alternativeResults.length > 0 || match.analysis.expected_goals) && (
        <section className="border-t border-slate-800 pt-5 md:col-span-2">
          <h3 className="mb-3 text-sm font-black text-slate-200">
            Alternatif Analiz Sonuçları
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {match.analysis.expected_goals && (
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <span className="text-[11px] text-slate-500">Beklenen Gol</span>
                <p className="mt-1 font-black text-sky-300">
                  {match.analysis.expected_goals.home} – {match.analysis.expected_goals.away}
                </p>
                <span className="text-[10px] text-slate-500">
                  Toplam {match.analysis.expected_goals.total} xG
                </span>
              </div>
            )}
            {match.analysis.score_band && (
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <span className="text-[11px] text-slate-500">Gol Aralığı</span>
                <p className="mt-1 font-black text-slate-200">
                  {match.analysis.score_band}
                </p>
              </div>
            )}
            {alternativeResults.map((result) => (
              <div
                key={result.key}
                className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
              >
                <span className="text-[11px] text-slate-500">{result.title}</span>
                <p className="mt-1 font-black text-emerald-300">
                  {result.value} · %{result.probability}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[10px] text-slate-500">
            Gol ve skor seçenekleri Poisson/Dixon-Coles dağılımından; çifte şans
            final ensemble 1X2 olasılıklarından hesaplanır.
          </p>
        </section>
      )}
    </div>
  );
}

export default AnalysisReport;
