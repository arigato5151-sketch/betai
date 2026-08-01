const OUTCOME_PREDICTION_LABELS = Object.freeze({
  HOME_WIN: "Ev Sahibi Kazanır",
  DRAW: "Beraberlik",
  AWAY_WIN: "Deplasman Kazanır",
});

const OUTCOME_RESULT_LABELS = Object.freeze({
  HOME_WIN: "Ev Sahibi Kazandı",
  DRAW: "Berabere Bitti",
  AWAY_WIN: "Deplasman Kazandı",
});

const ML_SAFETY_LABELS = Object.freeze({
  HIGH_CONFIDENCE: "Yüksek Güven",
  MEDIUM_CONFIDENCE: "Orta Güven",
  LOW_CONFIDENCE: "Düşük Güven",
  UPSET_CANDIDATE: "Sürpriz Adayı",
  RISKY_UPSET: "Riskli Sürpriz",
  RISKY_UNDERDOG: "Riskli Sürpriz",
  INSUFFICIENT_DATA: "Yetersiz Veri",
  "YETERLI VERI YOK": "Yetersiz Veri",
  "YETERLİ VERİ YOK": "Yetersiz Veri",
});

const ROLE_LABELS = Object.freeze({
  admin: "Yönetici",
  analyst: "Analist",
  viewer: "Görüntüleyici",
});

const LEAGUE_LABELS = Object.freeze({
  2: "UEFA Şampiyonlar Ligi",
  3: "UEFA Avrupa Ligi",
  848: "UEFA Konferans Ligi",
});

const PERMISSION_LABELS = Object.freeze({
  "analysis:create": "Analiz oluşturma",
  "history:read": "Tahmin geçmişini görüntüleme",
  "history:update_result": "Gerçek sonucu güncelleme",
  "backtest:run": "Geriye dönük test çalıştırma",
  "audit:read": "Denetim kayıtlarını görüntüleme",
  "users:manage": "Kullanıcıları yönetme",
  "roles:manage": "Rolleri yönetme",
});

const DATA_QUALITY_STATUS_LABELS = Object.freeze({
  healthy: "Sağlıklı",
  warning: "Uyarı",
  critical: "Kritik",
});

const SYNC_STATUS_LABELS = Object.freeze({
  succeeded: "Başarılı",
  partial: "Kısmen Başarılı",
  failed: "Başarısız",
  running: "Sürüyor",
  pending: "Bekliyor",
});

const MODEL_NAME_LABELS = Object.freeze({
  "Random Forest": "Rastgele Orman",
  random_forest: "Rastgele Orman",
  calibrated_random_forest: "Kalibre Edilmiş Rastgele Orman",
  "Calibrated Model": "Kalibre Edilmiş Model",
});

const BACKTEST_REASON_LABELS = Object.freeze({
  post_kickoff_analysis: "Maç başladıktan sonra yapılan analiz",
  below_edge: "Asgari avantaj eşiğinin altında",
  invalid_odds: "Geçersiz oran",
  missing_closing_odds: "Kapanış oranı eksik",
  daily_exposure_limit: "Günlük risk sınırı",
});

const hasValue = (value) =>
  value !== null && value !== undefined && String(value).trim() !== "";

const lookup = (labels, value, fallback) => {
  if (!hasValue(value)) return fallback;
  return labels[String(value)] ?? String(value);
};

const lookupClosedSet = (labels, value, fallback) => {
  if (!hasValue(value)) return fallback;
  return labels[String(value)] ?? fallback;
};

export const predictionLabel = (value) =>
  lookupClosedSet(OUTCOME_PREDICTION_LABELS, value, "Tahmin Bilinmiyor");

export const resultLabel = (value) =>
  lookupClosedSet(OUTCOME_RESULT_LABELS, value, "Sonuç Bilinmiyor");

export const mlSafetyLabel = (value) =>
  lookupClosedSet(ML_SAFETY_LABELS, value, "Durum Bilinmiyor");

export const mlSafetyTone = (value) => {
  if (value === "HIGH_CONFIDENCE" || value === "MEDIUM_CONFIDENCE") {
    return "positive";
  }
  if (
    value === "INSUFFICIENT_DATA" ||
    value === "YETERLI VERI YOK" ||
    value === "YETERLİ VERİ YOK"
  ) {
    return "neutral";
  }
  if (value === "UPSET_CANDIDATE") return "warning";
  return "negative";
};

export const roleLabel = (value) => lookup(ROLE_LABELS, value, "Rol Yok");

export const leagueLabel = (league) =>
  LEAGUE_LABELS[league?.id] ?? league?.name ?? "Bilinmeyen Lig";

export const permissionLabel = (value) =>
  lookup(PERMISSION_LABELS, value, "Yetki Yok");

export const dataQualityStatusLabel = (value) =>
  lookupClosedSet(
    DATA_QUALITY_STATUS_LABELS,
    hasValue(value) ? String(value).toLowerCase() : value,
    "Bilinmiyor",
  );

export const syncStatusLabel = (value) =>
  lookupClosedSet(
    SYNC_STATUS_LABELS,
    hasValue(value) ? String(value).toLowerCase() : value,
    "Senkron Yok",
  );

export const modelNameLabel = (value) =>
  lookup(MODEL_NAME_LABELS, value, "Model Yok");

export const backtestReasonLabel = (value) =>
  lookupClosedSet(BACKTEST_REASON_LABELS, value, "Bilinmeyen neden");

export const matchLabel = (value) =>
  hasValue(value)
    ? String(value).replace(/\s+vs\.?\s+/gi, " – ")
    : "Maç Bilgisi Yok";
