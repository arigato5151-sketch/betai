import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  backtestReasonLabel,
  dataQualityStatusLabel,
  eligibilityReasonLabel,
  leagueLabel,
  matchLabel,
  mlSafetyLabel,
  mlSafetyTone,
  modelNameLabel,
  permissionLabel,
  predictionLabel,
  resultLabel,
  roleLabel,
  syncStatusLabel,
} from "./localization.js";

test("API sonuç kodlarını bağlama uygun Türkçe etiketlere dönüştürür", () => {
  assert.equal(predictionLabel("HOME_WIN"), "Ev Sahibi Kazanır");
  assert.equal(predictionLabel("DRAW"), "Beraberlik");
  assert.equal(resultLabel("HOME_WIN"), "Ev Sahibi Kazandı");
  assert.equal(resultLabel("AWAY_WIN"), "Deplasman Kazandı");
  assert.equal(resultLabel(null), "Sonuç Bilinmiyor");
  assert.equal(predictionLabel("UNKNOWN"), "Tahmin Bilinmiyor");
});

test("model güveni ve operasyon durumlarını Türkçeleştirir", () => {
  assert.equal(mlSafetyLabel("HIGH_CONFIDENCE"), "Yüksek Güven");
  assert.equal(mlSafetyLabel("MEDIUM_CONFIDENCE"), "Orta Güven");
  assert.equal(mlSafetyLabel("UPSET_CANDIDATE"), "Sürpriz Adayı");
  assert.equal(mlSafetyLabel("RISKY_UPSET"), "Riskli Sürpriz");
  assert.equal(mlSafetyLabel("INSUFFICIENT_DATA"), "ML Modeli Hazır Değil");
  assert.equal(mlSafetyLabel("YETERLI VERI YOK"), "ML Modeli Hazır Değil");
  assert.equal(mlSafetyTone("HIGH_CONFIDENCE"), "positive");
  assert.equal(mlSafetyTone("INSUFFICIENT_DATA"), "neutral");
  assert.equal(mlSafetyTone("UPSET_CANDIDATE"), "warning");
  assert.equal(mlSafetyTone("RISKY_UNDERDOG"), "negative");
  assert.equal(dataQualityStatusLabel("CRITICAL"), "Kritik");
  assert.equal(syncStatusLabel("succeeded"), "Başarılı");
  assert.equal(syncStatusLabel(undefined), "Senkron Yok");
  assert.equal(syncStatusLabel("unknown"), "Senkron Yok");
});

test("ABSTAIN nedenlerini kullanıcıya Türkçe açıklar", () => {
  assert.equal(
    eligibilityReasonLabel("home_history_insufficient"),
    "Ev sahibinin yakın dönem maç geçmişi yetersiz",
  );
  assert.equal(
    eligibilityReasonLabel("market_unavailable"),
    "Güncel 1X2 oranları bulunamadı",
  );
});

test("rol, yetki, model ve test nedenlerini Türkçeleştirir", () => {
  assert.equal(roleLabel("admin"), "Yönetici");
  assert.equal(roleLabel("viewer"), "Görüntüleyici");
  assert.equal(
    permissionLabel("history:update_result"),
    "Gerçek sonucu güncelleme",
  );
  assert.equal(modelNameLabel("Random Forest"), "Rastgele Orman");
  assert.equal(modelNameLabel("CatBoost"), "CatBoost");
  assert.equal(
    backtestReasonLabel("missing_closing_odds"),
    "Kapanış oranı eksik",
  );
  assert.equal(backtestReasonLabel("future_reason"), "Bilinmeyen neden");
});

test("UEFA liglerini arayüzde güncel Türkçe adlarıyla gösterir", () => {
  assert.equal(
    leagueLabel({ id: 2, name: "UEFA Champions League" }),
    "UEFA Şampiyonlar Ligi",
  );
  assert.equal(
    leagueLabel({ id: 848, name: "UEFA Europa Conference League" }),
    "UEFA Konferans Ligi",
  );
  assert.equal(leagueLabel({ id: 39, name: "Premier League" }), "Premier League");
  assert.equal(
    leagueLabel({ id: 179, name: "Scottish Premiership" }),
    "İskoçya Premiership",
  );
  assert.equal(
    leagueLabel({ id: 218, name: "Austrian Bundesliga" }),
    "Avusturya Bundesliga",
  );
  assert.equal(
    leagueLabel({ id: 207, name: "Swiss Super League" }),
    "İsviçre Süper Ligi",
  );
  assert.equal(
    leagueLabel({ id: 197, name: "Super League 1" }),
    "Yunanistan Süper Ligi",
  );
  assert.equal(
    leagueLabel({ id: 119, name: "Superliga" }),
    "Danimarka Süper Ligi",
  );
  assert.equal(leagueLabel(null), "Bilinmeyen Lig");
});

test("maç ayırıcısını Türkçe arayüz biçimine getirir", () => {
  assert.equal(
    matchLabel("Fenerbahçe vs Galatasaray"),
    "Fenerbahçe – Galatasaray",
  );
  assert.equal(
    matchLabel("Fenerbahçe – Galatasaray"),
    "Fenerbahçe – Galatasaray",
  );
  assert.equal(matchLabel(undefined), "Maç Bilgisi Yok");
});

const collectJsxFiles = (directory) =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return collectJsxFiles(path);
    return entry.isFile() && entry.name.endsWith(".jsx") ? [path] : [];
  });

test("JSX kaynaklarında bozuk UTF-8 dizisi bırakmaz", () => {
  const sourceRoot = dirname(fileURLToPath(import.meta.url));
  for (const path of collectJsxFiles(sourceRoot)) {
    assert.doesNotMatch(readFileSync(path, "utf8"), /Ã|Ä|Å|Â|â€|â€™/);
  }
});
