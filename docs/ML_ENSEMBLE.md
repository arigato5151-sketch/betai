# ML Model Havuzu ve Lig Bazlı BMA

## Aday modeller

Eğitim havuzu aşağıdaki çok sınıflı adayları aynı kronolojik walk-forward
pencerelerinde Brier Score ve log-loss ile karşılaştırır:

- Regularized Logistic Regression
- Gradient Boosting veya kuruluysa XGBoost
- Random Forest
- CatBoost `1.2.10`
- LightGBM `4.7.0`

CatBoost ve LightGBM varsayılan olarak açıktır. Her iki aday da
`league_id`, `home_team_id` ve `away_team_id` alanlarını ham sayısal büyüklük
olarak kullanmak yerine kategorik olarak işler. CatBoost string kategorileri
native target-statistics akışına, LightGBM ise eğitimde öğrenilen ve inference
sırasında bilinmeyen değerleri missing olarak ele alan pandas kategorilerine
çevirir. Logistic Regression, Gradient Boosting, XGBoost ve Random Forest ham
ID kolonlarını görmez; mevcut lig one-hot feature'larını kullanmaya devam eder.

Eski `ml_features_v1-v5` snapshot'ları desteklenir. Eksik kategorik ID'ler `0`
unknown token'ına dönüşür. Yeni snapshot sürümü `ml_features_v6`'dır.

## Lig bazlı Bayesian Model Averaging

Ensemble, mevcut kaynak seti için ayrı posterior tutar:

- `stats+ml`
- `stats+market`
- `stats+ml+market`

Her çözülmüş pre-match tahminde kaynak ağırlığı, gerçekleşen sonucun kaynak
tarafından verilen olasılığıyla güncellenir:

```text
log_weight[source] += data_quality * log(P_source(actual_result))
```

Uzun veri boşluklarında posterior, yapılandırılmış prior'a yarı ömürlü olarak
geri döner. Yeterli ve düşük sürprizli liglerde ML prior payı yükselir. Az verili
veya henüz görülmemiş liglerde stats/Poisson prior'ı otomatik güçlendirilir.
Kaynak ağırlıkları `ENSEMBLE_MIN_SOURCE_WEIGHT` tabanının altına inemez.

Aktivasyon kronolojik holdout üzerinde yapılır. Global BMA, yapılandırılmış
baseline'dan gerekli log-loss iyileşmesini sağlamazsa artifact reddedilir. Bir
lig profili de global profile karşı aynı kontrolü ve Brier regresyon sınırını
geçmeden aktive edilmez.

Fallback sırası:

1. Lig ve kaynak setine özel BMA
2. Az veri için stats-ağırlıklı prior
3. Lig belirtilmemişse global BMA
4. Eski schema-v1 global artifact
5. Yapılandırılmış sabit ağırlıklar

`ensemble_weights.json` schema-v2 artifact'ı, `MODEL_SIGNING_KEY` ile domain
separation uygulanmış HMAC-SHA256 imzasını aynı atomik JSON içinde taşır.
İmza uyuşmazsa artifact kullanılmaz. ML model artifact'larının detached `.sig`
doğrulaması ve previous-model rollback akışı değişmeden korunur.

## Yapılandırma

```env
ENABLE_CATBOOST_CANDIDATE=true
ENABLE_LIGHTGBM_CANDIDATE=true
ML_BOOSTER_TREES=200
ML_BOOSTER_MAX_DEPTH=6
ML_BOOSTER_LEARNING_RATE=0.05
ML_BOOSTER_THREADS=2

ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES=30
ENSEMBLE_BMA_PRIOR_STRENGTH=50
ENSEMBLE_BMA_HALF_LIFE_DAYS=180
ENSEMBLE_BMA_MIN_DATA_QUALITY_SCORE=0
ENSEMBLE_BMA_MAX_BRIER_REGRESSION=0.005
ENSEMBLE_BMA_STATS_LOW_DATA_BOOST=1.5
ENSEMBLE_BMA_ML_HIGH_QUALITY_BOOST=1.5
```

Docker image'ı LightGBM çalışma zamanı için `libgomp1` içerir. Worker ve backend
aynı dependency manifestinden oluşturulduğu için CatBoost/LightGBM ile yazılmış
artifact serving sürecinde de açılabilir.
