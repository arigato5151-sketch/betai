# Kalibrasyon doğrulaması

Son çalışma: **26 Temmuz 2026**

## Sonuç

`Settings` içindeki 37 kalibrasyon alanının 27'si tarihsel duyarlılık
analizinden geçirildi. On alan, yürütülen tahmin yolunda etkisiz olması veya
yeterli tarihsel örneğin bulunmaması nedeniyle açık TODO olarak bırakıldı.

Bu çalışma production varsayılanlarını değiştirmedi. Bazı grid noktaları Brier
Score'u düşürse de:

- tarama tek-değişkenli ve aynı tarihsel küme üzerinde seçildi;
- en iyi değerlerin bir bölümü ±%20 sınırına dayandı;
- 1X2 rotalarının hiçbirinde kapanış oranı proxy'sine karşı pozitif ROI
  görülmedi.

Bu nedenle aralıklar bir sonraki out-of-time doğrulamanın arama alanıdır;
kârlılık veya production üstünlüğü iddiası değildir.

## Veri ve yöntem

- Sonuç verisi: 8.333 tamamlanmış maç, 14 lig, 2 sezon.
- Kaynak dağılımı: 3.569 API-Football (2024), 4.764 Football-Data CSV (2025).
- Tarih aralığı: 2 Ağustos 2024 – 24 Mayıs 2026.
- Nokta-zaman 1X2 örneği: 7.622. Her maç için yalnız daha eski maçlar
  kullanıldı; aynı başlama saatindeki maçlar aynı pre-kickoff state'i paylaştı.
- Oran verisi: 2025 Football-Data dosyalarından 4.145 kapanış oranı eşleşmesi.
  Öncelik `AvgCH/AvgCD/AvgCA`; yoksa kapanış bookmaker ve standart ortalama
  kolonları.
- Grid: mevcut değerin `0.8, 0.9, 1.0, 1.1, 1.2` katları. Form tuple'ında ilk
  ağırlık 1.0 tutulup kalan elemanlar ölçeklendi; lig rho sözlüğünün tüm
  değerleri birlikte ölçeklendi.
- Brier: 1X2 için ortalama multiclass Brier; çifte şans için binary Brier.
  Düşük değer daha iyidir.
- ROI: mevcut `BacktestEngine`, 10.000 başlangıç bakiyesi, 10 birim sabit bahis
  ve minimum %3 edge. Kapanış oranı kullanıldığı için yalnız araştırma
  proxy'sidir. Çifte şans oranı bulunmadığından o iki satırda ROI yoktur.
- Rich rota gerçek sezon profillerini; legacy rota aynı tarihsel profillerin
  eski payload biçimini; Elo rota prequential rating akışını kullanır.

Tekrar üretme komutu:

```bash
python scripts/calibrate_constants.py \
  --with-football-data-odds \
  --odds-season 2025 \
  --summary-only
```

## Doğrulanan alanlar

`Brier` ve `ROI` sütunları `mevcut → grid en iyisi` biçimindedir. Önerilen
aralık, en iyi Brier değerine en fazla 0,001 uzaklıktaki grid noktalarını
gösterir. Tek değerli ve grid sınırındaki aralıklar daha geniş bir takip
taraması gerektirir.

| Alan | Rota / örnek | Test edilen aralık | Önerilen aralık | Brier | ROI % | Karar |
|---|---:|---:|---:|---:|---:|---|
| `LEAGUE_BASELINE_GOALS` | rich / 7.622 | 1,056–1,584 | 1,584 (üst sınır) | 0,676043 → 0,667263 | -43,38 → -38,47 | Varsayılanı koru; geniş out-of-time tarama yap |
| `FORM_DECAY_WEIGHTS` | rich / 7.622 | son 4 ağırlık ×0,8–1,2 | son 4 ağırlık ×0,8–1,2 | 0,676043 → 0,675995 | -43,38 → -43,33 | Etki ihmal edilebilir; mevcut tuple uygun |
| `HOME_ATTACK_BOOST` | legacy / 7.622 | 0,888–1,332 | 0,999–1,332 | 0,675652 → 0,675652 | -39,05 → -39,05 | Mevcut 1,11 grid optimumu; koru |
| `AWAY_ATTACK_PENALTY` | rich / 7.622 | 0,744–1,116 | 0,744 (alt sınır) | 0,676043 → 0,666762 | -43,38 → -40,12 | Daha düşük değer takip taramasına alınmalı |
| `XG_OBSERVED_GOALS_WEIGHT` | legacy / 7.622 | 0,44–0,66 | 0,44 (alt sınır) | 0,675652 → 0,672257 | -39,05 → -41,12 | Brier iyileşiyor, ROI kötüleşiyor; değiştirme |
| `XG_ATTACK_BASELINE_WEIGHT` | legacy / 7.622 | 0,36–0,54 | 0,36 (alt sınır) | 0,675652 → 0,673208 | -39,05 → -40,11 | Brier/ROI ayrışıyor; değiştirme |
| `XG_CONSISTENCY_MAX_PENALTY` | legacy / 7.622 | 0,096–0,144 | 0,096–0,144 | 0,675652 → 0,675648 | -39,05 → -39,62 | Duyarsız; mevcut 0,12 uygun |
| `XG_CONSISTENCY_PENALTY_WEIGHT` | legacy / 7.622 | 0,032–0,048 | 0,032–0,048 | 0,675652 → 0,675466 | -39,05 → -39,53 | Duyarsız; mevcut 0,04 uygun |
| `PROFILE_FORM_FACTOR_BASE` | rich / 7.622 | 0,704–1,056 | 0,704 (alt sınır) | 0,676043 → 0,672346 | -43,38 → -37,38 | Takip taraması gerekli; varsayılanı koru |
| `PROFILE_FORM_FACTOR_WEIGHT` | rich / 7.622 | 0,192–0,288 | 0,192–0,216 | 0,676043 → 0,674293 | -43,38 → -41,69 | Out-of-time doğrulama öncesi değiştirme |
| `LEGACY_ATTACK_FACTOR_BASE` | legacy / 7.622 | 0,496–0,744 | 0,682–0,744 | 0,675652 → 0,674611 | -39,05 → -41,39 | Brier/ROI ayrışıyor; değiştirme |
| `LEGACY_ATTACK_FACTOR_WEIGHT` | legacy / 7.622 | 0,624–0,936 | 0,624 (alt sınır) | 0,675652 → 0,668656 | -39,05 → -42,44 | Brier/ROI ayrışıyor; değiştirme |
| `LEGACY_DEFENSE_FACTOR_BASE` | legacy / 7.622 | 0,576–0,864 | 0,576 (alt sınır) | 0,675652 → 0,672627 | -39,05 → -40,64 | Brier/ROI ayrışıyor; değiştirme |
| `LEGACY_DEFENSE_FACTOR_WEIGHT` | legacy / 7.622 | 0,44–0,66 | 0,44–0,495 | 0,675652 → 0,674003 | -39,05 → -37,40 | Takip out-of-time testine aday |
| `LEGACY_FORM_FACTOR_BASE` | legacy / 7.622 | 0,656–0,984 | 0,656–0,738 | 0,675652 → 0,673540 | -39,05 → -36,21 | Takip out-of-time testine aday |
| `LEGACY_FORM_FACTOR_WEIGHT` | legacy / 7.622 | 0,288–0,432 | 0,288 (alt sınır) | 0,675652 → 0,673354 | -39,05 → -40,69 | Brier/ROI ayrışıyor; değiştirme |
| `LEGACY_XG_OBSERVED_WEIGHT` | legacy / 7.622 | 0,464–0,696 | 0,464 (alt sınır) | 0,675652 → 0,669920 | -39,05 → -41,40 | Brier/ROI ayrışıyor; değiştirme |
| `LEGACY_XG_BASELINE_WEIGHT` | legacy / 7.622 | 0,336–0,504 | 0,378–0,504 | 0,675652 → 0,675332 | -39,05 → -40,63 | Duyarlılık düşük; mevcut 0,42 uygun |
| `HOME_ADVANTAGE_MIN_MULTIPLIER` | rich / 7.622 | 0,704–1,056 | 1,056 (üst sınır) | 0,676043 → 0,672316 | -43,38 → -45,63 | Brier/ROI ayrışıyor; değiştirme |
| `HOME_ADVANTAGE_MAX_MULTIPLIER` | rich / 7.622 | 0,976–1,464 | 1,22–1,464 | 0,676043 → 0,675787 | -43,38 → -43,33 | Duyarlılık düşük; mevcut 1,22 uygun |
| `HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR` | rich / 7.622 | 0,44–0,66 | 0,44–0,66 | 0,676043 → 0,676015 | -43,38 → -43,38 | Duyarsız; mevcut 0,55 uygun |
| `DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT` | 1X / 883 | 9,6–14,4 | 9,6–13,2 | 0,221426 → 0,221230 | yok | Duyarlılık düşük; mevcut 12 uygun |
| `DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT` | X2 / 2.695 | 11,2–16,8 | 11,2–14,0 | 0,220331 → 0,219899 | yok | Duyarlılık düşük; mevcut 14 sınırda kabul |
| `DEFAULT_DIXON_COLES_RHO` | global / 7.622 | -0,144–-0,096 | -0,144–-0,096 | 0,675937 → 0,675549 | -42,53 → -40,29 | Tüm grid kabul; mevcut -0,12 uygun |
| `LEAGUE_DIXON_COLES_RHO` | rich / 7.622 | sözlük ×0,8–1,2 | sözlük ×0,8–1,2 | 0,676043 → 0,675695 | -43,38 → -42,40 | Tüm grid kabul; mevcut sözlük uygun |
| `ELO_K_FACTOR` | Elo / 8.333 | 25,6–38,4 | 28,8–38,4 | 0,624520 → 0,624423 | -56,00 → -54,34 | Duyarlılık düşük; mevcut 32 uygun |
| `ELO_HOME_ADVANTAGE_POINTS` | Elo / 8.333 | 52–78 | 52–58,5 | 0,624520 → 0,623402 | -56,00 → -52,31 | 52–58,5 takip doğrulamasına aday |

## Doğrulanamayan alanlar

Bu alanların TODO yorumları, nedenleriyle birlikte kodda korunmuştur.

| Alan | Mevcut | Neden |
|---|---:|---|
| `FORM_DECAY_FALLBACK_WEIGHT` | 0,4 | Form kodu girdiyi beş maçta kesiyor; fallback index'i erişilemiyor |
| `STRENGTH_ATTACK_WEIGHT` | 0,4 | Yalnız `strength_rating` değişiyor; bu alan tahmin rotalarında tüketilmiyor |
| `STRENGTH_DEFENSE_WEIGHT` | 0,35 | Yalnız `strength_rating` değişiyor; bu alan tahmin rotalarında tüketilmiyor |
| `STRENGTH_FORM_WEIGHT` | 0,25 | Yalnız `strength_rating` değişiyor; bu alan tahmin rotalarında tüketilmiyor |
| `HOME_FORM_BASE_MULTIPLIER` | 1,08 | Tarihsel profillerin tamamında gol ortalaması mevcut; fallback rota oluşmuyor |
| `HOME_FORM_BOOST_DIVISOR` | 450 | Tarihsel profillerin tamamında gol ortalaması mevcut; fallback rota oluşmuyor |
| `ENSEMBLE_STATS_WEIGHT` | 0,4 | Üç bileşeni de içeren çözülmüş tahmin 0; minimum gereksinim 100 |
| `ENSEMBLE_ML_WEIGHT` | 0,2 | Üç bileşeni de içeren çözülmüş tahmin 0; minimum gereksinim 100 |
| `ENSEMBLE_MARKET_WEIGHT` | 0,4 | Üç bileşeni de içeren çözülmüş tahmin 0; minimum gereksinim 100 |
| `ELO_SEASON_REGRESSION` | 0,25 | Sezonlar farklı provider takım ID'leri kullanıyor; rating sezon sınırını geçmiyor |

## Takip kararları

1. 2026 sezonu tamamlandığında aynı taramayı yalnız ileri tarihli holdout üzerinde
   tekrar çalıştır.
2. API-Football ve Football-Data takım kimliklerini tek canonical ID altında
   birleştirip `ELO_SEASON_REGRESSION` alanını yeniden test et.
3. En az 100 çözülmüş ve üç bileşenli tahmin biriktiğinde ensemble ağırlıklarını
   `EnsembleWeightManager` holdout optimizasyonuyla doğrula.
4. `strength_rating` ve erişilemeyen form fallback alanları ya prediction
   sözleşmesine bağlanmalı ya da konfigürasyondan kaldırılmalı.
5. Pozitif out-of-time ROI görülmeden grid optimumlarını production değerlerine
   taşımama kuralını koru.
