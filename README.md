# Bet AI Platform

Futbol karşılaşmaları için istatistiksel tahmin, value bet değerlendirmesi, tahmin geçmişi, model denetimi ve strateji backtest'i sunan full-stack analiz uygulaması.

> **Uyarı:** Üretilen çıktılar istatistiksel tahmindir; kesin sonuç veya yatırım tavsiyesi değildir.

## Kod Deposu

Projenin kaynak koduna GitHub üzerinden ulaşabilirsiniz: [arigato5151-sketch/betai](https://github.com/arigato5151-sketch/betai)

## Özellikler

- Zaman ağırlıklı gol ortalamaları, Poisson skor matrisi ve Dixon-Coles düzeltmesiyle 1X2 olasılıkları
- Beklenen gol (xG), form, hücum ve savunma verilerinden manuel analiz
- API-Football üzerinden fikstür, takım istatistiği, H2H ve oran verisi
- UEFA Şampiyonlar Ligi, UEFA Avrupa Ligi ve UEFA Konferans Ligi desteği
- Piyasa olasılığını marjdan arındıran (de-vig) value bet ve Kelly stake hesabı
- Yeterli etiketli veri oluştuğunda devreye giren çok sınıflı ML pipeline'ı
- HMAC-SHA256 ile imzalanan ve yükleme/rollback öncesinde bütünlüğü doğrulanan ML artifact'ları
- Açılış ve güncel 1X2 oranlarından üretilen dropping/drifting odds ML feature'ları
- Oyuncu rating'i ve ilk 11 değişiminden üretilen Team Strength Ratio ile dinamik xG etkisi
- Son 14 günlük maç yükü, dinlenme süresi ve deplasman mesafesini birleştiren yorgunluk feature'ı
- Tahmin geçmişi, gerçek sonuç kaydı, ROI/Brier Score denetimi ve backtest
- Redis erişilemediğinde Memcached, o da yoksa bellek içi TTL cache; PostgreSQL erişilemediğinde SQLite fallback
- Redis kesintisinde process-local çalışan ve bağlantı geri geldiğinde otomatik olarak Redis'e dönen login rate limiter
- React tabanlı analiz paneli, geçmiş filtreleme/sıralama, olasılık ve interaktif bankroll grafikleri
- Backend kaynaklı lig seçimi; UEFA turnuvalarının Türkçe adlarla analize aktarılması
- Türkiye saatine göre kronolojik sıralanan, yenilenebilir 7 günlük maç fikstürü; kart seçimiyle analiz formuna otomatik veri aktarımı
- Tüm `ml_features_v8` girdilerini eksik/kısmi/hesaplandı durumlarıyla gösteren ve güvenli manuel override destekleyen gelişmiş analiz formu
- API-Football demo verisi kullanıldığında giriş ve uygulama başlığında görünür `Demo Modu` etiketi
- Admin rolü için kullanıcı oluşturma, rol atama ve hesap aktifliği yönetim paneli

## Teknoloji Yığını

| Katman | Teknolojiler |
| --- | --- |
| Backend | Python 3.11/3.12, FastAPI, Uvicorn, Pydantic, PyJWT, bcrypt |
| Veri | SQLAlchemy, PostgreSQL, SQLite fallback |
| Cache / görev | Redis, Memcached, Celery, `cachetools` |
| Analiz / ML | NumPy, Pandas, SciPy, scikit-learn, SHAP, Joblib |
| Frontend | React 18, Vite 8, Tailwind CSS, Chart.js |

## Mimari ve Analiz Akışı

```text
React/Vite UI
     |
     v
FastAPI endpoint'leri
     |
     +--> Manuel veri veya API-Football verisi
     +--> StatsEngine (Poisson + Dixon-Coles + oyuncu etkisi + ikincil marketler)
     +--> ValueCalc (de-vig + edge + Kelly)
     +--> ML pipeline / açıklanabilirlik (model hazırsa)
     |
     v
SQLAlchemy repository --> PostgreSQL veya SQLite
     |
     +--> Gerçek sonuç --> audit / backtest / yeniden eğitim görevi
```

ML modeli varsayılan olarak en az `200` etiketli örnek bekler. Bu eşik sağlanana kadar istatistik motoru çalışmaya devam eder ve API yanıtı ML durumunu `INSUFFICIENT_DATA` olarak bildirir.

## Uygulamanın Genel Kod Akışı

Uygulamanın uçtan uca çalışma zinciri aşağıdaki gibidir:

```text
frontend/src/App.jsx (auth, layout ve aktif görünüm)
    ├── frontend/src/containers/UpcomingFixturesContainer.jsx
    │       ├── GET /api/fixtures/upcoming?days=7&limit=100
    │       └── kart seçimi → GET /api/fixtures/{fixture_id}/prefill
    └── frontend/src/containers/AnalysisContainer.jsx
            └── POST /api/analyze (cookie auth + CSRF)
                    └── backend/app/api/endpoints.py
                            ├── StatsEngine.analyze_match()
                            ├── ValueCalc.calculate_professional()
                            ├── FeatureEngine + MLModelPipeline (model hazırsa)
                            ├── ExplainabilityService.generate_explanation()
                            └── MatchPredictionRepository.upsert_prediction()
                                    └── PostgreSQL veya SQLite fallback
```

Backend'deki ana orchestration fonksiyonu analiz, kalıcılık ve API yanıtı üretimini birbirinden ayırır:

```python
async def _run_analysis(payload: AnalysisRequest) -> dict:
    # İstatistik, value bet ve opsiyonel ML çıkarımı veritabanından bağımsız çalışır.
    computed = await _compute_analysis(payload)

    # Aynı fixture_id daha önce kaydedilmişse güncellenir; aksi halde yeni kayıt oluşur.
    db_record, labeled_samples_count = _persist_analysis(payload, computed)

    response = _build_analysis_response(
        db_record.id,
        payload.home_team,
        payload.away_team,
        computed["analysis"],
        computed["value_data"],
        computed["ml_result"],
        computed["insights"],
        labeled_samples_count,
    )
    if computed["value_data"].get("data_methodology"):
        response["data_methodology"] = computed["value_data"]["data_methodology"]
    return response
```

Katmanların sorumlulukları:

| Katman | Ana dosyalar | Sorumluluk |
| --- | --- | --- |
| Web arayüzü | `frontend/src/App.jsx`, `frontend/src/containers/`, `frontend/src/components/`, `frontend/src/hooks/` | Üst seviye auth/layout orkestrasyonu, özellik container'ları, sunumsal bileşenler ve oturum hook'u |
| HTTP/API | `backend/app/main.py`, `backend/app/api/endpoints.py`, `backend/app/api/admin.py` | Route, doğrulama, permission kontrolü ve yanıt sözleşmeleri |
| Analiz | `backend/app/prediction/stats_engine.py`, `value_calc.py` | Poisson/Dixon-Coles olasılıkları, de-vig, edge ve Kelly hesabı |
| Makine öğrenmesi | `backend/app/prediction/ml/` | Feature üretimi, eğitim, inference, kalibrasyon ve açıklanabilirlik |
| Veri erişimi | `backend/app/db/` | Tahmin/tarihsel fixture modelleri, transaction ve idempotent repository işlemleri |
| Entegrasyon | `backend/app/services/` | API-Football, Redis, Memcached ve yerel cache fallback'leri |
| Arka plan görevleri | `backend/app/tasks/` | Sonuç/tarihsel fixture senkronizasyonu, model eğitimi ve Celery sağlık kontrolü |
| Güvenlik | `backend/app/core/auth.py`, `security.py`, `rate_limit.py` | Cookie tabanlı JWT, CSRF, origin kontrolü ve brute-force koruması |

Tam implementasyon kod deposundaki `backend/app/`, `frontend/src/`, `tests/` ve `backend/migrations/` dizinlerinde tutulur. README'deki örnek, üretim kodunun çalışma sırasını özetler; API sözleşmesinin kaynağı `docs/openapi.json` dosyasıdır.

## Proje Yapısı

```text
bet-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI endpoint'leri ve istek modelleri
│   │   ├── core/         # Ayarlar, lig listesi, loglama, demo veri
│   │   ├── db/           # SQLAlchemy modelleri, session ve repository
│   │   ├── prediction/   # İstatistik, value, ML, audit ve backtest
│   │   ├── services/     # API-Football ve cache entegrasyonu
│   │   └── tasks/        # Celery görevleri
│   ├── artifacts/        # Eğitilmiş model çıktıları
│   ├── migrations/       # Alembic revision dosyaları
│   ├── alembic.ini       # Migration yapılandırması
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── containers/   # Analiz, geçmiş, admin ve operasyon akışları
│   │   └── components/   # Sunumsal React bileşenleri
│   └── package.json
├── docs/
│   └── CALIBRATION.md    # Backtest tabanlı kalibrasyon sonuçları
├── scripts/              # Kurulum, bakım ve kalibrasyon scriptleri
├── .github/              # CI, bağımlılık denetimi ve Dependabot ayarları
├── app.py                # Backend uygulama giriş noktası
├── run.py                # Ortam kontrolü ve Uvicorn launcher
└── docker-compose.yml
```

## Yapay Zekâ Analizi İçin Kod Bağlamı

Bu bölüm, depoyu başka bir yapay zekâ aracına verdiğinizde uygulama akışını hızlıca anlamasını sağlamak için hazırlanmıştır. Backend kaynak kodu `backend/app/`, frontend uygulaması ise `frontend/src/` dizinindedir.

### Aktif giriş noktaları

| Dosya | Görevi |
| --- | --- |
| `app.py` | Yerel çalıştırmada `backend.app.main:app` nesnesini dışarı açar |
| `run.py` | Ortam/bağımlılık kontrolünü yapar ve Uvicorn'u başlatır |
| `backend/app/main.py` | FastAPI uygulaması, CORS, startup, router ve health endpoint'i |
| `backend/app/api/endpoints.py` | HTTP sözleşmeleri ve analiz orchestration katmanı |
| `frontend/index.html` | Vite HTML giriş noktası |
| `frontend/src/main.jsx` | React root oluşturur |
| `frontend/src/App.jsx` | Auth durumu, layout ve aktif görünümü yöneten ince orkestrasyon bileşeni |
| `frontend/src/containers/` | Analiz, geçmiş, admin ve operasyon özelliklerinin state/API akışları |

### Backend modül haritası

```text
backend/app/
├── core/
│   ├── config.py            # Pydantic Settings ve tüm env değişkenleri
│   ├── auth.py              # PyJWT access/refresh üretimi ve auth dependency
│   ├── passwords.py         # bcrypt hash ve legacy hash doğrulaması
│   ├── rate_limit.py        # Redis login guard ve process-local recovery
│   ├── allowed_leagues.py   # Desteklenen ligler ve Dixon-Coles rho değerleri
│   ├── demo_data.py         # API anahtarı yokken kullanılan deterministik veri
│   └── logging_config.py    # Merkezi log yapılandırması
├── api/endpoints.py         # Request modelleri, endpoint'ler, analiz akışı
├── db/
│   ├── models.py            # MatchPrediction ve HistoricalFixture modelleri
│   ├── historical_repository.py # Tarihsel fixture upsert/zaman kesitli sorgular
│   ├── session.py           # PostgreSQL bağlantısı ve SQLite fallback
│   └── repository.py        # Tahmin CRUD/upsert işlemleri
├── prediction/
│   ├── stats_engine.py      # Poisson, Dixon-Coles, 1X2 ve yan marketler
│   ├── value_calc.py        # De-vig, edge, value bet ve Kelly hesabı
│   ├── audit.py             # ROI, CLV ve tahmin kalite metrikleri
│   ├── backtest.py          # Bankroll strateji simülasyonu
│   └── ml/
│       ├── features.py      # ML feature üretimi
│       ├── historical.py    # Kronolojik Elo ve H2H feature bağlamı
│       ├── model.py         # Eğitim, imzalı artifact, rollback ve inference
│       ├── calibrate.py     # Çok sınıflı olasılık kalibrasyonu
│       └── explain.py       # SHAP veya feature importance açıklaması
├── services/
│   ├── api_football.py      # API-Football istemcisi, retry ve normalizasyon
│   ├── external_features.py # Kaynak önceliği, eşleme ve snapshot yönetimi
│   └── cache.py             # Redis + Memcached + bellek içi TTL cache
├── providers/
│   ├── base.py              # Sağlayıcıdan bağımsız değer/provenance sözleşmesi
│   └── clubelo.py           # Doğrulamalı ClubElo fallback adaptörü
└── tasks/
    ├── celery_app.py        # Celery broker/backend ve periyodik görev ayarları
    ├── jobs.py              # Sonuç/tarihsel fixture senkronizasyonu ve model eğitimi
    └── health.py            # Broker/worker kontrolü ve güvenli enqueue
```

### Tarihsel fixture veri hattı

Celery Beat, izin verilen liglerin mevcut sezon tamamlanmış maçlarını Football-Data
CSV arşivinden günlük olarak `historical_fixtures` tablosuna idempotent biçimde
yazar. Football-Data'nın 14 yerel lig eşlemesi; şema doğrulama, yerel saatten
UTC'ye dönüşüm ve kaynak provenance'ı ile içe alınır. Sonuçlara ek olarak kaynakta
bulunan devre skoru, şut, isabetli şut, faul, korner, sarı/kırmızı kart ve Bet365
açılış/kapanış 1X2 oranları da nullable alanlar olarak saklanır. Eksik değerler
uydurulmaz; `NULL` bırakılır. Açılış-kapanış oranları tarihsel ML örneklerindeki
`odds_movement_*` feature'larını doğrudan besler. Analiz sırasında yalnızca maçın `kickoff` zamanından
önceki kayıtlar okunur; aynı anda başlayan maçlar tek batch olarak işlendiğinden Elo,
form ve H2H feature'larında gelecek bilgisi sızıntısı oluşmaz.

1 Ağustos 2026 tarihinde yapılan 2025/26 yeniden senkronizasyonunda 4.764 yerel lig
maçının 4.524'ünde maç istatistikleri, 4.520'sinde açılış ve 4.524'ünde kapanış
oranları doğrulandı. Rusya rolling feed'indeki 240 maç yalnızca sonuç içerdiğinden
zenginleştirme alanları boş kalır.

Beş büyük ligde maç bazlı beklenen gol verisi, ayrı ve kapatılabilir Understat
sağlayıcısından günlük olarak alınır (`UNDERSTAT_ENABLED`). JSON yanıtı boyut,
tip ve değer aralığı açısından doğrulanır; lig istekleri sağlayıcı yükünü sınırlamak
için varsayılan 1,5 saniye aralıkla gönderilir. Kayıtlar yalnızca normalize takım
adları, final skor ve en fazla 48 saatlik başlama zamanı farkı birlikte uyuştuğunda
eşleştirilir. Birden fazla eşit aday varsa kayıt reddedilir. 2025/26 pilotunda
beş ligdeki 1.752 maçın 1.751'i için xG kaynağı bulundu; Understat'ta bulunmayan
tek maçın alanları `NULL` kaldı. Bu gözlemler geçmiş ML eğitiminde gelecek veri
sızıntısı olmadan, önceki maçların zaman ağırlıklı xG ortalaması olarak kullanılır.

Yeni sezon yayını her lig için ayrı denetlenir. Örneğin Rusya rolling feed'i
yayımlanmışken Premier League dosyası henüz yoksa Rusya'nın güncel sezonu alınır,
yalnız yayımlanmayan lig bir önceki sezona düşer. API-Football ve CSV takım adları
ülke/lige bağlı geçmiş içinde muhafazakâr alias kurallarıyla eşleştirilir
(`FC/FK`, `Moskva/Moscow` gibi); eşsiz eşleşme bulunamazsa yanlış takıma veri
bağlamak yerine alan eksik bırakılır.

Uygulama toplam 17 organizasyonu destekler. Yerel liglere ek olarak API-Football
kimlikleri `2`, `3` ve `848` olan UEFA Şampiyonlar Ligi, UEFA Avrupa Ligi ve UEFA
Konferans Ligi yaklaşan fikstür, manuel analiz ve tarihsel backfill kapsamındadır.
UEFA turnuvaları Football-Data CSV kaynağında bulunmadığından tamamlanmış maç
sonuçları günlük olarak Fixture Download JSON akışından alınır. Bu açık veri
kaynağı kullanılamazsa plan erişimi bulunan sezonlarda API-Football backfill'i
çalıştırılabilir.

2025/26 sezonunu tekrar içe aktarmak için:

```powershell
cd backend
python -c "from app.tasks.jobs import sync_football_data_fixtures_task; print(sync_football_data_fixtures_task.run([2025]))"
```

API-Football'ın plan kapsamında erişilebilen eski sezonlarını backfill etmek için:

```powershell
cd backend
python -c "from app.tasks.jobs import sync_historical_fixtures_task; print(sync_historical_fixtures_task.run([2023, 2024]))"
```

Yalnız UEFA turnuvalarını kota kontrollü biçimde backfill etmek için:

```powershell
cd backend
python -c "from app.tasks.jobs import sync_historical_fixtures_task; print(sync_historical_fixtures_task.run([2023, 2024], [2, 3, 848]))"
```

UEFA 2025/26 sonuçlarını anahtarsız açık veri kaynağından içe aktarmak için:

```powershell
cd backend
python -c "from app.tasks.jobs import sync_uefa_fixtures_task; print(sync_uefa_fixtures_task.run([2025]))"
```

Football-Data içe aktarımı API anahtarı gerektirmez. API-Football backfill'i için
geçerli anahtar ve sezon erişimi gerekir. Harici fixture ve takım kimlikleri negatif
64-bit deterministik değerlerdir; API kimlikleriyle çakışmaz. Her kaydın
`data_source` alanı kaynağı taşır. Aynı maç yeniden çekildiğinde skor ve durum
güncellenir, yeni bir satır oluşturulmaz.

Kupa maçlarında ML hedefi 1X2 pazarıyla aynı sözleşmeyi kullanır: uzatma ve penaltı
skorları sonuç etiketine dahil edilmez, API-Football `score.fulltime` alanındaki
90 dakika skoru saklanır. Üç UEFA turnuvası için lig-özel Dixon-Coles kalibrasyonu
henüz bulunmadığından doğrulanmış global `rho=-0.12` güvenli varsayılanı kullanılır.

Son maç formu, dinlenme günü, gol ortalaması, clean-sheet ve gol serisi feature'ları
da aynı zaman kesitli tablodan üretilir. Yerel geçmişte varsayılan olarak en az 5 maç
yoksa veya son kayıt 45 günden eskiyse API-Football verisine fallback yapılır. Bu
eşikler `RECENT_FORM_MATCH_COUNT` ve `HISTORICAL_FORM_MAX_AGE_DAYS` ortam
değişkenleriyle ayarlanabilir.

Poisson gol profili, geçmiş maçları eşit ortalamak yerine
`exp(-GOAL_TIME_DECAY_FACTOR * days_ago)` ağırlığını kullanır. Varsayılan
`GOAL_TIME_DECAY_FACTOR=0.01` yaklaşık 69 günlük yarı ömürdür; `0` değeri eşit
ağırlık davranışını geri getirir. Gelecekteki veya geçersiz tarihli kayıtlar
hesaba katılmaz ve geçmiş bulunamazsa mevcut sezon profiline güvenli fallback yapılır.

Elo hesabı lig geçmişini sezonlar arasında kronolojik taşır. Yeni sezon başında
rating farkının varsayılan `%25` bölümü lig ortalaması olan `1500` değerine geri
çekilir ve beklenen ev sahibi skoruna varsayılan `65` Elo puanı eklenir. Güncelleme
hızı, ev avantajı ve sezon regresyonu sırasıyla `ELO_K_FACTOR`,
`ELO_HOME_ADVANTAGE_POINTS` ve `ELO_SEASON_REGRESSION` ile kalibre edilebilir.

Yerel geçmişte takım için Elo üretilemediğinde isteğe bağlı ClubElo fallback'i
devreye girer. Entegrasyon yalnızca aksan/noktalama duyarsız **birebir** takım adı
eşleşmesini kabul eder; belirsiz veya bulunamayan isimlerde tahmine değer enjekte
etmez. Eşlemeler `provider_team_mappings`, gözlemler
`external_feature_snapshots` tablosunda kaynak, yakalanma zamanı, güven skoru,
geçerlilik süresi ve fallback bayrağıyla saklanır. Analiz formu aynı bilgileri her
feature'ın altında gösterir. ClubElo'nun yayımlanan CSV uç noktası HTTP kullandığı
için veri düşük öncelikli (`CLUBELO_CONFIDENCE=0.80`) kabul edilir, şema/boyut/değer
aralığı sıkı doğrulanır ve yerel tarihsel veri bulunduğunda hiçbir zaman onu ezmez.

Fikstür ön-doldurma sırasında alınan API-Football 1X2 piyasası ayrıca
`fixture_odds_snapshots` tablosuna zaman damgalı olarak yazılır. İlk gözlem açılış
snapshot'ı olarak saklanır; `ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS` süresi geçmeden
ikinci gözlem varmış gibi hareket feature'ı üretilmez. En az iki geçerli gözlem
oluştuğunda `odds_movement_home`, `odds_movement_draw` ve
`odds_movement_away` otomatik hesaplanır. Kaynak güveni
`ODDS_SNAPSHOT_CONFIDENCE` ile yönetilir. Formda oranlar manuel değiştirilirse eski
otomatik snapshot çifti temizlenir ve yanlış provenance kullanılmaz.

Celery Beat aynı akışı kullanıcı fikstürü açmadan da besler. Varsayılan üç saatte
bir çalışan `collect_upcoming_odds_task`, yedi günlük penceredeki en fazla 20 maçı
inceler. Her maç için ilk gözlemi alır; sonraki API çağrılarını kickoff'a son 24 saat
kalana kadar erteler ve bu pencere içinde de üç saatten sık sorgulamaz. Böylece
açılış/güncel oran çifti otomatik oluşurken API kotası sınırlı tutulur. Davranış
`ODDS_COLLECTOR_*` ortam değişkenleriyle ayarlanabilir veya
`ODDS_COLLECTOR_ENABLED=false` ile kapatılabilir. Demo verisi hiçbir zaman kalıcı
oran snapshot'ı üretmez.

Celery Beat ayrıca `collect_upcoming_lineups_task` ile yaklaşan maçların resmi ilk
11'lerini maçtan önceki varsayılan 120 dakikalık pencerede saatte bir kontrol eder.
İki takımın da 11 oyunculuk kadrosu doğrulanınca sonuç altı saat cache'lenir; eksik
yanıtlar 15 dakika sonra yeniden denenebilir. Tarama yalnızca iki günlük penceredeki
en fazla 30 gerçek maçı kapsar, demo fikstürlerini atlar ve
`LINEUP_COLLECTOR_*` ortam değişkenleriyle ayarlanabilir.

Model eğitimi, doğrulanmış tahmin snapshot'larına ek olarak `historical_fixtures`
kayıtlarından üretilen nokta-zamanlı örnekleri kullanır. Bir tarihsel maçın feature
vektörü oluşturulurken yalnızca o maçın kickoff zamanından önce oynanmış karşılaşmalar
görülür. Her iki takım için varsayılan en az üç geçmiş maç şartı
`HISTORICAL_TRAINING_MIN_TEAM_MATCHES` ile değiştirilebilir.

Adaylar walk-forward pencerelerinde Brier skoru ve log-loss ile karşılaştırılır.
Regularized Logistic Regression, Gradient Boosting/XGBoost, Random Forest,
CatBoost ve LightGBM arasından seçilen aday; naive sınıf dağılımı baseline'ını
ve varsa aktif champion modeli geçmeden
yayına alınmaz. Isotonic kalibrasyon ayrı fit/doğrulama pencerelerinde değerlendirilir
ve en az `MIN_ISOTONIC_CALIBRATION_SAMPLES` örnek yoksa devre dışı kalır. Artifact
diske atomik yazılmadan süreç içi model değiştirilmez. Her model dosyası
`MODEL_SIGNING_KEY` ile HMAC-SHA256 olarak imzalanır ve detached `.sig` dosyasıyla
birlikte saklanır. Aktif model yükleme ve rollback işlemleri imzayı `joblib.load`
çağrısından önce doğrular; eksik veya uyuşmayan imza ERROR loglanır ve istatistik
motoru güvenli fallback olarak kullanılmaya devam eder.

Ensemble ağırlıkları kaynak setine ve `league_id` değerine göre prequential Bayesian
Model Averaging ile güncellenir. Yüksek kaliteli ve düşük sürprizli liglerde ML,
az verili liglerde Poisson/Dixon-Coles prior'ı güçlenir. Aday posterior son `%20`
kronolojik holdout kümesinde log-loss ve Brier guard'larını geçmeden aktive edilmez.
İmzalı schema-v2 artifact `backend/artifacts/models/ensemble_weights.json` altında
tutulur. Algoritma ve tüm yapılandırma seçenekleri
[docs/ML_ENSEMBLE.md](docs/ML_ENSEMBLE.md) içinde açıklanmıştır.

Fixture kimliği bulunan analizlerde API-Football sakatlık raporu da paralel çekilir ve
4 saat cache'lenir. `ml_features_v2`, ev/deplasman için `Missing Fixture` ile
`Questionable` sayılarını ve rapor bulunup bulunmadığını snapshot'a yazar. Güncel
normalizasyon bu sayaçları korurken bilinen oyuncunun kimliği, durumu ve nedeni de
oyuncu-etkisi hesabına taşır. Kimliği veya rating'i bulunmayan kayıtlar ham sayıdan
xG cezası türetmez.

`ml_features_v3`, maç saatine yakın yayınlanan doğrulanmış ilk 11'i takımın son
tamamlanmış maçındaki ilk 11 ile karşılaştırır. Her takım için lineup doğrulama,
tarihsel referans bulunma ve `ortak oyuncu / 11` süreklilik oranı saklanır. Eksik
veya 11 oyuncuya tamamlanmamış lineuplar süreklilik hesabına sokulmaz. Tamamlanmış
fixture lineupları `historical_fixtures` tablosundaki JSON kolonlarında tutulur;
bu değişiklik `20260720_0007` Alembic revision'ını gerektirir. API lineupları çoğu
desteklenen ligde maçtan 20–40 dakika önce geldiğinden erken analizlerde feature
nötr kalabilir ve maç saatine yakın yeniden analiz daha zengin snapshot üretir.

Birincil sağlayıcı en az 11 oyunculuk rating havuzu üretemezse isteğe bağlı
Sportmonks fallback'i takım adını aksan/noktalama duyarsız **birebir** eşleştirir
ve son 120 gündeki en fazla 10 maçın lineup rating'lerini dakika ağırlıklı toplar.
Oyuncu kimlikleri API-Football kimlikleriyle yanlışlıkla birleşmemesi için ayrı bir
sayısal namespace'te tutulur; eksik veya belirsiz eşleşme tahmine veri enjekte
etmez. Entegrasyon `SPORTMONKS_ENABLED=true` ve yalnızca backend ortamında tutulan
`SPORTMONKS_API_TOKEN` ile açılır. Takım eşlemeleri mevcut
`provider_team_mappings` tablosunda denetlenebilir şekilde saklanır.

`ml_features_v4`, desteklenen ligleri one-hot feature olarak ekler. Böylece model
ligler arasındaki ev sahibi/beraberlik/deplasman dağılımı farklarını öğrenebilir;
eski snapshot'lar bu alanlarda sıfır varsayılanıyla geriye uyumlu kalır.

`ml_features_v5`, açılış oranı ile analiz anındaki güncel/kapanışa yakın oran
arasındaki `((current / opening) - 1) * 100` değişimini ev sahibi, beraberlik ve
deplasman için ayrı feature olarak ekler. Negatif değer dropping odds, pozitif değer
drifting odds anlamına gelir. İki eksiksiz 1X2 snapshot bulunmadığında üç feature da
`0.0` olur. Sonuçtan sonra alınan closing odds bu pre-match feature'lara bağlanmaz;
böylece gelecek bilgisi sızıntısı engellenir.

`ml_features_v6`, `league_id`, `home_team_id` ve `away_team_id` alanlarını CatBoost
ve LightGBM için native kategorik feature olarak ekler. Bu kimlikler bulunmadığında
`0` unknown token'ı kullanılır; sayısal modeller ham ID kolonlarını görmez.

`ml_features_v7`, ev ve deplasman için `team_strength_ratio` ile tek bir
`fatigue_index` ekler. Son geçerli ilk 11'de yeterli rating kapsamı varsa güncel kadro
kalitesi referans kadroya oranlanır; referans ortalamasının üzerindeki eksikler
kritik kabul edilerek Poisson xG çarpanına daha güçlü yansır. Rating veya geçerli ilk
11 yoksa oran ve xG çarpanı `1.0` olur.

Yorgunluk skoru son 14 gündeki maç sayısını, son maçtan beri dinlenme gününü ve
deplasman seyahat mesafesini birleştirir. `fatigue_index = away_load - home_load`
olduğundan pozitif değer deplasman takımının daha yorgun olduğunu ifade eder. Eksik
veya geçersiz tarih/mesafe verisi ceza üretmez; güvenli varsayılan `0.0`'dır.
`ml_features_v8`, üç UEFA turnuvası için `league_2`, `league_3` ve `league_848`
one-hot alanlarını ekler. Eski imzalı `ml_features_v7` artifact'ları kendi
`feature_names` sözleşmeleriyle yüklenmeye ve rollback edilmeye devam eder; yeni
alanların öğrenilebilmesi için UEFA backfill'inden sonra model yeniden eğitilmelidir.

`ml_features_v1-v7` snapshot'ları nötr varsayılanlarla eğitimde kullanılmaya devam
eder. Formüller, tablolar ve tüm ayarlar
[`docs/PLAYER_IMPACT_FATIGUE.md`](docs/PLAYER_IMPACT_FATIGUE.md) belgesindedir.

### Kalibrasyon doğrulaması

`backend/app/core/config.py` içindeki 37 tahmin sabiti, mevcut
`BacktestEngine` kullanılarak varsayılan değerin `%80`, `%90`, `%100`, `%110` ve
`%120` noktalarında duyarlılık analizinden geçirilmiştir. Çalışma 14 ligde 8.333
tarihsel fixture ve 7.622 nokta-zamanlı örneği kapsar; 2025 closing odds bulunan
4.145 örnek ayrıca ROI değerlendirmesinde kullanılır. 27 sabit doğrulanmış, gerekli
tarihsel girdi bulunmayan 10 sabit gerekçesiyle açık TODO olarak bırakılmıştır.

Sonuçlar, veri kapsamı, Brier Score/ROI etkileri ve önerilen aralıklar
[`docs/CALIBRATION.md`](docs/CALIBRATION.md) dosyasında tutulur. Analizi mevcut
veritabanıyla tekrar üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_constants.py `
  --with-football-data-odds `
  --odds-season 2025 `
  --summary-only
```

### Temel analiz algoritması

Aktif analiz akışı kavramsal olarak aşağıdaki kodla özetlenebilir:

```python
async def analyze(payload):
    # Fixture kimlikleri varsa API-Football'dan form/H2H geçmişi alınır.
    match_data = await fetch_optional_match_data(payload)

    statistical_result = StatsEngine.analyze_match(
        home_stats=payload.home_stats,
        away_stats=payload.away_stats,
        league_id=payload.league_id,
    )

    market = payload.market_1x2 or ValueCalc.default_market(
        home_odd=payload.odd,
        model_probs=statistical_result["all_probabilities"],
    )
    value_result = ValueCalc.calculate_professional(
        analysis=statistical_result,
        market=market,
        fallback_odd=payload.odd,
    )

    # Model 200 etiketli örnek eşiğini geçmediyse istatistik sonucu kullanılır.
    ml_result = ml_pipeline.predict_match(build_features(match_data, payload))
    record = MatchPredictionRepository.upsert_prediction(
        combine(payload, statistical_result, value_result, ml_result)
    )
    return build_api_response(record, statistical_result, value_result, ml_result)
```

Gerçek implementasyon `backend/app/api/endpoints.py` içindeki `_compute_analysis`, `_persist_analysis` ve `_run_analysis` fonksiyonlarındadır.

### Matematiksel kurallar

- Takımların gol sayıları Poisson dağılımıyla modellenir: `P(X=k) = exp(-λ) * λ^k / k!`.
- `0-0`, `0-1`, `1-0` ve `1-1` hücreleri lig bazlı `rho` ile Dixon-Coles düzeltmesinden geçer.
- Skor matrisi normalize edilir; `HOME_WIN + DRAW + AWAY_WIN ≈ 100` olmalıdır.
- Oranların ham implied olasılığı `1 / odd` formülüyle hesaplanır.
- De-vig işlemi eksiksiz ve `>1.0` üç oranı zorunlu tutar; implied olasılıkları toplamlarına bölerek oransal normalize eder.
- Edge yüzdesi: `(model_probability * bookmaker_odd - 1) * 100`.
- Kelly: `((odd - 1) * p - (1 - p)) / (odd - 1)`; uygulama çeyrek Kelly ve oran bazlı üst limit uygular.
- Gerçek sonuç değerleri yalnızca `HOME_WIN`, `DRAW`, `AWAY_WIN` olabilir.

### Ana veri modeli

`MatchPrediction`, tek tahmin kaydında şu veri gruplarını tutar:

```text
Kimlik:       id, fixture_id, league_id, home_team_id, away_team_id
Takımlar:     home_team, away_team
Girdiler:     home/away xg, form, attack, defense
Tahmin:       prediction, probability, prob_home, prob_draw, prob_away
Value:        odd, edge, is_value_bet, kelly_stake
ML:           ml_cluster, ml_confidence
Gerçek sonuç: actual_result, actual_score_home, actual_score_away
Audit:        roi, closing_odds, clv, created_at
```

`fixture_id` benzersizdir. Aynı fixture tekrar analiz edildiğinde repository yeni satır eklemek yerine mevcut kaydı günceller. Manuel analizlerde `fixture_id` boş olabilir.

Oyuncu etkisi ve seyahat bağlamı iki normalize tabloda tutulur:

- `historical_player_performances`: tamamlanmış fixture'a bağlı ilk 11 durumu,
  dakika, rating, pozisyon, gol ve asist. `(fixture_id, player_id)` benzersizdir.
- `team_locations`: provider ve takım kimliğine göre takım/tesis konumu. Enlem ve
  boylam bilinmiyorsa nullable kalır ve seyahat etkisi nötrdür.

Doğrulanmış takım koordinatları admin tarafından `PUT /api/admin/team-locations`
ile toplu yüklenir ve `GET /api/admin/team-locations` ile denetlenir. Provider
doğrudan koordinat sunmadığında ve `AUTO_TEAM_LOCATION_ENABLED=true` olduğunda
API-Football takım/tesis şehri alınır; şehir merkezi koordinatı ağ çağrısı yapmayan
GeoNames veri setinden ülke kısıtlı birebir eşleşmeyle çözülür. Otomatik kayıtlar
`location_source=geonames_city`, güven skoru, tesis metadatası ve
`approximation=city_centre` bilgisiyle saklanır. Admin tarafından girilmiş veya
koordinatsız bırakılmış mevcut kayıtlar otomatik süreçte ezilmez. Veri hâlâ
çözülemiyorsa travel bileşeni `0.0` kalır; nötr saha için analiz isteğindeki
`away_travel_distance_km` kullanılabilir.

Bu tablolar `20260726_0010` Alembic revision'ıyla oluşturulur; takım konumu
provenance alanları `20260730_0013` revision'ıyla eklenir. Oyuncu performansı
yalnız daha sonraki fixture'larda kullanılır; sorgular tahmin kickoff'u için katı
`performance.kickoff < prediction.kickoff` sınırı uygular. API-Football
`fixtures/players` backfill'i her senkronizasyonda varsayılan olarak en fazla 20
eksik fixture'ı, 3 eşzamanlı istekle kademeli biçimde tamamlar.

Şehir verisi [GeoNames](https://www.geonames.org/) tarafından CC BY 4.0 lisansıyla
sağlanır. Şehir merkezi yaklaşık konumdur; kesin stadyum koordinatı olarak
yorumlanmamalıdır.

### Auth ve istemci davranışı

```text
POST /api/auth/login
    -> access ve refresh JWT'leri HttpOnly cookie olarak ayarlanır
    -> frontend token değerlerine JavaScript üzerinden erişmez
    -> korunan istekler credentials: include ile cookie gönderir
    -> 401 alınırsa /api/auth/refresh bir kez çağrılır
    -> refresh başarısızsa login ekranı açılır
    -> logout iki cookie'yi de siler
```

Access ve refresh token farklı secret anahtarlarla imzalanır. Token payload'ında kullanıcı UUID'si olan `sub` ile `type`, `ver`, `iat`, `exp`; refresh token'da ayrıca `jti` alanı bulunur. Cookie'ler `HttpOnly`, `Secure` ve varsayılan olarak `SameSite=lax` özelliklerini taşır. Refresh token yalnızca SHA-256 hash'iyle `refresh_sessions` tablosunda tutulur ve her yenilemede rotate edilerek eski session revoke edilir.

JWT üretimi/doğrulaması PyJWT ile yapılır ve izin verilen algoritma yalnızca
`JWT_ALGORITHM` ayarından sabit olarak okunur; `none` algoritması kabul edilmez.
Yeni parola hash'leri doğrudan bcrypt ile 12 cost kullanılarak `$2b$` formatında
üretilir. Önceki passlib sürümünün ürettiği bcrypt ve PBKDF2-SHA256 hash'leri
doğrulama sırasında geriye uyumlu olarak desteklenir.

### Cache, veritabanı ve worker fallback'leri

- Cache okuma/yazma zinciri `Redis → Memcached → process-local TTLCache` sırasındadır. `MEMCACHED_HOST` boşsa Memcached katmanı devre dışı kalır.
- Redis başlangıçta erişilemezse backend Memcached ile çalışmaya devam eder. Her iki dağıtık cache yoksa WARNING loglanır ve health yanıtında `cache.status=degraded`, `cache.active_layer=local` görünür.
- Development/test ortamında `ALLOW_DATABASE_FALLBACK=true` ise PostgreSQL bağlantısı kurulamadığında WARNING loglanır, `sqlite:///./matches.db` kullanılır ve health yanıtında `database.status=degraded` görünür.
- Production ortamı `ALLOW_DATABASE_FALLBACK=false` zorunlu kılar. PostgreSQL erişilemezse servis SQLite'a geçmez ve fail-fast kapanır.
- Redis bağlantısı kurulamazsa kategori bazlı bellek içi `TTLCache` kullanılır.
- Login rate limiter Redis kesintisinde process-local sayaçlarla korumayı sürdürür,
  varsayılan olarak her 30 saniyede bir Redis'i tekrar dener ve bağlantı
  sağlandığında eski local sayaçları temizleyerek Redis'i yeniden etkinleştirir.
- Celery broker yoksa sonuç kaydı yine tamamlanır; yeniden eğitim kuyruğa alınmaz ve PATCH yanıtında `broker_unavailable` döner.
- Broker açık fakat worker yoksa görev kuyruğa alınabilir; yanıtta `worker_unavailable` bildirilir.
- Worker hazırsa PATCH yanıtında `status=ready`, `task_queued=true` ve `task_id` bulunur.

### Değişiklik yaparken korunması gereken sözleşmeler

1. Backend importları `app.*` namespace'i üzerinden yapılmalıdır.
2. `all_probabilities` anahtarları daima `HOME_WIN`, `DRAW`, `AWAY_WIN` olmalı ve değerler yüzde formatında kalmalıdır.
3. Frontend API base URL'si `/api` ile biten `VITE_API_BASE_URL` değeridir; endpoint çağrıları tekrar `/api` eklememelidir.
4. Korumalı endpoint'ler `require_authenticated_user` dependency'sini kaybetmemelidir.
5. Redis/PostgreSQL/Celery kesintileri ana analiz ve sonuç kaydetme akışını çökertmemelidir.
6. Yeni matematiksel davranışlar `tests/` altında deterministik örneklerle test edilmelidir.
7. Teslimden önce şu kontroller geçmelidir:

```powershell
.\.venv\Scripts\python.exe -m ruff check backend/app backend/migrations tests scripts
.\.venv\Scripts\python.exe -m black --check backend/app backend/migrations tests scripts
.\.venv\Scripts\python.exe -m pytest tests -q
docker compose config --quiet
```

### Başka bir yapay zekâya verilebilecek hazır bağlam

Aşağıdaki metni README ile birlikte başka bir analiz aracına verebilirsiniz:

```text
Bu depo FastAPI + React/Vite tabanlı bir futbol tahmin ve value bet sistemidir.
Backend kaynak kodu backend/app, frontend kaynak kodu frontend/src altındadır.
Backend akışı endpoints.py -> StatsEngine -> ValueCalc -> opsiyonel ML -> repository
şeklindedir. Auth JWT access/refresh token kullanır. PostgreSQL yoksa SQLite, Redis
yoksa yerel TTL cache devreye girer. Gerçek sonuç kaydı Celery ile model eğitimini
kuyruğa alır. Matematik veya API sözleşmesi değiştirilecekse tests klasöründeki
Poisson, Dixon-Coles, de-vig, Kelly ve worker testleri korunmalı/genişletilmelidir.
Önce README'deki Bilinen Sınırlamalar ve Güvenlik bölümlerini değerlendir; ardından
ilgili kaynak dosyalarını okuyarak bulgularını dosya ve fonksiyon adıyla raporla.
```

## Gereksinimler

- Python `3.11` veya `3.12`
- Node.js `^20.19.0` veya `>=22.12.0` ve npm
- Opsiyonel: Redis 7, PostgreSQL 15
- Canlı veri için API-Football anahtarı

Python 3.13 ve üzeri proje metadata'sında desteklenmemektedir.

## Yerel Kurulum

### 1. Backend

PowerShell:

```powershell
cd "bet-ai-platform"
Copy-Item .env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
.\scripts\setup_venv.ps1
```

PowerShell script çalıştırma ilkesi engel olursa yalnızca mevcut terminal için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_venv.ps1
```

Linux/macOS:

```bash
cd bet-ai-platform
cp .env.example backend/.env
cp frontend/.env.example frontend/.env
./scripts/setup_venv.sh
```

### 2. Ortam değişkenleri

Yerel ve harici servis gerektirmeyen başlangıç için `backend/.env` dosyasını şu şekilde düzenleyin:

```env
ENVIRONMENT=development
DEBUG=true
API_FOOTBALL_KEY=DEMO_KEY
CLUBELO_ENABLED=true
CLUBELO_BASE_URL=http://api.clubelo.com
CLUBELO_TIMEOUT_SECONDS=15
CLUBELO_CACHE_HOURS=24
CLUBELO_CONFIDENCE=0.80
SPORTMONKS_ENABLED=false
SPORTMONKS_API_TOKEN=
SPORTMONKS_BASE_URL=https://api.sportmonks.com/v3/football
SPORTMONKS_PLAYER_LOOKBACK_DAYS=120
SPORTMONKS_PLAYER_LOOKBACK_MATCHES=10
ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS=300
ODDS_SNAPSHOT_CONFIDENCE=0.90
DATABASE_URL=sqlite:///./matches.db
ALLOW_DATABASE_FALLBACK=true
GOAL_TIME_DECAY_FACTOR=0.008
PLAYER_IMPACT_MIN_RATED_STARTERS=7
PLAYER_IMPACT_LOOKBACK_MATCHES=10
PLAYER_IMPACT_RATING_DECAY=0.85
PLAYER_CONTEXT_SYNC_MAX_FIXTURES=20
PLAYER_CONTEXT_SYNC_CONCURRENCY=3
FATIGUE_LOOKBACK_DAYS=14
FATIGUE_MATCH_WEIGHT=0.45
FATIGUE_REST_WEIGHT=0.40
FATIGUE_TRAVEL_WEIGHT=0.15
ENABLE_CATBOOST_CANDIDATE=true
ENABLE_LIGHTGBM_CANDIDATE=true
ML_BOOSTER_THREADS=2
ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES=30
ENSEMBLE_BMA_HALF_LIFE_DAYS=180
REDIS_URL=redis://localhost:6379/0
MEMCACHED_HOST=localhost
MEMCACHED_PORT=11211
MEMCACHED_TIMEOUT_SECONDS=2
JWT_SECRET_KEY=development-only-secret
JWT_REFRESH_SECRET_KEY=development-only-refresh-secret
MODEL_SIGNING_KEY=development-only-model-signing-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
ACCESS_TOKEN_COOKIE_NAME=bet_ai_access
REFRESH_TOKEN_COOKIE_NAME=bet_ai_refresh
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
REQUIRE_ORIGIN_HEADER=false
ADMIN_USERNAME=admin
ADMIN_PASSWORD=replace-with-a-strong-password
ADMIN_EMAIL=admin@example.com
FRONTEND_URL=http://localhost:5173
BACKEND_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
LOG_LEVEL=INFO
LOG_FORMAT=text
```

`API_FOOTBALL_KEY=DEMO_KEY` uygulamanın gömülü demo verisini kullanmasını sağlar. Canlı fikstürler için değeri geçerli API-Football anahtarıyla değiştirin. Redis kapalıysa Memcached, Memcached de kapalıysa süreç içi TTL cache kullanılır.

Frontend API adresi ayrı olarak `frontend/.env` dosyasında tutulur:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

`run.py` kök `.env` dosyasını da destekler; ancak Alembic ve doğrudan backend CLI komutları `backend/.env` dosyasını okuduğu için yerel geliştirmede önerilen konum `backend/.env`'dir.

PostgreSQL kullanacaksanız örnek bağlantı:

```env
DATABASE_URL=postgresql://betai:strong_password@localhost:5432/bet_ai
```

Production yapılandırması fail-fast doğrulanır. Aşağıdaki minimum politika sağlanmazsa servis başlamaz:

```env
ENVIRONMENT=production
DEBUG=false
API_FOOTBALL_KEY=live_api_key
DATABASE_URL=postgresql://betai:strong_password@postgres:5432/bet_ai
ALLOW_DATABASE_FALLBACK=false
GOAL_TIME_DECAY_FACTOR=0.01
PLAYER_IMPACT_MIN_RATED_STARTERS=7
PLAYER_IMPACT_LOOKBACK_MATCHES=10
PLAYER_IMPACT_RATING_DECAY=0.85
PLAYER_CONTEXT_SYNC_MAX_FIXTURES=20
PLAYER_CONTEXT_SYNC_CONCURRENCY=3
FATIGUE_LOOKBACK_DAYS=14
FATIGUE_MATCH_WEIGHT=0.45
FATIGUE_REST_WEIGHT=0.40
FATIGUE_TRAVEL_WEIGHT=0.15
ENABLE_CATBOOST_CANDIDATE=true
ENABLE_LIGHTGBM_CANDIDATE=true
ML_BOOSTER_THREADS=2
ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES=30
ENSEMBLE_BMA_HALF_LIFE_DAYS=180
JWT_SECRET_KEY=minimum_32_character_unique_access_secret
JWT_REFRESH_SECRET_KEY=minimum_32_character_unique_refresh_secret
MODEL_SIGNING_KEY=minimum_32_character_unique_model_signing_secret
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
REQUIRE_ORIGIN_HEADER=true
ADMIN_PASSWORD=minimum-12-character-password
FRONTEND_URL=https://bets.example.com
BACKEND_CORS_ORIGINS=https://bets.example.com
```

`BACKEND_CORS_ORIGINS` virgülle ayrılmış liste veya JSON dizisi kabul eder. Production'da wildcard, HTTP origin, SQLite, varsayılan API anahtarı, zayıf/aynı JWT secret'lar, varsayılan veya 32 karakterden kısa model imza anahtarı ve eksik `Origin` başlığı reddedilir. CLI ve servis-to-servis state-changing isteklerinde de production ortamında güvenilen `Origin` başlığı gönderilmelidir.

### 3. Backend'i çalıştırma

```powershell
.\.venv\Scripts\python.exe run.py --check
.\.venv\Scripts\python.exe run.py --reload
```

Backend varsayılan olarak `http://127.0.0.1:8000` adresinde açılır:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Sağlık durumu: `http://127.0.0.1:8000/`

Alternatif olarak Windows'ta `start-backend.ps1` veya `start-backend.bat` kullanılabilir.

### 4. Frontend'i çalıştırma

İkinci terminalde:

```powershell
cd frontend
npm install
npm run dev
```

Vite arayüzü varsayılan olarak `http://localhost:5173` adresinde çalışır. Backend adresi `frontend/.env` içindeki `VITE_API_BASE_URL` ile yapılandırılır. Arayüz, korunan endpoint'ler için login ve otomatik refresh-token akışını kullanır.

## API

Tüm işlev endpoint'leri `/api` prefix'i altındadır.

| Metot | Endpoint | Açıklama |
| --- | --- | --- |
| `GET` | `/` | Servis, veri modu ve ML durumunu döndürür |
| `GET` | `/health/live` | Sürecin liveness durumunu döndürür |
| `GET` | `/health/ready` | Veritabanı ve cache readiness durumunu döndürür |
| `GET` | `/metrics` | Prometheus uyumlu HTTP metriklerini döndürür |
| `POST` | `/api/auth/login` | HttpOnly access ve refresh cookie oluşturur |
| `POST` | `/api/auth/register` | `ALLOW_SELF_REGISTRATION=true` ise sınırlı rolle hesap ve cookie oturumu oluşturur |
| `POST` | `/api/auth/refresh` | Refresh cookie ile cookie çiftini yeniler |
| `GET` | `/api/auth/session` | Access cookie oturumunu doğrular |
| `GET` | `/api/auth/sessions` | Kullanıcının aktif refresh oturumlarını güvenli metadata ile listeler |
| `DELETE` | `/api/auth/sessions/{session_id}` | Kullanıcının seçili refresh oturumunu iptal eder |
| `POST` | `/api/auth/logout` | Auth cookie'lerini siler |
| `GET` | `/api/status` | API-Football veri kaynağının `demo` veya `live` modunu döndürür |
| `GET` | `/api/admin/users` | `users:manage` izniyle kullanıcıları ve rollerini listeler |
| `POST` | `/api/admin/users` | Güçlü parola ve en az bir rolle kullanıcı oluşturur |
| `PATCH` | `/api/admin/users/{user_id}` | Rolleri/aktifliği günceller ve mevcut oturumları iptal eder |
| `GET` | `/api/admin/roles` | `roles:manage` izniyle rol ve permission kataloğunu döndürür |
| `GET` | `/api/leagues` | Desteklenen ligleri listeler |
| `GET` | `/api/fixtures/upcoming?days=7&limit=100` | Belirtilen ufuktaki fikstürü kronolojik getirir |
| `GET` | `/api/fixtures/{fixture_id}/prefill` | Fikstür analiz girdisini hazırlar |
| `POST` | `/api/analyze/preview` | Tahmini kaydetmeden hesaplanan model girdilerini ve eksik veri durumlarını döndürür |
| `POST` | `/api/analyze` | Manuel verilerle analiz yapar ve kaydeder |
| `POST` | `/api/analyze/fixture/{fixture_id}` | API-Football fikstürünü analiz eder |
| `GET` | `/api/history` | Varsayılan olarak son tahminleri; `paginated=true` ile sunucu tarafı arama, filtreleme, sıralama ve sayfalama sonucu döndürür |
| `PATCH` | `/api/history/{record_id}/result` | Gerçek sonucu ve opsiyonel skoru kaydeder |
| `GET` | `/api/ml/labeling-queue?limit=20` | Aktif öğrenme için en belirsiz etiketsiz tahminleri sıralar |
| `GET` | `/api/ml/status` | Aktif artifact sürümü, kalite metrikleri ve inference sayaçlarını döndürür |
| `POST` | `/api/ml/rollback` | `users:manage` izniyle doğrulanmış önceki model artifact'ına döner |
| `POST` | `/api/backtest` | Kayıtlı tahminlerde strateji simülasyonu yapar |
| `GET` | `/api/audit` | Tahmin kalitesi ve performans metriklerini üretir |
| `GET` | `/api/operations/data-quality` | Tarihsel veri, etiket, closing odds, provenance ve son senkronizasyon kalitesini döndürür |

Analiz, geçmiş, backtest ve audit endpoint'leri geçerli access cookie gerektirir. Tarayıcı istekleri `credentials: "include"`, curl örnekleri cookie jar kullanmalıdır.

### Aktif öğrenme etiketleme kuyruğu

`GET /api/ml/labeling-queue?limit=20`, sonucu henüz girilmemiş tahminleri normalize entropy ve ilk iki sınıf arasındaki margin ile sıralar. Yüksek `uncertainty_score`, kaydın insan tarafından önce etiketlenmesinin modele daha fazla bilgi kazandırabileceğini belirtir. Endpoint `history:update_result` permission'ı gerektirir; otomatik/pseudo etiket üretmez.

```json
{
  "strategy": "uncertainty_sampling",
  "candidates": [
    {
      "id": 42,
      "fixture_id": 123456,
      "home_team": "Home FC",
      "away_team": "Away FC",
      "prediction": "HOME_WIN",
      "probabilities": {"HOME_WIN": 34.2, "DRAW": 33.1, "AWAY_WIN": 32.7},
      "uncertainty_score": 0.993821,
      "normalized_entropy": 0.9998,
      "class_margin": 0.011
    }
  ],
  "candidate_count": 1,
  "labeled_samples_count": 87,
  "remaining_to_threshold": 113
}
```

### Oturum açma

```bash
curl -c cookies.txt -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"replace-with-a-strong-password"}'
```

### Manuel analiz örneği

```bash
curl -b cookies.txt -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Fenerbahce",
    "away_team": "Galatasaray",
    "kickoff": "2030-07-20T18:00:00Z",
    "home_stats": {"form": 85, "attack": 88, "defense": 82, "xg": 2.1},
    "away_stats": {"form": 78, "attack": 84, "defense": 76, "xg": 1.8},
    "odd": 2.30,
    "opening_odds_1x2": {
      "HOME_WIN": 2.50,
      "DRAW": 3.20,
      "AWAY_WIN": 2.90
    },
    "current_odds_1x2": {
      "HOME_WIN": 2.30,
      "DRAW": 3.30,
      "AWAY_WIN": 3.10
    },
    "opening_odds_at": "2030-07-17T18:00:00Z",
    "current_odds_at": "2030-07-20T17:00:00Z"
  }'
```

Doğrulama sınırları: form/hücum/savunma `0-100`, xG `0-5`, oran `1.0` değerinden büyük olmalıdır.
Odds snapshot'ları opsiyoneldir; verildiğinde `HOME_WIN`, `DRAW` ve `AWAY_WIN`
alanlarının üçü de geçerli decimal oran içermelidir. Her snapshot timezone içeren
bir `*_odds_at` alanı ve maçın `kickoff` zamanını gerektirir; kickoff anındaki veya
sonrasındaki oranlar veri sızıntısını engellemek için reddedilir.

### Gerçek sonuç kaydetme

```bash
curl -b cookies.txt -X PATCH http://localhost:8000/api/history/1/result \
  -H "Content-Type: application/json" \
  -d '{"actual_result":"HOME_WIN","actual_score_home":2,"actual_score_away":1}'
```

`actual_result` yalnızca `HOME_WIN`, `DRAW` veya `AWAY_WIN` olabilir.

### Backtest örneği

```bash
curl -b cookies.txt -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "initial_bankroll": 1000,
    "strategy": "fractional_kelly",
    "flat_stake_amount": 10,
    "kelly_fraction": 0.25,
    "min_edge_pct": 3
  }'
```

Desteklenen stratejiler: `kelly`, `fractional_kelly`, `flat`.

### Backtest yanıtı

`POST /api/backtest`, yalnızca gerçek sonucu girilmiş ve `min_edge_pct` eşiğini geçen tahminleri kronolojik olarak simüle eder.

```json
{
  "initial_bankroll": 1000.0,
  "final_bankroll": 1042.75,
  "total_roi_pct": 4.28,
  "total_bets": 18,
  "wins": 10,
  "losses": 8,
  "win_rate_pct": 55.56,
  "accuracy_pct": 55.56,
  "max_drawdown_pct": 3.41,
  "sharpe_ratio": 1.2841,
  "sortino_ratio": 1.9473,
  "calibration_score": 0.0832,
  "bankroll_history": [1000.0, 1015.0, 1005.0, 1020.08, 1042.75]
}
```

| Alan | Açıklama |
| --- | --- |
| `initial_bankroll` | Simülasyon başlangıç bakiyesi |
| `final_bankroll` | Simülasyon sonundaki bakiye |
| `total_roi_pct` | Başlangıç ve final bakiye arasındaki toplam getiri yüzdesi |
| `total_bets` | Filtrelerden geçip simüle edilen bahis sayısı |
| `wins` / `losses` | Kazanan ve kaybeden bahis sayıları |
| `win_rate_pct` | Kazanan bahis yüzdesi |
| `accuracy_pct` | `win_rate_pct` ile aynı değeri taşıyan geriye uyumlu alias |
| `max_drawdown_pct` | Tepe bakiyeden görülen en büyük yüzdesel düşüş |
| `sharpe_ratio` | Sıfır risksiz faiz ve 252 dönem yıllıklaştırmayla Sharpe oranı |
| `sortino_ratio` | Yalnızca negatif getirileri kullanan Sortino oranı |
| `calibration_score` | Expected Calibration Error; düşük değer daha iyidir |
| `bankroll_history` | Her simüle edilen bahis sonrasındaki bakiye serisi |

Çözümlenmiş tahmin yoksa sayısal metrikler `0.0`, `total_bets` ise `0` döner ve `bankroll_history` yalnızca başlangıç bakiyesini içerir.

### Audit yanıtı

`GET /api/audit`, gerçek sonucu girilmiş tüm tahminleri bir birimlik bahis varsayımıyla denetler.

```json
{
  "total_predictions": 24,
  "win_rate_pct": 58.33,
  "total_roi_pct": 12.5,
  "brier_score": 0.4217,
  "avg_clv_pct": 2.14
}
```

| Alan | Açıklama |
| --- | --- |
| `total_predictions` | Gerçek sonucu bulunan tahmin sayısı |
| `win_rate_pct` | Tahmini gerçek sonuçla eşleşen kayıtların yüzdesi |
| `total_roi_pct` | Birim bahis başına ortalama ROI yüzdesi |
| `brier_score` | HOME/DRAW/AWAY için çok sınıflı Brier skoru; düşük değer daha iyidir |
| `avg_clv_pct` | Closing odds bulunan kayıtların ortalama Closing Line Value yüzdesi |

Çözümlenmiş tahmin yoksa endpoint yukarıdaki beş alanı da sıfır değerleriyle döndürür.

## Geliştirme Komutları

Kök `Makefile` yaygın işlemleri tek komutta birleştirir:

```bash
make install     # Python ve frontend bağımlılıkları
make dev         # FastAPI ve Vite geliştirme sunucuları
make test        # Pytest
make lint        # Ruff, Black ve ESLint
make typecheck   # Mypy ile tiplenmiş domain sınırı
make migrate     # Alembic upgrade head
make migration-check # Model/migration farkını kontrol et
make migration-check-docker # Compose PostgreSQL üzerinde şema farkını kontrol et
make create-admin # Eksikse env ayarlarından admin oluştur
make create-user ARGS='--username analyst1 --email analyst@example.com --role analyst'
make docker-up   # Tüm Compose servisleri
make docker-down
make openapi     # docs/openapi.json dosyasını yeniler
```

Windows'ta GNU Make yoksa hedeflerin içindeki komutlar doğrudan çalıştırılabilir. Farklı Python executable için `make PYTHON=backend/.venv/Scripts/python.exe test` kullanılabilir.

## Kullanıcı ve RBAC Yönetimi

Migration varsayılan `admin`, `analyst` ve `viewer` rollerini ve bunların permission eşleşmelerini oluşturur. Docker akışındaki `bootstrap-admin` servisi yalnızca admin hesabı yoksa `ADMIN_USERNAME`, `ADMIN_EMAIL` ve `ADMIN_PASSWORD` değerlerinden ilk hesabı oluşturur; mevcut hesabın parolasını otomatik değiştirmez.

| Rol | Permission'lar |
| --- | --- |
| `admin` | Tüm permission'lar; kullanıcı ve rol yönetimi dahil |
| `analyst` | `analysis:create`, `history:read`, `history:update_result`, `backtest:run`, `audit:read` |
| `viewer` | `history:read`, `audit:read` |

Frontend session yanıtındaki permission listesini kullanır. Analiz formu, gerçek sonuç girişi ve backtest paneli yalnızca sırasıyla `analysis:create`, `history:update_result` ve `backtest:run` yetkileri bulunan kullanıcılara gösterilir. `viewer` rolü salt-okunur arayüz kullanır; backend permission kontrolleri nihai yetki sınırı olmaya devam eder.

Yeni kullanıcı oluşturmak için parola terminal argümanı olarak verilmez. İnteraktif olarak sorulur veya otomasyon sırasında `NEW_USER_PASSWORD` environment değişkeninden okunur:

```bash
python scripts/create_user.py \
  --username analyst1 \
  --email analyst1@example.com \
  --role analyst
```

Bir kullanıcıya birden fazla rol vermek için `--role` tekrarlanabilir. Parola en az 12 karakter olmalıdır.

RBAC tabloları:

```text
users <-> user_roles <-> roles <-> role_permissions <-> permissions
  |
  +---- refresh_sessions
```

## Veritabanı Migration Akışı

Şema yönetimi Alembic üzerinden yapılır; uygulama startup sırasında tablo oluşturmaz veya değiştirmez.

```bash
# Mevcut migration'ları uygula
python -m alembic -c backend/alembic.ini upgrade head

# Model değişikliğinden sonra revision üret
cd backend
python -m alembic -c alembic.ini revision --autogenerate -m "add example field"

# Üretilen revision'ı incele, ardından uygula
python -m alembic -c alembic.ini upgrade head

# Model ile migration head arasında fark kontrolü
python -m alembic -c alembic.ini check
```

Yeni revision dosyaları commit edilmeden önce hem boş veritabanında hem production yedeğinin anonimleştirilmiş kopyasında test edilmelidir. `downgrade` veri kaybına yol açabilecekse revision içinde açıkça belgelenmelidir.

## Test ve Kod Kalitesi

Geliştirme bağımlılıklarını kurmak için:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --cov=backend/app --cov-report=term-missing --cov-report=xml
.\.venv\Scripts\python.exe -m ruff check backend/app tests
.\.venv\Scripts\python.exe -m black --check backend/app tests
.\.venv\Scripts\python.exe -m mypy backend/app
npm --prefix frontend run lint
npm --prefix frontend run test
npm --prefix frontend run build
```

Test paketi StatsEngine skor matrisi/xG uçları/Dixon-Coles ağırlıkları, ValueCalc, API-Football retry/demo/prefill fallback, ML feature/Elo/readiness/olasılık ve artifact imza güvenliğini, SHAP/feature-importance açıklanabilirlik fallback'lerini, backtest stratejileri/Kelly limitlerini, cookie auth, Redis rate-limiter recovery, RBAC yönetimi, DB/cache fallback, Celery sağlık/enqueue, frontend container davranışı, geçmiş filtreleri, admin rol seçimi ve bankroll seri dönüşümünü kapsar. CI aynı kontrolleri çalıştırır ve `coverage.xml` üretir.

CI, Ruff adımından önce `pip-audit` ile hem `requirements.txt` hem de
`requirements-dev.txt` bağımlılıklarını denetler. İncelenmiş düşük önem seviyeli
istisnalar, kaldırma gerekçesi ve hedef tarihi belirtilerek
`.github/pip-audit-ignore.txt` dosyasına eklenebilir. Dependabot, backend pip ve
frontend npm bağımlılıklarını her pazartesi kontrol eder.

Mypy `backend/app` altındaki tüm backend paketini zorunlu kontrol eder. ORM modelleri SQLAlchemy 2 `DeclarativeBase`, `Mapped` ve `mapped_column` yapısındadır; nullable legacy tahmin alanları audit, backtest, task ve endpoint katmanlarında güvenli varsayılanlarla ele alınır.

CI, `coverage.xml` dosyasını Codecov'a yükler ve toplam backend coverage değerinin `%73` altına düşmesine izin vermez. Badge yayımlamak için repository Codecov'da etkinleştirildikten sonra servisin verdiği repository-specific badge URL'si README'nin başına eklenmelidir; repository/organizasyon adresi bilinmediği için yanıltıcı bir URL sabitlenmemiştir.

## OpenAPI Şeması

Swagger tarafından üretilen sürümlenebilir şema `docs/openapi.json` dosyasındadır. API sözleşmesi değiştiğinde `make openapi` çalıştırılmalıdır.

## Docker Durumu

`docker-compose.yml` PostgreSQL, Redis, tek seferlik Alembic migration, backend, Celery worker, Celery Beat ve frontend servislerini tanımlar. Backend, worker ve Beat yalnızca migration başarıyla tamamlandıktan sonra başlar. Worker/Beat `restart: unless-stopped` politikasıyla broker yeniden başlatmalarından sonra toparlanır:

```bash
docker compose up --build -d
docker compose ps
```

Doğrulanan adresler: backend `http://localhost:8000`, frontend `http://localhost:3000`. PostgreSQL, Redis ve Memcached healthcheck'leri Compose tarafından izlenir.

## Bilinen Sınırlamalar

- Development SQLite fallback'i tek process ve yerel kullanım içindir; production'da zorunlu olarak kapalıdır.
- Self-service kayıt güvenlik nedeniyle varsayılan kapalıdır. Açıldığında hesaplar yalnızca `SELF_REGISTRATION_ROLE=viewer|analyst` ile oluşturulur; e-posta doğrulaması ve parola sıfırlama sağlayıcısı henüz yoktur.
- İlk baseline migration mevcut eski tabloyu koruyup eksik kolonları tamamlar; sonraki tüm model değişiklikleri yeni revision gerektirir. CI temiz veritabanında `upgrade head` ve `alembic check` çalıştırır; production revision'ları ayrıca anonimleştirilmiş yedek üzerinde doğrulanmalıdır.
- ML modeli minimum etiket eşiğine ulaşmadan istatistik motoru güvenli fallback olarak kullanılır. Semi-supervised pseudo-label üretimi bilinçli olarak etkin değildir; belirsiz örnekler `/api/ml/labeling-queue` üzerinden insan etiketlemesine yönlendirilir.

Backend imajı Vite production build'ini çok aşamalı olarak üretir ve `/ui` ile `/ui/*` rotalarında aynı SPA giriş dosyasını sunar; statik asset'ler fingerprint'li `/assets/*` yolundan servis edilir.

## Güvenlik

### HashiCorp Vault secret kaynağı

Varsayılan `SECRET_PROVIDER=env` mevcut environment akışını korur. Vault KV v2 kullanmak için production process'ine yalnızca bootstrap kimlik bilgilerini verin:

```env
SECRET_PROVIDER=vault
VAULT_ADDR=https://vault.example.com
VAULT_MOUNT_POINT=secret
VAULT_SECRET_PATH=bet-ai/production
VAULT_ROLE_ID=application-role-id
VAULT_SECRET_ID=runtime-injected-secret-id
```

KV v2 kaydında `API_FOOTBALL_KEY`, `DATABASE_URL`, `JWT_SECRET_KEY`, `JWT_REFRESH_SECRET_KEY`, `MODEL_SIGNING_KEY` ve `ADMIN_PASSWORD` zorunludur; `REDIS_URL` opsiyoneldir. Yalnızca bu allowlist'teki anahtarlar process environment'a aktarılır. `VAULT_TOKEN` doğrudan token akışı için desteklenir, ancak production'da kısa ömürlü AppRole/SecretID tercih edilmelidir. Vault kimlik doğrulaması veya secret okuması başarısızsa uygulama fail-fast kapanır; env fallback yaparak güvenliği sessizce zayıflatmaz.

Health yanıtı yalnızca `secrets.provider` ve `secrets.loaded_keys` sayısını gösterir; secret değerleri döndürülmez.

### Azure Key Vault secret kaynağı

Azure ortamında Managed Identity/Workload Identity ile `DefaultAzureCredential` kullanılır:

```env
SECRET_PROVIDER=azure_key_vault
AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net
AZURE_KEY_VAULT_PREFIX=bet-ai
```

Secret adları `<prefix>-<küçük-harf-env-anahtarı>` biçimindedir; örneğin `bet-ai-jwt-secret-key`. HashiCorp Vault ile aynı allowlist ve zorunlu anahtar politikası uygulanır. Kimlik doğrulama veya zorunlu secret okuması başarısızsa servis env fallback yapmadan kapanır.

### Cookie, CSRF ve brute-force politikası

- Access/refresh token'lar `HttpOnly + Secure + SameSite` cookie'lerinde tutulur; JavaScript token değerlerini okuyamaz.
- Ayrı `bet_ai_csrf` cookie'si okunabilir durumdadır ve tüm oturumlu state-changing isteklerde `X-CSRF-Token` başlığıyla double-submit doğrulaması yapılır. Origin whitelist kontrolü ayrıca devam eder.
- Başarısız login denemeleri kullanıcı+IP karmasıyla Redis'te sayılır. Redis yoksa process-local fail-safe devreye girer ve bağlantı periyodik olarak tekrar denenir. Varsayılan politika 5 deneme/300 saniye ve 900 saniye kilittir.
- Refresh token her kullanımda rotate edilir; tekrar kullanım bütün token ailesini iptal eder. Kullanıcılar `/api/auth/sessions` ile cihaz oturumlarını görüp tek tek kapatabilir.

- `.env`, veritabanı dosyaları ve model artifact'larını kaynak kontrolüne eklemeyin.
- Production'da CORS origin listesini yalnızca güvenilen domain'lerle sınırlandırın.
- `ENVIRONMENT=production` iken güvenlik kuralları fail-fast uygulanır; geçersiz ayarlar sessiz fallback ile zayıflatılmaz.
- State-changing istekler (`POST`, `PUT`, `PATCH`, `DELETE`) için origin doğrulaması yapılır; production'da `REQUIRE_ORIGIN_HEADER=true` zorunludur.
- API'yi TLS kullanan bir reverse proxy arkasında yayınlayın.
- Demo/varsayılan secret değerleri production ortamında kullanmayın.
- `ADMIN_PASSWORD`, `JWT_SECRET_KEY`, `JWT_REFRESH_SECRET_KEY` ve `MODEL_SIGNING_KEY` değerlerini secret manager üzerinden sağlayın.
- Cookie auth nedeniyle production frontend ve API'yi HTTPS üzerinden, uyumlu same-site domain yapısıyla yayınlayın.
- `COOKIE_SAMESITE=none` yalnızca cross-site deployment zorunluysa ve `COOKIE_SECURE=true` ile kullanılmalıdır.
- Harici API anahtarlarını frontend koduna veya loglara yazmayın.

## Lisans

Proje, kök dizindeki `LICENSE` dosyasında yer alan MIT lisansı altında dağıtılır.
