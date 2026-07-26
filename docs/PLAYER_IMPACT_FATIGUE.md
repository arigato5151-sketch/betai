# Oyuncu Etkisi ve Yorgunluk Feature'ları

## Kapsam

`ml_features_v7`, kadro kalitesi ve fikstür yükünü modele üç yeni alanla taşır:

| Feature | Aralık | Nötr değer | Anlam |
|---|---:|---:|---|
| `home_team_strength_ratio` | `0.70–1.05` | `1.0` | Ev sahibi güncel/referans kadro kalitesi |
| `away_team_strength_ratio` | `0.70–1.05` | `1.0` | Deplasman güncel/referans kadro kalitesi |
| `fatigue_index` | `-1.0–1.0` | `0.0` | Deplasman yükü eksi ev sahibi yükü |

Eski `ml_features_v1-v6` snapshot'ları desteklenmeye devam eder. Bu üç alan eski
snapshot'larda yukarıdaki nötr değerlerle doldurulur; mevcut kayıtların migration ile
yeniden yazılması gerekmez.

## Player Impact Score

### Nokta-zamanlı rating

Tamamlanmış bir maçın oyuncu performansı
`historical_player_performances` tablosunda saklanır. Her satır fixture, lig, kickoff,
takım ve oyuncu kimliğiyle birlikte şu bağlamı taşır:

- ilk 11'de başlayıp başlamadığı;
- dakika ve provider rating'i;
- pozisyon, gol ve asist;
- veri kaynağı ile ingestion/update zamanları.

`(fixture_id, player_id)` benzersizdir. Rating sorgusu katı
`performance.kickoff < prediction.kickoff` filtresi kullanır. Böylece tahmin edilen
maçın maç içi veya maç sonu rating'i aynı maçın feature'ına giremez.

Her oyuncu için en fazla `PLAYER_IMPACT_LOOKBACK_MATCHES=10` geçmiş rating alınır.
En yeni gözlem `1.0`, daha eskiler sırasıyla
`PLAYER_IMPACT_RATING_DECAY ** age` ağırlığıyla ortalanır. Geçmiş rating kapsamı
yetersiz olan gelecekteki bir fixture için güncel sezon aggregate'i kullanılabilir;
tarihsel replay'de bu canlı fallback kapalıdır. Rating bulunmayan fakat dakika ve
gol/asist katkısı bulunan satırlar, 90 dakika/maç başına katkı fallback'iyle korunur.

Tarihsel senkronizasyon, API-Football `fixtures/players` yanıtlarını API kotasını
koruyan bounded bir backfill ile toplar. Her çalışmada en fazla
`PLAYER_CONTEXT_SYNC_MAX_FIXTURES=20` eksik fixture,
`PLAYER_CONTEXT_SYNC_CONCURRENCY=3` eşzamanlı istekle işlenir. Başarılı fixture'lar
bir sonraki senkronizasyonda tekrar istenmez; limit `0` yapılarak backfill kapatılabilir.

### Team Strength Ratio

Referans kadro, lig ayrımı olmadan takımın tahmin zamanından önceki son geçerli ilk
11'idir. Böyle bir
kadro yoksa canlı sezon verisindeki dakika, maç sayısı ve ardından rating sırasıyla
en yüksek kullanım payına sahip tipik 11 türetilir.
Hesap ancak referans 11'in en az `PLAYER_IMPACT_MIN_RATED_STARTERS=7` oyuncusu
rating'e sahipse etkinleşir. Kapsanmayan referans oyuncular, bilinen ilk 11 rating
ortalamasıyla tamamlanır; böylece eksik veri taraflardan birini yapay biçimde
güçlendirmez veya zayıflatmaz.

```text
reference_total = sum(reference_XI_player_impact)
strength_ratio = adjusted_current_total / reference_total
```

Güncel ilk 11 doğrulanmış ve yeterince rating'liyse doğrudan güncel kadro
karşılaştırılır. İlk 11 henüz açıklanmamışsa sakat/cezalı ve şüpheli oyuncular
referans kadro üzerinden projekte edilir:

- replacement katkısı `PLAYER_IMPACT_REPLACEMENT_FACTOR=0.75` ile kırpılır;
- rating'i referans ilk 11 ortalamasından yüksek eksik oyuncu kritik kabul edilir;
- kritik eksikler `PLAYER_CRITICAL_ABSENCE_WEIGHT=0.25` ile ek kayıp üretir;
- şüpheli oyuncular `PLAYER_QUESTIONABLE_ABSENCE_WEIGHT=0.35` ağırlığında işlenir.

Son oran `PLAYER_IMPACT_MIN_STRENGTH_RATIO=0.70` ile
`PLAYER_IMPACT_MAX_STRENGTH_RATIO=1.05` arasında sınırlandırılır.

### StatsEngine xG bağlantısı

`poisson_dixon_coles_v5`, ev/deplasman saha çarpanını uyguladıktan sonra takım
lambda değerini aşağıdaki multiplier ile günceller:

```text
xg_multiplier =
    clamp(
        1 + (strength_ratio - 1) * PLAYER_IMPACT_XG_ELASTICITY,
        PLAYER_IMPACT_MIN_XG_MULTIPLIER,
        PLAYER_IMPACT_MAX_STRENGTH_RATIO,
    )
```

Varsayılan elasticity `1.0`, minimum çarpan `0.75`'tir. Kritik bir eksik
`adjusted_current_total` değerini daha çok düşürdüğü için xG cezası da düşük rating'li
bir eksikten daha büyüktür. Çarpan son Poisson güvenlik sınırından önce uygulanır;
nihai lambda yine `0.35–3.4` aralığındadır.

Rating kapsamı, referans ilk 11 veya oyuncu kimliği yetersizse
`data_available=false`, `strength_ratio=1.0` ve `xg_multiplier=1.0` kullanılır.
Ham eksik oyuncu sayısı tek başına xG cezası oluşturmaz.

## Rest & Travel Index

Yorgunluk hesabı her takım için üç normalize bileşenden oluşur:

```text
match_load  = clamp(last_14_days_matches / 4, 0, 1)
rest_load   = clamp((ideal_rest_days - actual_rest_days) / ideal_rest_days, 0, 1)
travel_load = clamp(away_travel_distance_km / 3000, 0, 1)

team_fatigue =
    0.45 * match_load
  + 0.40 * rest_load
  + 0.15 * travel_load

fatigue_index = clamp(away_fatigue - home_fatigue, -1, 1)
```

Ev sahibinin seyahat yükü bu maç için `0` kabul edilir. Pozitif `fatigue_index`,
deplasmanın daha yorgun; negatif değer ev sahibinin daha yoğun fikstüre sahip
olduğunu gösterir.

Maç sayımı tüm müsabakalarda tahmin kickoff'undan önceki
`FATIGUE_LOOKBACK_DAYS=14` günü kapsar. Aynı kickoff'taki veya gelecekteki satırlar
sayılmaz. Son maç yoksa rest yükü `0`; mesafe yoksa travel yükü `0`; fixture tarihi
geçersizse tüm fatigue feature'ı `0.0` olur. Schedule sorgusu, dinlenme bileşenini
kaybetmemek için maç sayımı penceresi ile `FATIGUE_IDEAL_REST_DAYS` değerinin
büyüğünü kapsar. Yalnız tek takımın schedule geçmişi bulunuyorsa match/rest
bileşenleri iki taraf için nötr tutulur; doğrulanmış seyahat mesafesi yine kullanılabilir.

Deplasman mesafesi öncelik sırasıyla:

1. doğrulanmış analiz girdisindeki `away_travel_distance_km`;
2. `team_locations` tablosundaki deplasman ve ev sahibi koordinatlarının Haversine
   uzaklığı;
3. eksik koordinat veya eşleşme durumunda `0.0`.

`team_locations`, `(data_source, team_id)` anahtarında benzersizdir. Enlem ve boylam
nullable'dır; repository yalnız sonlu ve geçerli aralıktaki koordinatları kabul eder.
Nötr saha veya farklı stat biliniyorsa takım-merkezli Haversine yerine analiz
isteğindeki doğrulanmış `away_travel_distance_km` override'ı kullanılmalıdır.

API-Football takım/stadyum yanıtları güvenilir koordinat sağlamadığı için konumlar
uydurulmaz. Admin yetkili istemci, doğrulanmış takım/tesis koordinatlarını toplu ve
idempotent biçimde `PUT /api/admin/team-locations` ile yükleyebilir; kayıtlar
`GET /api/admin/team-locations` ile denetlenebilir. Enlem ve boylam birlikte
sağlanmalıdır. Böylece normal analiz akışı mesafeyi sunucu tarafında otomatik
hesaplar; veri yüklenmemiş kurulumlarda nötr `0.0` fallback korunur.

## Veritabanı ve migration

İki bağlam tablosu `20260726_0010_add_player_impact_context.py` revision'ıyla eklenir:

```text
historical_fixtures.fixture_id
    └── historical_player_performances.fixture_id (ON DELETE CASCADE)

team_locations
    └── UNIQUE(data_source, team_id)
```

Player performance upsert'i PostgreSQL ve SQLite'ta idempotenttir. Takım+kickoff ve
oyuncu+kickoff indeksleri point-in-time sorgularını destekler. Konum bulunmadığında
Haversine yardımcı fonksiyonu güvenli `0.0 km` döndürür.

Migration uygulama ve metadata kontrolü:

```bash
python -m alembic -c backend/alembic.ini upgrade head
python -m alembic -c backend/alembic.ini check
```

## Yapılandırma

```env
PLAYER_IMPACT_MIN_RATED_STARTERS=7
PLAYER_IMPACT_LOOKBACK_MATCHES=10
PLAYER_IMPACT_RATING_DECAY=0.85
PLAYER_IMPACT_REPLACEMENT_FACTOR=0.75
PLAYER_IMPACT_MIN_STRENGTH_RATIO=0.70
PLAYER_IMPACT_MAX_STRENGTH_RATIO=1.05
PLAYER_IMPACT_XG_ELASTICITY=1.0
PLAYER_IMPACT_MIN_XG_MULTIPLIER=0.75
PLAYER_CRITICAL_ABSENCE_WEIGHT=0.25
PLAYER_QUESTIONABLE_ABSENCE_WEIGHT=0.35
PLAYER_CONTEXT_SYNC_MAX_FIXTURES=20
PLAYER_CONTEXT_SYNC_CONCURRENCY=3

FATIGUE_LOOKBACK_DAYS=14
FATIGUE_MATCH_REFERENCE_COUNT=4
FATIGUE_IDEAL_REST_DAYS=7
FATIGUE_TRAVEL_REFERENCE_KM=3000
FATIGUE_MATCH_WEIGHT=0.45
FATIGUE_REST_WEIGHT=0.40
FATIGUE_TRAVEL_WEIGHT=0.15
```

Üç fatigue ağırlığı tam olarak `1.0` toplamına sahip olmalıdır; aksi durumda
`Settings` uygulamayı fail-fast durdurur. Tüm oran, pencere ve koordinat girdileri
Pydantic/repository sınır kontrollerinden geçer.

Bu varsayılanlar güvenli başlangıç değerleridir, out-of-time kalibrasyon sonucu
değildir. Kalibrasyon durumu ve sonraki doğrulama koşulları
[`CALIBRATION.md`](CALIBRATION.md) belgesinde tutulur.
