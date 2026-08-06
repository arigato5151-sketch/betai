# Ücretsiz Veri Kaynakları

Bu belge, Bet AI Platform içinde otomatik kullanılan ücretsiz kaynakları ve veri
kalitesi nedeniyle tahmin girdisine alınmayan adayları kaydeder.

## Aktif kaynaklar

| Kaynak | Kullanılan veri | Kapsam / yöntem | Lisans ve sınır |
| --- | --- | --- | --- |
| API-Football | Fikstür, sonuç, oran, kadro, sakatlık, istatistik | Ücretsiz planın erişebildiği ligler | Kota ve abonelik kapsamı |
| FixtureDownload | Gelecek fikstür, UEFA tarihsel sonuç | Desteklenen sezon feed'leri | Her 15 dakikada cache |
| OpenLigaDB | Bundesliga, 2. Bundesliga ve UCL fikstürü | Anahtarsız resmî API | ODbL; arayüzde atıf |
| football-data.co.uk | Sonuç, şut, kart, korner ve tarihsel oran | 18 ulusal lig; İskoçya, Avusturya, İsviçre ve Danimarka dahil | Ücretsiz CSV arşivi |
| OpenFootball | Yunanistan Süper Ligi fikstür ve sonuçları | Sezonluk CC0 JSON; günlük idempotent sync | Public domain / CC0 |
| Understat | Gözlemlenmiş maç xG | Avrupa'nın büyük beş ligi | İstek aralığı ve günlük sync |
| ClubElo | Takım güç puanı | İsim ve tarih eşlemesi | Günlük cache |
| Sportmonks | Oyuncu rating, kadro ve sakatlık | Yapılandırılmış ücretsiz plan kapsamı | Token ve abonelik kapsamı |
| TheSportsDB | Tamamlayıcı fikstür | Anahtarsız günlük futbol feed'i | Crowd-sourced; tek başına güven kaynağı değil |
| Wikidata / GeoNames | Kulüp ve şehir koordinatı | Seyahat/yorgunluk hesabı | Güven skoru ve yerel cache |
| StatsBomb Open Data | Olay akışı, xG, şut, kart, korner ve ilk 11 | Seçili açık sezonlar; olaylar günlük sınırlı batch ile zenginleştirilir | Kaynak/logo atfı gerekir |
| Open-Meteo | Maç saati sıcaklık, yağış ve rüzgâr | Stadyum/ev sahibi konumu bilinen maçlar; günlük artımlı backfill | Non-commercial anahtarsız API; kaynak atfı gerekir |

## Doğrulanan ancak tahmine bağlanmayan kaynaklar

- **SofaScore:** Resmî spor verisi API'si sunmuyor. Özel web endpoint'leri lisans
  ve süreklilik garantisi taşımadığı için otomatik scraping yapılmıyor.
- **SportScore:** Ücretsiz ve atıflı API mevcut; canlı doğrulamada takım slug'ları
  yaş kategorilerine yanlış eşleşti ve bazı maç durum/tarihleri tutarsızdı. Bu
  nedenle tahmin girdisine alınmadı.

## xG doluluk sonucu

2 Ağustos 2026 senkronizasyonunda Understat'tan 1.991 adet 2025 sezonu maçı xG ile
eşleştirildi. Kalite kapısını geçen türetilmiş şut modeli, gözlemlenmiş xG olmayan
6.627 tarihsel kaydı tamamladı. Türetilmiş model holdout MAE değeri `0.481604`,
baseline MAE değeri `0.711590` oldu.
