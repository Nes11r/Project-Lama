# Qaz sərfiyyatı proqnozu və operator qərar dəstəyi (demo)

ROC flow computer-lərindən (Mərdəkan, Sumqayıt Şəhər-4, Türkan) toplanmış saatlıq məlumat əsasında
**növbəti 24 saatın saatlıq qaz sərfiyyatını** qeyri-müəyyənlik intervalı ilə proqnozlaşdırır. Operatora
xəbərdarlıqlar və izahlı tövsiyələr verir.

> 🛡️ **Təhlükəsizlik prinsipi:** sistem yalnız **tövsiyə rejimində** işləyir. ROC və SCADA-ya heç bir
> yazma və ya idarəetmə əmri göndərilmir. `/api/v1/control/*` yalnız gələcək avtomatika üçün təsvir olunmuş
> və **deaktiv** edilmiş interfeysdir (HTTP 403). Son qərar həmişə operatordadır.

## Sürətli başlanğıc

### Docker
```bash
cp .env.example .env          # istəyə bağlı
docker compose up --build
```
- Operator paneli: http://localhost:8501
- API və sənədləşmə: http://localhost:8000/docs

İlk işə salmada `storage/models/LATEST` yoxdursa konteyner aşağıdakıları ardıcıl edir (~30 dəq):
1. datanı təmizləyir;
2. hava məlumatını çəkir;
3. backtest edir;
4. modeli öyrədir.

`storage/` qovluğu volume kimi qoşulduğundan bu, yalnız bir dəfə baş verir. `SKIP_EVAL=1` backtest-i ötürür.

### Lokal (Python 3.11)
```bash
pip install -r requirements.txt
python -m data.pipeline                 # 1. təmizləmə → storage/processed/hourly.parquet
python -m external.weather 2024-08-31   # 2. hava (Open-Meteo, keşlənir)
python -m external.osm                  # 3. əhali + OSM (statik, keşlənir)
python -m models.evaluate               # 4. walk-forward backtest, ablation, SHAP → reports/model/
python -m models.train --skip-data      # 5. yekun model → storage/models/<versiya>/
uvicorn api.main:app --port 8000        # 6. API
streamlit run frontend/app.py           # 7. panel (API_URL=http://localhost:8000)
python -m pytest tests                  # testlər
```

## Modeli yenidən öyrətmə
```bash
python -m models.evaluate        # (tövsiyə olunur) backtest metrikləri və interval kalibrasiyası yenilənir
python -m models.train           # pipeline + hava yeniləməsi + təlim + yeni versiya
python -m models.train --offline # API çağırmadan, yalnız keşlə
```
- Hər təlim `storage/models/vYYYYMMDD-HHMMSS/` qovluğunda yeni versiya yaradır: kvantil modellər, klaster modeli və `meta.json`. `meta.json`-da təlim dövrü, xüsusiyyətlər, HDD bazası, interval kalibrasiyası və CV metrikləri saxlanılır.
- `LATEST` faylı aktiv versiyanı göstərir.
- Geri qaytarmaq üçün `LATEST`-ə köhnə versiyanın adını yazın.
- İşləyən API yeni modeli yalnız yenidən başladıldıqda və ya data yükləndikdə oxuyur.

## Yeni data
- **Panel:** Tənzimləmələr → "Yeni data yüklə". ROC CSV ixracı qəbul olunur, `Cəmi` sətirləri də daxil.
- **Qovluq:** faylı `dataset/incoming/` qovluğuna atın. API hər dəqiqə yoxlayır və pipeline-ı avtomatik işə salır.
- Fayl adı `config.yaml → stations.*.file_patterns` ilə məntəqəyə bağlanır (məs. `turkan_2024-09.csv`).

## Rejimlər (`config.yaml → demo.mode`)
| Rejim | "İndi" | Hava |
|---|---|---|
| `replay` (default) | Paneldən idarə olunan demo saatı | İndiyə qədər faktiki, sonrası **1 gün əvvəl verilmiş real proqnoz** (Previous Runs API). Sızma yoxdur |
| `live` | Sistem saatı | Open-Meteo Forecast API (əlçatmazsa keş + paneldə xəbərdarlıq) |

## Arxitektura

```
 ROC CSV ixracları ──► data/loader ──► data/cleaning ──► storage/processed/hourly.parquet
 (dataset/, incoming/)      │  "Cəmi" ilə itən saatların bərpası, sensor xətaları, hadisələr
                            ▼
 Open-Meteo ─► external/weather ─┐
 holidays+YAML ─► external/calendar ─┤──► features/build (direct multi-horizon, sızmasız)
 WorldPop/OSM ─► external/population,osm ┘          │
                                                    ▼
                         models/lgbm (kvantil LightGBM q10/q50/q90, qlobal: 3 məntəqə)
                         models/evaluate (walk-forward, baseline, ablation, SHAP)
                         models/train + registry (versiyalama)
                                                    │
                                                    ▼
                  models/predict (ForecastService: proqnoz + SHAP izahı + hadisə korreksiyası)
                  decision/alerts, decision/recommendations (yalnız tövsiyə)
                                                    │
                                     api/main.py (FastAPI, REST)
                                                    │
                                     frontend/app.py (Streamlit, Azərbaycan dili)
```

| Qovluq | Məzmun |
|---|---|
| `core/` | konfiqurasiya, yollar |
| `data/` | ROC CSV oxuyucu, təmizləmə, pipeline |
| `external/` | hava, təqvim, əhali, OSM; keş və mənbə statusu |
| `features/` | hava törəmələri, profil klasterləri, xüsusiyyət cədvəli |
| `models/` | metriklər, baseline-lar, LightGBM, qiymətləndirmə, təlim, reyestr, proqnoz servisi |
| `decision/` | xəbərdarlıqlar, tövsiyələr |
| `api/` | FastAPI |
| `frontend/` | Streamlit paneli |
| `tests/` | unit testlər (sızma testi daxil) |
| `eda/` | Mərhələ 1 kəşfiyyat analizi |
| `config.yaml`, `config/` | limitlər, yollar, üfüq, təqvim düzəlişləri, hadisələr |
| `docs/` | data pipeline sənədi |

### Frontend seçimi
**Streamlit** prototip üçün seçilib: tək dil (Python), sürətli iterasiya və analitiklərin özləri dəyişiklik edə bilməsi. **İstehsalat üçün React** (+ TypeScript, WebSocket) tövsiyə olunur:
- çoxistifadəçili sessiyalar və rol əsaslı giriş;
- server-push ilə real-vaxt yenilənmə (Streamlit hər qarşılıqlı əlaqədə skripti yenidən işlədir);
- SCADA HMI standartlarına uyğun incə UI;
- oflayn və ekran divarı rejimləri.

Bütün məntiq API-dədir, ona görə keçid yalnız frontend-in əvəzlənməsi deməkdir.

## Əsas API nöqtələri
| Metod | Yol | Təsvir |
|---|---|---|
| GET | `/api/v1/dashboard/{station}` | cari vəziyyət + 24 saat proqnoz + xəbərdarlıq + tövsiyə |
| GET | `/api/v1/forecast/{station}?origin=` | proqnoz (q10/q50/q90) + izah |
| GET | `/api/v1/history/{station}?start&end` | faktiki + 1 gün əvvəlki proqnozlar |
| GET | `/api/v1/accuracy/{station}` | backtest metrikləri |
| GET/POST | `/api/v1/alerts`, `/alerts/{id}/ack` | xəbərdarlıqlar, təsdiq |
| GET/PUT | `/api/v1/settings/limits[/{station}]` | limitlər |
| POST | `/api/v1/upload` | yeni ROC CSV |
| GET/PUT | `/api/v1/clock` | demo saatı |
| GET | `/api/v1/control/spec` | gələcək avtomatika interfeysi (deaktiv) |

## Fərziyyələr və məhdudiyyətlər
- **Koordinatlar təxminidir** (qəsəbə və şəhər mərkəzi), xidmət əraziləri radius ilə modelləşdirilib.
- Vahid "min m³" min kubmetr kimi qəbul edilib. Standart şəraitə gətirilməsi təsdiqlənməyib. Təzyiq **mütləqdir**: boşaldılmış xətt ≈ 100 kPa göstərir.
- Limitlər (`config.yaml`) tarixi p99/maksimumdan götürülüb. **Operator dəqiqləşdirməlidir.**
- Türkanda 2024-07-09 10:00-dan planlı təmir başlayıb (`config/events.yaml`). Bu dövr təlimdən çıxarılıb, canlı proqnoz isə faktiki səviyyəyə avtomatik uyğunlaşdırılır.
- 1 gün əvvəlki hava proqnozu temperatur üçün 2021-03-dən, digər dəyişənlər üçün 2024-01-dən mövcuddur. Ətraflı: `reports/model/report.md`.
- `config/calendar_overrides.yaml`-da 2018–2019 köçürmələri, məktəb tətilləri və karantin tarixləri **yoxlanmamışdır**.
