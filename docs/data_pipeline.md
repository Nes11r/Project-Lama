# Data pipeline: çevrilmələrin sənədləşdirilməsi

Kod: [data/loader.py](../data/loader.py), [data/cleaning.py](../data/cleaning.py), [data/pipeline.py](../data/pipeline.py).
Parametrlər: `config.yaml → cleaning`. Hər işə salmada sayğaclar `reports/cleaning_report.json`-a yazılır.

## 1. Oxuma (`loader.py`)

| # | Çevrilmə | Səbəb |
|---|---|---|
| 1.1 | `TARİX` → `ts_end`; `ts = ts_end − 1 saat` (interval başlanğıcı) | ROC damğası saat-sonudur: `01:00` = 00:00–01:00. Daxildə interval başlanğıcı standartdır, hava/təqvim ilə birləşmə sadələşir |
| 1.2 | `Cəmi: X` sətirləri ayrıca `totals` cədvəlinə (`gas_day`, `total`) çıxarılır | Fayl gündəlik hesabatların birləşməsidir; `Cəmi` ROC-un gündəlik akkumulyatorudur |
| 1.3 | `SƏRF (min m3)` sütunu atılır | Həmişə `saatlıq × 24`-ə bərabərdir (törəmə sütun) |
| 1.4 | Bir məntəqənin bütün faylları (`dataset/` + `dataset/incoming/`) birləşdirilir | Yeni data əlavə olunduqda pipeline yenidən işə salınır; sonrakı fayl üstünlük alır |

## 2. Təmizləmə (`cleaning.py`), ardıcıllıqla

| # | Addım | Qayda | Nəticə |
|---|---|---|---|
| 2.1 | Natamam saat damğaları | dəqiqəsi ≠ 0 (məs. `17:58`) | atılır (logger yenidən başlaması) |
| 2.2 | Təkrarlar | eyni `ts`: sıfır olmayan sərf/təzyiq üstün, sonra sonrakı fayl | 1 təkrar saxlanır |
| 2.3 | Saatlıq şəbəkə | min→max tam saatlar | çatışmayan saat → `f_missing_raw` |
| 2.4 | Fiziki hüdudlar | `valid_ranges` (SG 0.70–0.80, T −15…60 °C, P 150–900 kPa, ΔP ≤ 70) | kənar → NaN, `f_bad_<sütun>` |
| 2.5 | Xəttin boşaldılması | mütləq təzyiq < 150 kPa | `f_depressurized` (təlimdən çıxarılır) |
| 2.6 | ΔP saturasiyası | ΔP ≥ 68.5 kPa | `f_dp_saturated` (yalnız bayraq; operator məlumatına görə düzəliş edilib) |
| 2.7 | Donmuş sensor | ≥ 6 saat eyni sərf | təzyiq normaldırsa → NaN (`f_flatline`); sıfır + boşaldılma → `f_outage` |
| 2.8 | Sıçrayış/çöküş | eyni saatın ±7 günlük median/MAD-ı üzrə \|z\| > 6 **və** dəyər medianın < 0.5× və ya > 3× | ≤ 6 saat → NaN (`f_spike`); daha uzun → saxlanır, təlimdən çıxarılır (`f_anomaly_long`, adətən qismən təchizat kəsintisi) |
| 2.9 | **İtən saatların bərpası** | gün üçün `Cəmi` var və ≤ 6 saat itibsə: `itən_cəm = Cəmi − Σ(mövcud)`, interpolasiya formasına uyğun paylanır; interpolasiyaya nisbət 0.2–3 aralığında olmalıdır | `f_reconstructed`. Yoxlama: Σ23 + interpolasiya / Cəmi ≈ 1.000 |
| 2.10 | Hadisələr | `config/events.yaml` | `f_event`; `exclude_from_training` olduqda təlim/qiymətləndirmədən çıxarılır |
| 2.11 | Xüsusiyyətlər üçün doldurma | ≤ 6 saat xətti interpolasiya; uzun boşluq → əvvəlki gün/həftənin eyni saatı | `q_filled`, `f_interpolated`, `f_profile_filled` |
| 2.12 | Təlim yararlılığı | `q_clean` mövcuddur və boşaldılma/kəsinti/uzun anomaliya/xaric edilmiş hadisə yoxdur | `train_ok` |

**Hədəf dəyişən**: `q_clean` (orijinal + `Cəmi` ilə bərpa olunanlar). İnterpolasiya edilmiş dəyərlər heç vaxt hədəf olmur.

**Kiçik sızma qeydi**: `q_filled` interpolasiyası və 2.8-in mərkəzləşdirilmiş pəncərələri tarixi emalda gələcək nöqtələrdən istifadə edir. Bu yalnız ≈2% boşluq saatına və keyfiyyət bayraqlarına təsir edir. Hədəf dəyərlərə təsir etmir.

## 3. Xarici məlumatlar

| Modul | Mənbə | Lisenziya | Qeyd |
|---|---|---|---|
| `external/weather.py` | Open-Meteo Archive (faktiki), Previous Runs (1 gün əvvəlki proqnoz), Forecast (canlı) | CC BY 4.0 | Previous Runs: temperatur 2021-03-dən, digər dəyişənlər 2024-01-dən |
| `external/calendar.py` | `holidays` (AZ) + `config/calendar_overrides.yaml` + `astral` | MIT / Apache | Nager.Date AZ-ni dəstəkləmir |
| `external/population.py` | WorldPop stats API (2020) | CC BY 4.0 | Radius əsaslı (xidmət ərazisi məlum deyil) |
| `external/osm.py` | OSM Overpass | ODbL | Binaların 61–95%-i tipsizdir (`building=yes`) |

Vaxt düzülüşü: Open-Meteo dəyərləri HH:00 anına aiddir. Ani dəyişənlər üçün interval `[t, t+1)` dəyəri `(v(t)+v(t+1))/2`, radiasiya və yağıntı üçün `v(t+1)` götürülür.
Keş: `storage/cache/` (Parquet/JSON). API əlçatmaz olduqda keş istifadə olunur və status `storage/cache/external_status.json`-a yazılır (operator panelində göstərilir).
