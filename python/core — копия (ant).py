import bisect
import json
import math
from pathlib import Path

def get_all_spectrum_peaks_dynamic(
    file_paths: list[str], min_db_threshold: float
) -> tuple[set[int], list[dict]]:
    """Динамически ищет доминирующие пики по всему спектру частот."""
    detected_frequencies = set()
    all_found_peaks_details = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        file_peaks = []
        with open(path, "r", encoding="utf-8") as file:
            next(file, None)  # Пропускаем заголовок Audacity
            for line in file:
                if not line.strip():
                    continue
                try:
                    freq_str, level_str = line.split()
                    freq = float(freq_str)
                    level = float(level_str)

                    if level >= min_db_threshold:
                        file_peaks.append({
                            "freq": int(round(freq)),
                            "level": round(level, 1),
                            "source": path.name,
                        })
                except ValueError:
                    continue

        file_peaks.sort(key=lambda x: x["level"], reverse=True)
        for peak in file_peaks:
            detected_frequencies.add(peak["freq"])
            all_found_peaks_details.append(peak)

    all_found_peaks_details.sort(key=lambda x: x["level"], reverse=True)
    return detected_frequencies, all_found_peaks_details


FREQ_MIN_HZ = 20
FREQ_MAX_HZ = 20000
MAX_EXTRAPOLATION_DB_PER_OCT = 24.0


def _clean_pairs(pairs) -> dict[float, float]:
    """Приводит любые пары (частота, дБ) к отсортированному словарю без дублей."""
    acc: dict[float, list[float]] = {}
    for pair in pairs:
        try:
            freq, level = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if freq > 0 and math.isfinite(freq) and math.isfinite(level):
            acc.setdefault(freq, []).append(level)
    return {f: sum(v) / len(v) for f, v in sorted(acc.items())}


def load_speaker_response(source, profile_name: str | None = None) -> dict[float, float]:
    """Универсальная загрузка АЧХ устройства.

    source: dict {Гц: дБ} | список пар [[Гц, дБ], ...] |
            путь к .json (формат {"profiles": {имя: пары}} или просто пары/словарь) |
            путь к текстовому файлу (.txt/.csv/.frd: «частота дБ» в строке, заголовки пропускаются).
    """
    if source is None:
        return {}
    if isinstance(source, dict):
        return _clean_pairs(source.items())
    if isinstance(source, (list, tuple)):
        return _clean_pairs(source)

    path = Path(source)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "profiles" in data:
            data = data["profiles"]
        if isinstance(data, dict) and data and all(isinstance(v, (list, tuple)) for v in data.values()):
            if profile_name is None:
                if len(data) != 1:
                    raise ValueError("В JSON несколько профилей — укажите profile_name")
                profile_name = next(iter(data))
            data = data[profile_name]
        if isinstance(data, dict):
            return _clean_pairs(data.items())
        return _clean_pairs(data)

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) >= 2:
                pairs.append(parts[:2])
    return _clean_pairs(pairs)


class SpeakerResponse:
    """АЧХ устройства: интерполяция по лог-шкале частот, экстраполяция за краями измерения,
    пороги заданы относительно опорного уровня (максимума АЧХ), а не абсолютных значений."""

    def __init__(self, data: dict | None):
        data = _clean_pairs((data or {}).items())
        self.freqs = list(data.keys())
        self.levels = list(data.values())
        self.empty = not self.freqs
        self.reference = max(self.levels) if self.levels else 0.0

    def at(self, freq: float) -> float:
        """Уровень АЧХ (дБ) на частоте freq."""
        if self.empty:
            return 0.0
        f, v = self.freqs, self.levels
        freq = max(freq, 1e-3)
        if len(f) == 1:
            return v[0]
        if freq <= f[0]:
            slope = (v[1] - v[0]) / math.log2(f[1] / f[0])
            slope = min(MAX_EXTRAPOLATION_DB_PER_OCT, max(0.0, slope))
            return v[0] - slope * math.log2(f[0] / freq)
        if freq >= f[-1]:
            slope = (v[-2] - v[-1]) / math.log2(f[-1] / f[-2])
            slope = min(MAX_EXTRAPOLATION_DB_PER_OCT, max(0.0, slope))
            return v[-1] - slope * math.log2(freq / f[-1])
        i = bisect.bisect_right(f, freq) - 1
        t = math.log2(freq / f[i]) / math.log2(f[i + 1] / f[i])
        return v[i] + (v[i + 1] - v[i]) * t

    def relative(self, freq: float) -> float:
        """Уровень относительно опорного (0 дБ = максимум АЧХ)."""
        return self.at(freq) - self.reference

    def usable_band(self, cutoff_db: float, f_lo: int, f_hi: int) -> tuple[int, int]:
        """Границы рабочего диапазона: где АЧХ не ниже опорного уровня + cutoff_db."""
        if self.empty:
            return f_lo, f_hi
        threshold = self.reference + cutoff_db
        good = [f for f in range(f_lo, f_hi + 1) if self.at(f) >= threshold]
        if not good:
            return f_lo, f_hi
        return good[0], good[-1]


def interpolate_speaker_response(speaker_data: dict, target_freq: float) -> float:
    """Совместимость: уровень АЧХ на частоте target_freq."""
    return SpeakerResponse(speaker_data).at(target_freq)


def build_normalized_preset_dynamic(
    profile_path: str | None,
    spectrum_paths: list[str],
    max_peak_limit: float,
    min_db_threshold: float,
    min_gain: float,
    masking_margin: float,
    filter_length: int,
    h2_gain: float,
    h3_gain: float,
    speaker_response=None,
    profile_name: str | None = None,
    cutoff_db: float = -12.0,
    infra_db: float = -8.0,
    fundamental_gain: float = 6.0,
    freq_min: int = FREQ_MIN_HZ,
    freq_max: int = FREQ_MAX_HZ,
):
    """Строит психоакустический купол с вершиной точно в max_peak_limit.

    АЧХ устройства — любая: speaker_response (dict / пары / путь к файлу) либо
    profile_path (+ profile_name). Пороги относительные (от максимума АЧХ):
      cutoff_db  — ниже этого уровня устройство считается не воспроизводящим (границы диапазона);
      infra_db   — пики, лежащие ниже этого уровня, воспроизводятся через гармоники.
    """
    base_peaks, peaks_details = get_all_spectrum_peaks_dynamic(
        spectrum_paths, min_db_threshold
    )

    response = SpeakerResponse(
        load_speaker_response(
            speaker_response if speaker_response is not None else profile_path,
            profile_name,
        )
    )

    # Рабочий диапазон устройства определяется по его АЧХ, а не зашит в код
    f_min, f_max = response.usable_band(cutoff_db, freq_min, freq_max)

    dense_frequencies = set()
    if base_peaks:
        min_peak = min(base_peaks)
        max_peak = max(base_peaks)
        start_dome = max(freq_min, min_peak - 10)
        end_dome = min(freq_max, (max_peak * 3) + 30)
        for freq in range(start_dome, end_dome + 1, 1):
            dense_frequencies.add(freq)
    else:
        for freq in range(f_min, min(f_max, f_min + 200) + 1, 1):
            dense_frequencies.add(freq)

    # Предрасчёт по пикам: не пересчитываем АЧХ в цикле по частотам
    peak_infra = {p: response.relative(p) < infra_db for p in base_peaks}

    def octave_dist(f_center, f_target):
        if f_center <= 0 or f_target <= 0:
            return float("inf")
        return abs(math.log2(f_target / f_center))

    # Ширина купола: ~0.5 октавы в каждую сторону от центра
    half_width = 0.5

    def bump(od, gain):
        return max(0.0, gain * (1.0 + math.cos(min(od, half_width) / half_width * math.pi)) / 2.0)

    raw_points = {}
    base_zone_gains = []

    # ШАГ 1: Расчёт идеальной, несжатой психоакустической формы
    for freq in sorted(dense_frequencies):
        if freq < f_min or freq > f_max:
            raw_points[freq] = min_gain
            continue

        speaker_efficiency = response.at(freq)
        center_bonus = 0.0
        is_pure_base = False

        for p in base_peaks:
            od_f1 = octave_dist(p, freq)
            od_f2 = octave_dist(p * 2, freq)
            od_f3 = octave_dist(p * 3, freq)

            if not peak_infra[p]:
                b1 = bump(od_f1, fundamental_gain)
                if b1 > 0:
                    is_pure_base = True
                b2 = bump(od_f2, h2_gain)
                center_bonus = max(center_bonus, b1 + b2)
            else:
                b2 = bump(od_f2, h2_gain)
                b3 = bump(od_f3, h3_gain)
                center_bonus = max(center_bonus, b2 + b3)

        calculated_gain = (0.0 - speaker_efficiency) + masking_margin + center_bonus
        raw_points[freq] = calculated_gain
        if is_pure_base or (not base_peaks and f_min <= freq <= f_min + 30):
            base_zone_gains.append(calculated_gain)

    # ШАГ 2: Расчёт величины глобального сдвига для идеальной посадки пика
    # Находим абсолютный максимум исходной формы купола
    unlimited_max = max(base_zone_gains) if base_zone_gains else max(raw_points.values()) if raw_points else 0.0
    
    # Величина коррекции: сдвигаем всю форму целиком, сохраняя её геометрию
    global_shift = unlimited_max - max_peak_limit

    # ШАГ 3: Формирование финального массива точек
    preset_points = []
    active_gains = {}
    for freq, raw_gain in raw_points.items():
        if freq < f_min or freq > f_max:
            final_gain = min_gain
        else:
            final_gain = raw_gain - global_shift
            final_gain = max(min_gain, final_gain)  # Ограничение снизу (пол купола)
            
        preset_points.append((freq, round(final_gain, 1)))
        if final_gain > min_gain:
            active_gains[freq] = final_gain

    first_active_freq = min(active_gains.keys()) if active_gains else f_min
    first_active_gain = active_gains[first_active_freq] if active_gains else min_gain
    last_active_freq = max(active_gains.keys()) if active_gains else 250
    last_active_gain = active_gains[last_active_freq] if active_gains else min_gain

    # Убираем «полочные» точки пола за пределами активной зоны: иначе они
    # перемешиваются со сглаживающими рампами и дают гребёнку провалов
    preset_points = [
        (f, g) for f, g in preset_points
        if first_active_freq <= f <= last_active_freq
    ]

    # Сглаживание спадов к краям спектра
    preset_points.append((freq_min, min_gain))
    if first_active_freq > freq_min and first_active_gain > min_gain:
        log_start_l = math.log10(freq_min)
        log_end_l = math.log10(first_active_freq)
        steps_l = 30
        for i in range(1, steps_l):
            ratio = i / steps_l
            log_f = log_start_l + ratio * (log_end_l - log_start_l)
            f = int(round(10**log_f))
            if freq_min < f < first_active_freq:
                smooth_factor = (1.0 - math.cos(ratio * math.pi)) / 2.0
                f_gain = min_gain + smooth_factor * (first_active_gain - min_gain)
                preset_points.append((f, round(f_gain, 1)))

    start_r_freq = last_active_freq + 1
    if start_r_freq < freq_max and last_active_gain > min_gain:
        log_start_r = math.log10(start_r_freq)
        log_end_r = math.log10(freq_max)
        steps_r = 100
        for i in range(0, steps_r + 1):
            ratio = i / steps_r
            log_f = log_start_r + ratio * (log_end_r - log_start_r)
            f = int(round(10**log_f))
            if f > last_active_freq and f <= freq_max:
                smooth_factor = (1.0 - math.cos(ratio * math.pi)) / 2.0
                f_gain = last_active_gain - smooth_factor * (last_active_gain - min_gain)
                preset_points.append((f, round(f_gain, 1)))
    else:
        preset_points.append((freq_max, min_gain))

    unique_points = {}
    for freq, gain in preset_points:
        if freq not in unique_points or gain > unique_points[freq]:
            unique_points[freq] = gain

    preset_points = sorted(list(unique_points.items()), key=lambda x: x[0])
    return preset_points, sorted(list(base_peaks)), global_shift, filter_length, peaks_details
