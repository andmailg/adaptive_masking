"""Психоакустический купол: маскировка конкретного шума на конкретном устройстве.

Алгоритм:
  1. Спектры шума -> уровни в полосах ERB (огибающая «худшего случая» по всем файлам).
  2. АЧХ устройства -> рабочий диапазон и компенсация.
  3. Воспроизводимая часть шума маскируется кривой, повторяющей уровень шума в полосе ERB.
  4. Для невоспроизводимых пиков строятся купола 2-й и 3-й гармоник (виртуальный бас):
     уровень купола = уровень шума в полосе пика + усиление гармоники, склоны
     несимметричные (функция распространения маскирования), ширина — по шкале ERB.
  5. Учитывается спектр исходного маскирующего сигнала (по умолчанию коричневый шум
     с полюсом 0.995, как в генераторе интерфейса).
"""
import bisect
import json
import math
from pathlib import Path

import numpy as np

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

    def measured_band(self, f_lo: int, f_hi: int) -> tuple[int, int]:
        """Границы диапазона = первая и последняя измеренные точки АЧХ (как заданы в профиле)."""
        if self.empty:
            return f_lo, f_hi
        lo = max(f_lo, int(math.ceil(self.freqs[0])))
        hi = min(f_hi, int(math.floor(self.freqs[-1])))
        return (lo, hi) if lo < hi else (f_lo, f_hi)

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



# --------------------------------------------------------------------------------------
# Психоакустика
# --------------------------------------------------------------------------------------
ERB_TO_BARK_SCALE = 0.63      # перевод шкалы ERB-rate в «барк-подобные» единицы для SF
PEAK_PROMINENCE_DB = 4.0      # минимальная выделенность пика над окружением
PEAK_BAND_OCT = 1.0 / 6.0     # полоса сглаживания при поиске пиков
GRID_ERB_STEP = 0.1           # шаг сетки по шкале ERB-number
DOME_SLOPE_DOWN = 8.0         # крутизна нижнего склона купола относительно SF (больше — круче)
DOME_SLOPE_UP = 3.0           # крутизна верхнего склона: маскирование вверх распространяется шире
LOW_EDGE_TAPER_OCT = 0.3      # плавный спад ниже нижней границы устройства


def _erb_number(f):
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, dtype=float))


def _erb_number_inv(e):
    return (10.0 ** (np.asarray(e, dtype=float) / 21.4) - 1.0) / 0.00437


def _erb_bandwidth(f):
    return 24.7 * (4.37 * np.asarray(f, dtype=float) / 1000.0 + 1.0)


def _critical_rate(f):
    return _erb_number(f) * ERB_TO_BARK_SCALE


def _spreading_db(dz):
    """Функция распространения маскирования (Schroeder), дБ.
    dz = z(маскируемого) - z(маскера): dz > 0 — маскируемый выше маскера (спад пологий),
    dz < 0 — ниже (спад крутой). SF(0) = 0 дБ."""
    u = np.asarray(dz, dtype=float) + 0.474
    return 15.81 + 7.5 * u - 17.5 * np.sqrt(1.0 + u * u)


def source_spectrum_db(freqs, pole: float | None = 0.995, sample_rate: float = 44100.0):
    """Спектр исходного маскирующего сигнала (дБ, относительный).
    pole=0.995 — коричневый шум генератора (y = 0.995*y + x); None — плоский (белый)."""
    freqs = np.asarray(freqs, dtype=float)
    if pole is None:
        return np.zeros_like(freqs)
    w = 2.0 * np.pi * freqs / sample_rate
    mag = np.abs(1.0 - pole * np.exp(-1j * w))
    return -20.0 * np.log10(mag)


# --------------------------------------------------------------------------------------
# Анализ шума
# --------------------------------------------------------------------------------------
def _load_noise_file(path: Path):
    freqs, levels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        next(fh, None)  # заголовок Audacity
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                f, lv = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if f > 0 and math.isfinite(lv):
                freqs.append(f)
                levels.append(lv)
    return np.array(freqs), np.array(levels)


def _band_levels_db(freqs, levels_db, centers, widths):
    """Суммарный уровень (дБ) в полосах [c - w/2, c + w/2]."""
    if len(freqs) < 2:
        return np.full(len(centers), -200.0)
    power = 10.0 ** (levels_db / 10.0)
    cum = np.concatenate(([0.0], np.cumsum(power)))
    edges_x = np.concatenate(([freqs[0] - (freqs[1] - freqs[0]) / 2.0],
                              (freqs[:-1] + freqs[1:]) / 2.0,
                              [freqs[-1] + (freqs[-1] - freqs[-2]) / 2.0]))
    lo = np.interp(centers - widths / 2.0, edges_x, cum)
    hi = np.interp(centers + widths / 2.0, edges_x, cum)
    total = np.maximum(hi - lo, 1e-30)
    return 10.0 * np.log10(total)


def _find_peaks(values, prominence):
    """Индексы локальных максимумов с заданной выделенностью (дБ)."""
    n = len(values)
    peaks = []
    for i in range(1, n - 1):
        v = values[i]
        if v < values[i - 1] or v <= values[i + 1]:
            continue
        left_base, j = v, i - 1
        while j >= 0 and values[j] <= v:
            left_base = min(left_base, values[j])
            j -= 1
        right_base, j = v, i + 1
        while j < n and values[j] <= v:
            right_base = min(right_base, values[j])
            j += 1
        if v - max(left_base, right_base) >= prominence:
            peaks.append(i)
    return peaks


def _window_max_db(freqs, levels_db, centers, widths):
    """Максимальный по бинам уровень (дБ) внутри полос [c - w/2, c + w/2]."""
    lo = np.searchsorted(freqs, centers - widths / 2.0, side="left")
    hi = np.searchsorted(freqs, centers + widths / 2.0, side="right")
    out = np.full(len(centers), -200.0)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = levels_db[a:b].max()
    return out


def analyze_noise(file_paths, min_db_threshold, grid_freqs, freq_min, freq_max, combine="max"):
    """Анализ спектров шума.

    Возвращает (band_db, peak_db, base_peaks, peaks_details):
      band_db  — суммарный уровень шума в полосе ERB на сетке (дБ), «худший случай» по файлам;
      peak_db  — максимальный бинный уровень в той же полосе (для сравнения с порогом);
      base_peaks — отсортированные частоты пиков (целые Гц); peaks_details — freq/level/source.
    """
    grid_freqs = np.asarray(grid_freqs, dtype=float)
    widths = _erb_bandwidth(grid_freqs)

    octaves = math.log2(freq_max / freq_min)
    fine_n = max(8, int(24 * octaves))
    fine_centers = freq_min * 2.0 ** (np.arange(fine_n + 1) * (octaves / fine_n))
    fine_widths = fine_centers * (2.0 ** (PEAK_BAND_OCT / 2.0) - 2.0 ** (-PEAK_BAND_OCT / 2.0))

    bands, peakmax, fine_sum, fine_peak, files = [], [], [], [], []
    peaks_details = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue
        freqs, levels = _load_noise_file(path)
        if len(freqs) < 2:
            continue
        order = np.argsort(freqs)
        freqs, levels = freqs[order], levels[order]

        bands.append(_band_levels_db(freqs, levels, grid_freqs, widths))
        peakmax.append(_window_max_db(freqs, levels, grid_freqs, widths))
        fs = _band_levels_db(freqs, levels, fine_centers, fine_widths)
        fp = _window_max_db(freqs, levels, fine_centers, fine_widths)
        fine_sum.append(fs)
        fine_peak.append(fp)
        files.append((path.name, freqs, levels))

        for idx in _find_peaks(fs, PEAK_PROMINENCE_DB):
            if fp[idx] < min_db_threshold:
                continue
            lo, hi = fine_centers[idx] - fine_widths[idx] / 2, fine_centers[idx] + fine_widths[idx] / 2
            m = (freqs >= lo) & (freqs <= hi)
            k = int(np.argmax(levels[m]))
            peaks_details.append({
                "freq": int(round(freqs[m][k])),
                "level": round(float(levels[m][k]), 1),
                "source": path.name,
            })

    if not bands:
        return None, None, [], []

    stack = np.vstack(bands)
    if combine == "mean":
        band_db = 10.0 * np.log10(np.mean(10.0 ** (stack / 10.0), axis=0))
    else:
        band_db = stack.max(axis=0)
    peak_db = np.vstack(peakmax).max(axis=0)

    # Пики для гармоник — по огибающей «худшего случая»
    fine_env = np.vstack(fine_sum).max(axis=0)
    fine_peak_env = np.vstack(fine_peak).max(axis=0)
    base_peaks = set()
    for idx in _find_peaks(fine_env, PEAK_PROMINENCE_DB):
        if fine_peak_env[idx] < min_db_threshold:
            continue
        lo, hi = fine_centers[idx] - fine_widths[idx] / 2, fine_centers[idx] + fine_widths[idx] / 2
        best_f, best_l = None, -1e9
        for _, freqs, levels in files:
            m = (freqs >= lo) & (freqs <= hi)
            if m.any():
                k = int(np.argmax(levels[m]))
                if levels[m][k] > best_l:
                    best_l, best_f = levels[m][k], freqs[m][k]
        if best_f is not None:
            base_peaks.add(int(round(best_f)))

    peaks_details.sort(key=lambda p: p["level"], reverse=True)
    return band_db, peak_db, sorted(base_peaks), peaks_details


def get_all_spectrum_peaks_dynamic(file_paths, min_db_threshold):
    """Совместимость: (множество частот пиков, подробности пиков)."""
    grid = _erb_number_inv(np.arange(_erb_number(FREQ_MIN_HZ), _erb_number(FREQ_MAX_HZ), GRID_ERB_STEP))
    _, _, peaks, details = analyze_noise(file_paths, min_db_threshold, grid, FREQ_MIN_HZ, FREQ_MAX_HZ)
    return set(peaks), details


# --------------------------------------------------------------------------------------
# Решение задачи маскировки
# --------------------------------------------------------------------------------------
def build_normalized_preset_dynamic(
    profile_path: str | None,
    spectrum_paths: list[str],
    max_peak_limit: float,
    min_db_threshold: float,
    min_gain: float,
    filter_length: int,
    h2_gain: float,
    h3_gain: float,
    speaker_response=None,
    profile_name: str | None = None,
    freq_low: float | None = None,
    infra_db: float = -8.0,
    freq_min: int = FREQ_MIN_HZ,
    freq_max: int = FREQ_MAX_HZ,
    source_pole: float | None = 0.995,
    noise_combine: str = "max",
    dome_slope_down: float = DOME_SLOPE_DOWN,
    dome_slope_up: float = DOME_SLOPE_UP,
):
    """Строит маскирующую кривую фильтра под заданный шум и устройство.

    Параметры АЧХ: speaker_response (dict / пары / путь) либо profile_path (+profile_name).
    min_db_threshold — шум в полосе ниже этого уровня маскировать не нужно.
    h2_gain, h3_gain — высота куполов 2-й и 3-й гармоник (дБ); масштабируется уровнем шума пика.
    freq_low — нижняя граница воспроизведения, Гц (по умолчанию — первая точка АЧХ из профиля);
    infra_db — относительный (от максимума АЧХ) порог «инфра» пиков.
    source_pole — полюс исходного шума (0.995 — коричневый, None — белый).
    noise_combine — объединение файлов: "max" (худший случай) или "mean".
    dome_slope_down / dome_slope_up — крутизна нижнего / верхнего склона куполов гармоник.
    """
    response = SpeakerResponse(
        load_speaker_response(
            speaker_response if speaker_response is not None else profile_path,
            profile_name,
        )
    )
    # Рабочий диапазон берётся из самого профиля (первая/последняя точка АЧХ);
    # нижнюю границу можно переопределить через freq_low
    f_min, f_max = response.measured_band(freq_min, freq_max)
    if freq_low is not None:
        f_min = min(max(freq_min, int(round(freq_low))), f_max - 1)

    # Сетка по шкале ERB
    grid = _erb_number_inv(np.arange(_erb_number(freq_min), _erb_number(freq_max) + 1e-9, GRID_ERB_STEP))
    z = _critical_rate(grid)

    band_db, peak_db, base_peaks, peaks_details = analyze_noise(
        spectrum_paths, min_db_threshold, grid, freq_min, freq_max, noise_combine
    )

    reproducible = (grid >= f_min) & (grid <= f_max)
    device_rel = np.array([response.relative(f) for f in grid])
    source_db = source_spectrum_db(grid, source_pole)

    # Целевой уровень маскера (дБ) на сетке; -inf — маскер не нужен
    target = np.full(len(grid), -np.inf)

    if band_db is not None:
        # 1) Воспроизводимая часть шума: маскер повторяет уровень шума в полосе ERB
        significant = (peak_db >= min_db_threshold) & reproducible
        target = np.where(significant, band_db, target)

        # 2) Невоспроизводимые пики — купола 2-й и 3-й гармоник; воспроизводимые — 2-й.
        #    Уровень купола = уровень шума в полосе пика + усиление гармоники, поэтому
        #    высота зависит от того, насколько пик силён; ширина и склоны — по ERB и SF.
        for p in base_peaks:
            level = float(np.interp(p, grid, band_db))
            infra = response.relative(p) < infra_db or p < f_min
            harmonics = ((2, h2_gain), (3, h3_gain)) if infra else ((2, h2_gain),)
            for order, amp in harmonics:
                fc = p * order
                if fc > f_max:
                    continue
                dz = z - _critical_rate(fc)
                slope = np.where(dz > 0, dome_slope_up, dome_slope_down)
                dome = level + amp + slope * _spreading_db(dz)
                target = np.where(reproducible, np.maximum(target, dome), target)
    else:
        target = np.where(reproducible, 0.0, -np.inf)

    # Усиление фильтра: нужный выход маскера минус исходный сигнал минус АЧХ устройства
    gain = target - source_db - device_rel
    gain = np.where(reproducible & np.isfinite(gain), gain, -1e9)

    # Нормализация: вершина ровно в max_peak_limit
    unlimited_max = float(gain[reproducible].max()) if reproducible.any() and (gain[reproducible] > -1e8).any() else 0.0
    global_shift = unlimited_max - max_peak_limit

    final = np.where(gain > -1e8, gain - global_shift, min_gain)
    final = np.maximum(final, min_gain)

    # Мягкий спад под нижней границей устройства вместо вертикального обрыва
    below = (~reproducible) & (grid < f_min) & (grid >= f_min * 2.0 ** (-LOW_EDGE_TAPER_OCT))
    if below.any() and reproducible.any():
        edge_gain = float(final[reproducible][0])
        t = np.log2(f_min / grid[below]) / LOW_EDGE_TAPER_OCT  # 0 у границы -> 1 внизу
        smooth = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0)))
        final[below] = np.maximum(final[below], min_gain + (edge_gain - min_gain) * smooth)

    points = {}
    for f, g in zip(grid, final):
        fi = int(round(f))
        if freq_min <= fi <= freq_max:
            points[fi] = max(points.get(fi, min_gain), round(float(g), 1))
    points[freq_min] = points.get(freq_min, min_gain)
    points[freq_max] = points.get(freq_max, min_gain)

    preset_points = sorted(points.items(), key=lambda x: x[0])
    return preset_points, list(base_peaks), global_shift, filter_length, peaks_details
