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
HARMONIC_MAX_FREQ = 250.0     # купола гармоник строятся только для пиков суб-баса не выше этой частоты
PEAK_PROMINENCE_DB = 4.0      # минимальная выделенность пика над окружением
PEAK_BAND_OCT = 1.0 / 6.0     # полоса сглаживания при поиске пиков
GRID_ERB_STEP = 0.1           # шаг сетки по шкале ERB-number
DOME_SLOPE_DOWN = 8.0         # крутизна нижнего склона купола относительно SF (больше — круче)
AUTO_SLOPE_DOWN_GRID = (8.0, 12.0, 16.0)
AUTO_SLOPE_UP_GRID = (1.0, 2.0, 3.0)
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


def source_spectrum_db(freqs, pole="brown_default", sample_rate: float = 44100.0):
    """Спектр исходного шума (плотность мощности, дБ, относительный).
    pole: число (0.995) — коричневый шум генератора (y = 0.995*y + x), -6 дБ/окт выше ~35 Гц;
          "pink" — розовый (-3 дБ/окт); None — белый (плоский)."""
    freqs = np.asarray(freqs, dtype=float)
    if pole == "brown_default":
        pole = 0.995
    if pole is None:
        return np.zeros_like(freqs)
    if isinstance(pole, str) and pole == "pink":
        return -10.0 * np.log10(np.maximum(freqs, 1e-3) / 1000.0)
    w = 2.0 * np.pi * freqs / sample_rate
    mag = np.abs(1.0 - float(pole) * np.exp(-1j * w))
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


NORMALIZATION_MODES = ("peak", "floor", "none")


def analyze_noise(file_paths, min_db_threshold, grid_freqs, freq_min, freq_max,
                  combine="max", normalize="peak", auto_range_db=20.0):
    """Анализ спектров шума с оценкой качества каждого файла.

    Каждый спектр оценивается по собственному уровню: записи сделаны в разные дни, в разных
    помещениях и на разных устройствах, поэтому
      * порог пиков задаётся как «динамический диапазон ниже самого сильного бина ФАЙЛА»
        (range = самый громкий бин среди всех файлов − min_db_threshold; при min_db_threshold=None
        range = auto_range_db), и тихий файл не теряет свои пики из-за общего порога;
      * уровни приводятся к общей шкале (normalize): "peak" — по пику файла, "floor" — по
        шумовому полу файла (медиана), "none" — без изменений (одно и то же устройство);
      * значимой считается полоса, значимая хотя бы в одном файле (объединение, а не пересечение).

    Возвращает словарь:
      band_db — суммарный уровень в полосах ERB на сетке (после приведения шкал и объединения);
      significant — маска значимых полос; base_peaks — частоты пиков (целые Гц);
      peaks_details — пики по файлам (freq/level/source, уровни исходные); file_stats — оценка файлов;
      threshold — порог для самого громкого файла (абсолютный, дБ); range_db — применённый диапазон.
    """
    grid_freqs = np.asarray(grid_freqs, dtype=float)
    widths = _erb_bandwidth(grid_freqs)

    octaves = math.log2(freq_max / freq_min)
    fine_n = max(8, int(24 * octaves))
    fine_centers = freq_min * 2.0 ** (np.arange(fine_n + 1) * (octaves / fine_n))
    fine_widths = fine_centers * (2.0 ** (PEAK_BAND_OCT / 2.0) - 2.0 ** (-PEAK_BAND_OCT / 2.0))

    loaded = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue
        freqs, levels = _load_noise_file(path)
        if len(freqs) < 2:
            continue
        order = np.argsort(freqs)
        loaded.append((path.name, freqs[order], levels[order]))
    if not loaded:
        return None

    def in_band(freqs):
        return (freqs >= freq_min) & (freqs <= freq_max)

    tops = [float(lv[in_band(f)].max()) if in_band(f).any() else float(lv.max()) for _, f, lv in loaded]
    top_ref = max(tops)
    range_db = auto_range_db if min_db_threshold is None else max(top_ref - float(min_db_threshold), 0.1)

    bands, fine_sums, sig_grid, sig_fine = [], [], [], []
    peaks_details, file_stats, norm_files = [], [], []
    for (name, freqs, levels), top in zip(loaded, tops):
        m_band = in_band(freqs)
        floor = float(np.median(levels[m_band])) if m_band.any() else float(np.median(levels))
        ref = {"peak": top, "floor": floor}.get(normalize, 0.0)
        norm = levels - ref
        thr_file = top - range_db            # исходная шкала файла
        thr_norm = thr_file - ref            # шкала после приведения

        bands.append(_band_levels_db(freqs, norm, grid_freqs, widths))
        sig_grid.append(_window_max_db(freqs, norm, grid_freqs, widths) >= thr_norm)
        fs = _band_levels_db(freqs, norm, fine_centers, fine_widths)
        fp = _window_max_db(freqs, norm, fine_centers, fine_widths)
        fine_sums.append(fs)
        sig_fine.append(fp >= thr_norm)
        norm_files.append((freqs, norm))

        file_peaks = []
        for idx in _find_peaks(fs, PEAK_PROMINENCE_DB):
            if fp[idx] < thr_norm:
                continue
            lo, hi = fine_centers[idx] - fine_widths[idx] / 2, fine_centers[idx] + fine_widths[idx] / 2
            m = (freqs >= lo) & (freqs <= hi)
            k = int(np.argmax(levels[m]))
            peak_f = int(round(freqs[m][k]))
            file_peaks.append(peak_f)
            peaks_details.append({"freq": peak_f, "level": round(float(levels[m][k]), 1), "source": name})

        top_f = float(freqs[m_band][int(np.argmax(levels[m_band]))]) if m_band.any() else 0.0
        file_stats.append({
            "name": name, "top": top, "top_freq": int(round(top_f)), "floor": floor,
            "dynamic": top - floor, "threshold": thr_file, "peaks": file_peaks,
            "offset": ref,
        })

    stack = np.vstack(bands)
    if combine == "mean":
        band_db = 10.0 * np.log10(np.mean(10.0 ** (stack / 10.0), axis=0))
    else:
        band_db = stack.max(axis=0)
    significant = np.any(np.vstack(sig_grid), axis=0)

    # Пики для гармоник — по общей огибающей; значима, если значима хотя бы в одном файле
    fine_env = np.vstack(fine_sums).max(axis=0)
    fine_sig = np.any(np.vstack(sig_fine), axis=0)
    base_peaks = set()
    for idx in _find_peaks(fine_env, PEAK_PROMINENCE_DB):
        if not fine_sig[idx]:
            continue
        lo, hi = fine_centers[idx] - fine_widths[idx] / 2, fine_centers[idx] + fine_widths[idx] / 2
        best_f, best_l = None, -1e9
        for freqs, norm in norm_files:
            m = (freqs >= lo) & (freqs <= hi)
            if m.any():
                k = int(np.argmax(norm[m]))
                if norm[m][k] > best_l:
                    best_l, best_f = norm[m][k], freqs[m][k]
        if best_f is not None:
            base_peaks.add(int(round(best_f)))

    peaks_details.sort(key=lambda p: p["level"], reverse=True)
    return {
        "band_db": band_db, "significant": significant, "base_peaks": sorted(base_peaks),
        "peaks_details": peaks_details, "file_stats": file_stats,
        "threshold": top_ref - range_db, "range_db": range_db,
    }


def estimate_auto_threshold(file_paths, range_db: float = 20.0, default: float = -70.0) -> float:
    """Автопорог пиков: самый сильный бин по всем спектрам минус range_db."""
    top = None
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue
        _, levels = _load_noise_file(path)
        if len(levels):
            top = float(levels.max()) if top is None else max(top, float(levels.max()))
    return default if top is None else top - range_db


def auto_filter_length(lowest_freq: float, sample_rate: float = 44100.0) -> int:
    """Окно фильтра: частотное разрешение fs/N не грубее четверти полосы ERB на низшей частоте шума."""
    need = 4.0 * sample_rate / float(_erb_bandwidth(max(lowest_freq, 1.0)))
    for n in (1023, 2047, 4095, 8191, 16383, 32767):
        if n >= need:
            return n
    return 32767


def get_all_spectrum_peaks_dynamic(file_paths, min_db_threshold):
    """Совместимость: (множество частот пиков, подробности пиков)."""
    grid = _erb_number_inv(np.arange(_erb_number(FREQ_MIN_HZ), _erb_number(FREQ_MAX_HZ), GRID_ERB_STEP))
    res = analyze_noise(file_paths, min_db_threshold, grid, FREQ_MIN_HZ, FREQ_MAX_HZ)
    if res is None:
        return set(), []
    return set(res["base_peaks"]), res["peaks_details"]


# --------------------------------------------------------------------------------------
# Решение задачи маскировки
# --------------------------------------------------------------------------------------
def build_normalized_preset_dynamic(
    profile_path: str | None,
    spectrum_paths: list[str],
    max_peak_limit: float,
    min_db_threshold: float | None,
    min_gain: float,
    filter_length: int | None,
    h2_gain: float | None,
    h3_gain: float | None,
    speaker_response=None,
    profile_name: str | None = None,
    freq_low: float | None = None,
    freq_min: int = FREQ_MIN_HZ,
    freq_max: int = FREQ_MAX_HZ,
    source_pole: float | None = 0.995,
    noise_combine: str = "max",
    dome_slope_down: float | None = DOME_SLOPE_DOWN,
    dome_slope_up: float | None = DOME_SLOPE_UP,
    return_info: bool = False,
    auto_range_db: float = 20.0,
    sample_rate: float = 44100.0,
    auto_tol_db: float = 0.0,
    file_normalization: str = "peak",
    harmonic_max_freq: float = HARMONIC_MAX_FREQ,
):
    """Строит маскирующую кривую фильтра под заданный шум и устройство.

    Параметры АЧХ: speaker_response (dict / пары / путь) либо profile_path (+profile_name).
    min_db_threshold — шум в полосе ниже этого уровня маскировать не нужно.
    h2_gain, h3_gain — высота куполов 2-й и 3-й гармоник (дБ); масштабируется уровнем шума пика.
    freq_low — нижняя граница воспроизведения, Гц (по умолчанию — первая точка АЧХ из профиля).
    Пики ниже этой границы считаются невоспроизводимыми (получают купола 2-й и 3-й гармоник).
    min_db_threshold=None — автопорог: самый сильный бин шума минус auto_range_db. Порог применяется к каждому
    файлу относительно его собственного максимума (тихая запись не теряет пики), min_db_threshold задаёт
    диапазон относительно самого громкого файла. file_normalization: "peak" | "floor" | "none".
    filter_length=None — окно фильтра подбирается по разрешению на самой низкой частоте шума.
    h2_gain / h3_gain = None — автоподбор: наибольшие усиления гармоник, не ухудшающие модельную
    маскировку шума более чем на auto_tol_db (по умолчанию 0 — строгий максимум).
    dome_slope_down / dome_slope_up = None — автоподбор склонов по тому же критерию (строгий максимум).
    source_pole — полюс исходного шума (0.995 — коричневый, None — белый).
    noise_combine — объединение файлов: "max" (худший случай) или "mean".
    dome_slope_down / dome_slope_up — крутизна нижнего / верхнего склона куполов гармоник.
    return_info — дополнительно вернуть словарь с разбором формирования купола (для лога и графика).
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

    noise = analyze_noise(
        spectrum_paths, min_db_threshold, grid, freq_min, freq_max, noise_combine,
        normalize=file_normalization, auto_range_db=auto_range_db,
    )
    if noise is None:
        band_db, sig_mask, base_peaks, peaks_details, file_stats = None, None, [], [], []
        min_db_threshold = -70.0 if min_db_threshold is None else min_db_threshold
    else:
        band_db, sig_mask = noise["band_db"], noise["significant"]
        base_peaks, peaks_details, file_stats = noise["base_peaks"], noise["peaks_details"], noise["file_stats"]
        min_db_threshold = noise["threshold"]

    if filter_length is None:
        lowest = min(base_peaks) if base_peaks else float(f_min)
        filter_length = auto_filter_length(lowest, sample_rate)

    reproducible = (grid >= f_min) & (grid <= f_max)
    device_rel = np.array([response.relative(f) for f in grid])
    source_db = source_spectrum_db(grid, source_pole)

    # Целевой уровень маскера (дБ) на сетке; -inf — маскер не нужен
    follow = np.full(len(grid), -np.inf)
    peak_info = []
    if band_db is not None:
        # 1) Воспроизводимая часть шума: маскер повторяет уровень шума в полосе ERB
        significant = sig_mask & reproducible
        follow = np.where(significant, band_db, -np.inf)
        for p in base_peaks:
            peak_info.append({
                "freq": p, "level": float(np.interp(p, grid, band_db)), "infra": bool(p < f_min),
            })

    def compose(h2, h3, sd, su):
        """Кривая для заданных усилений гармоник.
        Невоспроизводимые пики — купола 2-й и 3-й гармоник; воспроизводимые — 2-й.
        Уровень купола = уровень шума в полосе пика + усиление гармоники (высота зависит от силы
        пика); ширина и склоны — по шкале ERB и функции распространения маскирования."""
        target = follow.copy() if band_db is not None else np.where(reproducible, 0.0, -np.inf)
        domes = []
        for pk in peak_info:
            p, level = pk["freq"], pk["level"]
            if p > harmonic_max_freq:
                continue
            harmonics = ((2, h2), (3, h3)) if pk["infra"] else ((2, h2),)
            for order, amp in harmonics:
                fc = p * order
                if fc > f_max:
                    continue
                dz = z - _critical_rate(fc)
                slope = np.where(dz > 0, su, sd)
                dome = np.where(reproducible, level + amp + slope * _spreading_db(dz), -np.inf)
                target = np.maximum(target, dome)
                domes.append({
                    "label": f"{order}×{p}", "order": order, "peak": p, "center": fc,
                    "level": level + amp, "array": dome,
                })

        # Усиление фильтра: нужный выход маскера минус исходный сигнал минус АЧХ устройства
        gain = target - source_db - device_rel
        gain = np.where(reproducible & np.isfinite(gain), gain, -1e9)

        # Нормализация: вершина ровно в max_peak_limit
        has_gain = reproducible.any() and (gain[reproducible] > -1e8).any()
        unlimited_max = float(gain[reproducible].max()) if has_gain else 0.0
        shift = unlimited_max - max_peak_limit
        curve = np.where(gain > -1e8, gain - shift, min_gain)
        curve = np.maximum(curve, min_gain)

        # Мягкий спад под нижней границей устройства вместо вертикального обрыва
        below = (~reproducible) & (grid < f_min) & (grid >= f_min * 2.0 ** (-LOW_EDGE_TAPER_OCT))
        if below.any() and reproducible.any():
            edge_gain = float(curve[reproducible][0])
            t = np.log2(f_min / grid[below]) / LOW_EDGE_TAPER_OCT  # 0 у границы -> 1 внизу
            smooth = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0)))
            curve[below] = np.maximum(curve[below], min_gain + (edge_gain - min_gain) * smooth)
        return curve, domes, shift, target

    # Автоподбор гармоник и склонов по модели покрытия шума (взвешенной энергией шума).
    # Склоны — строгий максимум; гармоники — наибольшие усиления, не ухудшающие покрытие
    # более чем на auto_tol_db относительно лучшего.
    auto_harmonics = h2_gain is None or h3_gain is None
    auto_slopes = dome_slope_down is None or dome_slope_up is None
    h2 = 7.0 if h2_gain is None else float(h2_gain)
    h3 = 4.5 if h3_gain is None else float(h3_gain)
    sd = DOME_SLOPE_DOWN if dome_slope_down is None else float(dome_slope_down)
    su = DOME_SLOPE_UP if dome_slope_up is None else float(dome_slope_up)

    sig_bands = sig_mask if band_db is not None else None
    if (auto_harmonics or auto_slopes) and peak_info and sig_bands is not None and sig_bands.any():
        spread_lin = 10.0 ** (_spreading_db(z[:, None] - z[None, :]) / 10.0)
        weights = 10.0 ** (band_db[sig_bands] / 10.0)
        weights = weights / weights.sum()
        has_infra = any(pk["infra"] for pk in peak_info)

        def coverage(a2, a3, down, up):
            curve, _, _, _ = compose(a2, a3, down, up)
            out = curve + source_db + device_rel
            excitation = 10.0 * np.log10(spread_lin @ 10.0 ** (out / 10.0) + 1e-300)
            return float(((excitation - band_db)[sig_bands] * weights).sum())

        def best_harmonics(down, up, tol):
            def search(v2, v3):
                scored = [(coverage(a2, a3, down, up), a2, a3) for a2 in v2 for a3 in v3]
                best = max(sc[0] for sc in scored)
                good = [sc for sc in scored if sc[0] >= best - tol]
                chosen = max(good, key=lambda sc: (sc[1] + sc[2], sc[2]))
                return chosen[1], chosen[2]

            h3_coarse = [float(v) for v in range(0, 16)] if has_infra else [0.0]
            c2, c3 = search([float(v) for v in range(0, 16)], h3_coarse)
            near = np.arange(-1.0, 1.01, 0.25)
            fine2 = sorted({max(0.0, min(15.0, c2 + d)) for d in near})
            fine3 = sorted({max(0.0, min(15.0, c3 + d)) for d in near}) if has_infra else [0.0]
            r2, r3 = search(fine2, fine3)
            return round(r2, 2), round(r3, 2)

        if auto_slopes:
            # Склоны и гармоники связаны, поэтому для каждой пары склонов гармоники подбираются заново
            # (грубо, шаг 2 дБ); итоговые гармоники затем уточняются для лучшей пары
            coarse = [float(v) for v in range(0, 16, 2)]
            coarse3 = coarse if has_infra else [0.0]

            def slope_score(down, up):
                if not auto_harmonics:
                    return coverage(h2, h3, down, up)
                return max(coverage(a2, a3, down, up) for a2 in coarse for a3 in coarse3)

            scored = [
                (slope_score(down, up), down, up)
                for down in AUTO_SLOPE_DOWN_GRID for up in AUTO_SLOPE_UP_GRID
            ]
            _, sd, su = max(scored, key=lambda c: c[0])
        if auto_harmonics:
            h2, h3 = best_harmonics(sd, su, auto_tol_db)

    h2_gain, h3_gain, dome_slope_down, dome_slope_up = h2, h3, sd, su
    final, dome_list, global_shift, final_target = compose(h2_gain, h3_gain, sd, su)

    points = {}
    for f, g in zip(grid, final):
        fi = int(round(f))
        if freq_min <= fi <= freq_max:
            points[fi] = max(points.get(fi, min_gain), round(float(g), 1))
    points[freq_min] = points.get(freq_min, min_gain)
    points[freq_max] = points.get(freq_max, min_gain)

    preset_points = sorted(points.items(), key=lambda x: x[0])
    result = (preset_points, list(base_peaks), global_shift, filter_length, peaks_details)
    if return_info:
        info = {
            "grid": grid, "f_min": f_min, "f_max": f_max,
            "follow": follow, "domes": dome_list, "peaks": peak_info,
            "device_comp": -device_rel, "shift": global_shift,
            "source": ("розовый" if (isinstance(source_pole, str) and source_pole == "pink")
                       else "белый" if source_pole is None else "коричневый"),
            "source_key": ("pink" if (isinstance(source_pole, str) and source_pole == "pink")
                           else "white" if source_pole is None else "brown"),
            "target": final_target, "reproducible": reproducible,
            "max_peak_limit": float(max_peak_limit), "min_gain": float(min_gain),
            "threshold": float(min_db_threshold), "filter_length": int(filter_length),
            "file_stats": file_stats, "normalization": file_normalization,
            "harmonic_max_freq": float(harmonic_max_freq),
            "h2_gain": float(h2_gain), "h3_gain": float(h3_gain), "auto_harmonics": bool(auto_harmonics),
            "slope_down": float(sd), "slope_up": float(su), "auto_slopes": bool(auto_slopes),
        }
        return result + (info,)
    return result


def explain_frequency(info: dict, freq: float) -> dict:
    """Из чего сложилось усиление кривой на частоте freq.

    components — вклады (целевой уровень маскера, дБ) в порядке убывания: «шум в полосе»
    (воспроизводимая часть шума) и купола гармоник; побеждает наибольший.
    device_comp_db — компенсация спада АЧХ устройства на этой частоте (дБ)."""
    grid = info["grid"]
    i = int(np.argmin(np.abs(grid - freq)))
    comps = []
    if np.isfinite(info["follow"][i]):
        comps.append(("шум в полосе", float(info["follow"][i])))
    for d in info["domes"]:
        v = d["array"][i]
        if np.isfinite(v):
            comps.append((f"купол {d['label']} ({d['center']} Гц)", float(v)))
    comps.sort(key=lambda c: -c[1])
    return {
        "freq": int(round(grid[i])),
        "components": comps,
        "dominant": comps[0] if comps else None,
        "gap_db": (comps[0][1] - comps[1][1]) if len(comps) > 1 else None,
        "device_comp_db": float(info["device_comp"][i]),
    }


SOURCE_TYPE_OPTIONS = (
    ("brown", "коричневый", 0.995),
    ("pink", "розовый", "pink"),
    ("white", "белый", None),
)


def compare_source_types(info: dict) -> list:
    """Как итоговая кривая работала бы с разными исходными шумами (при том же целевом маскере).

    Кривая пересчитывается под каждый тип (компенсирует его наклон), затем по спектру выхода
    оценивается, какая доля мощности уходит ВНЕ активной полосы кривой (на «полу» min_gain)
    и в область выше 2 кГц (слышимое шипение). Меньше — лучше: больше запаса по громкости и
    перегрузке достаётся полезной полосе маскера."""
    grid, target = info["grid"], info["target"]
    reproducible = info["reproducible"]
    device_rel = -info["device_comp"]
    limit, floor = info["max_peak_limit"], info["min_gain"]
    df = np.gradient(grid)
    results = []
    for key, label, pole in SOURCE_TYPE_OPTIONS:
        src = source_spectrum_db(grid, pole)
        gain = np.where(reproducible & np.isfinite(target), target - src - device_rel, -1e9)
        valid = gain > -1e8
        top = float(gain[valid].max()) if valid.any() else 0.0
        curve = np.maximum(np.where(valid, gain - (top - limit), floor), floor)
        power = 10.0 ** ((curve + src + device_rel) / 10.0) * df
        total = float(power.sum())
        active = curve > floor + 0.5
        results.append({
            "key": key, "label": label,
            "outside_pct": 100.0 * float(power[~active].sum()) / total,
            "high_pct": 100.0 * float(power[grid >= 2000.0].sum()) / total,
            "range_db": float(curve[active].max() - curve[active].min()) if active.any() else 0.0,
        })
    return results
