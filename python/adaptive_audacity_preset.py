from __future__ import annotations

import json
import math
import re
import statistics
import tkinter as tk

from pathlib import Path
from tkinter import filedialog, messagebox


# ============================================================
# Общие параметры обработки
# ============================================================

# Окно сглаживания исходного спектра.
SPECTRUM_SMOOTHING_LOG_WINDOW = 0.08


# ============================================================
# Чтение чисел и TXT-файлов
# ============================================================

def parse_number(value: str) -> float:
    """
    Извлекает число из текстового поля.
    """
    value = value.strip().replace(",", ".")

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        value
    )

    if not match:
        raise ValueError

    return float(match.group(0))


def read_spectrum(filename: str) -> tuple[list[float], list[float]]:
    """
    Читает один TXT-файл со спектром.
    """
    frequencies: list[float] = []
    levels: list[float] = []

    path = Path(filename)

    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if "\t" in line:
            parts = line.split("\t")
        elif ";" in line:
            parts = line.split(";")
        else:
            parts = line.split()

        if len(parts) < 2:
            continue

        try:
            frequency = parse_number(parts[0])
            level = parse_number(parts[1])
        except ValueError:
            continue

        if frequency <= 0 or not math.isfinite(frequency) or not math.isfinite(level):
            continue

        frequencies.append(frequency)
        levels.append(level)

    if len(frequencies) < 3:
        raise ValueError(f"В файле недостаточно точек спектра:\n{filename}")

    pairs = sorted(zip(frequencies, levels), key=lambda item: item[0])

    grouped: dict[float, list[float]] = {}
    for frequency, level in pairs:
        grouped.setdefault(frequency, []).append(level)

    result_frequencies: list[float] = []
    result_levels: list[float] = []

    for frequency, values in grouped.items():
        result_frequencies.append(frequency)
        result_levels.append(sum(values) / len(values))

    if len(result_frequencies) < 3:
        raise ValueError(f"После очистки в файле осталось мало частот:\n{filename}")

    return result_frequencies, result_levels


# ============================================================
# Работа с профилем колонки
# ============================================================

def load_profile(filename: str) -> dict:
    """
    Загружает JSON-профиль колонки. Поле baseline_band больше не требуется.
    """
    path = Path(filename)
    profile = json.loads(path.read_text(encoding="utf-8-sig"))

    required_fields = [
        "name",
        "preset_frequencies",
        "speaker_response",
        "masking_margin_db",
        "minimum_mask_db",
        "min_gain",
        "max_gain"
    ]

    for field in required_fields:
        if field not in profile:
            raise ValueError(f"В профиле отсутствует поле: {field}")

    return profile


# ============================================================
# Математические функции
# ============================================================

def percentile(values: list[float], percentile_value: float) -> float:
    """
    Линейно интерполированный процентиль.
    """
    if not values:
        raise ValueError("Невозможно вычислить процентиль пустого списка.")

    sorted_values = sorted(values)

    if percentile_value <= 0:
        return sorted_values[0]
    if percentile_value >= 100:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    fraction = position - lower_index
    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def interpolate_log_curve(points: list, target_frequency: float) -> float:
    """
    Интерполяция кривой по логарифму частоты.
    """
    normalized_points = sorted([(float(p[0]), float(p[1])) for p in points], key=lambda item: item[0])

    if target_frequency <= normalized_points[0][0]:
        return normalized_points[0][1]
    if target_frequency >= normalized_points[-1][0]:
        return normalized_points[-1][1]

    for index in range(len(normalized_points) - 1):
        f1, y1 = normalized_points[index]
        f2, y2 = normalized_points[index + 1]

        if f1 <= target_frequency <= f2:
            x1 = math.log(f1)
            x2 = math.log(f2)
            x = math.log(target_frequency)
            if x2 == x1:
                return (y1 + y2) / 2.0
            return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

    return normalized_points[-1][1]


def interpolate_log_spectrum(frequencies: list[float], levels: list[float], target_frequency: float) -> float:
    if target_frequency <= frequencies[0]:
        return levels[0]
    if target_frequency >= frequencies[-1]:
        return levels[-1]

    for index in range(len(frequencies) - 1):
        f1 = frequencies[index]
        f2 = frequencies[index + 1]

        if f1 <= target_frequency <= f2:
            x1 = math.log(f1)
            x2 = math.log(f2)
            x = math.log(target_frequency)
            if x2 == x1:
                return (levels[index] + levels[index + 1]) / 2.0
            return levels[index] + (levels[index + 1] - levels[index]) * ((x - x1) / (x2 - x1))

    return levels[-1]


def smooth_spectrum_at(frequencies: list[float], levels: list[float], target_frequency: float) -> float:
    selected_levels = []
    for frequency, level in zip(frequencies, levels):
        distance = abs(math.log10(frequency / target_frequency))
        if distance <= SPECTRUM_SMOOTHING_LOG_WINDOW:
            selected_levels.append(level)
    if selected_levels:
        return sum(selected_levels) / len(selected_levels)
    return interpolate_log_spectrum(frequencies, levels, target_frequency)


def median_in_log_window(frequencies: list[float], values: list[float], target_frequency: float, window_octaves: float) -> float:
    selected_values = []
    for frequency, value in zip(frequencies, values):
        distance_octaves = abs(math.log2(frequency / target_frequency))
        if distance_octaves <= window_octaves:
            selected_values.append(value)
    if not selected_values:
        return 0.0
    return statistics.median(selected_values)


def gaussian_smooth_log_curve(frequencies: list[float], values: list[float], sigma_octaves: float) -> list[float]:
    if sigma_octaves <= 0:
        return list(values)

    smoothed = []
    for target_frequency in frequencies:
        weighted_sum = 0.0
        weight_sum = 0.0

        for frequency, value in zip(frequencies, values):
            distance_octaves = math.log2(frequency / target_frequency)
            weight = math.exp(-0.5 * (distance_octaves / sigma_octaves) ** 2)
            weighted_sum += value * weight
            weight_sum += weight

        if weight_sum == 0:
            smoothed.append(values[frequencies.index(target_frequency)])
        else:
            smoothed.append(weighted_sum / weight_sum)

    return smoothed


# ============================================================
# Автоматический динамический расчет превышения шума
# ============================================================

def get_percentile_prominence_dynamic(
    spectra: list[tuple[list[float], list[float]]],
    target_frequency: float,
    profile: dict
) -> tuple[float, float, list[float]]:
    """
    Вычисляет превышение над динамическим порогом тишины (без baseline_band).
    """
    relative_prominences = []
    absolute_levels = []

    for frequencies, levels in spectra:
        # Автоматический поиск уровня "пола" тишины (5-й процентиль самых тихих участков)
        dynamic_baseline = percentile(levels, 5.0) 
        
        spectrum_level = smooth_spectrum_at(frequencies, levels, target_frequency)
        prominence = max(0.0, spectrum_level - dynamic_baseline)
        
        relative_prominences.append(prominence)
        absolute_levels.append(spectrum_level)

    percentile_value = float(profile.get("spectrum_percentile", 90.0))
    high_percentile = percentile(relative_prominences, percentile_value)
    absolute_percentile = percentile(absolute_levels, percentile_value)

    return high_percentile, absolute_percentile, relative_prominences


# ============================================================
# Основной расчёт маскирующей кривой (Автономный)
# ============================================================

def create_gain_curve(
    spectra: list[tuple[list[float], list[float]]],
    profile: dict
):
    preset_frequencies = [float(value) for value in profile["preset_frequencies"]]
    speaker_response = profile["speaker_response"]

    masking_margin = float(profile.get("masking_margin_db", 3.0))
    minimum_mask = float(profile.get("minimum_mask_db", 2.0))
    level_offset = float(profile.get("level_offset_db", 0.0))
    narrow_peak_limit = float(profile.get("narrow_peak_limit_db", 4.0))
    peak_detection_window = float(profile.get("peak_detection_window_octaves", 0.35))
    curve_smoothing = float(profile.get("curve_smoothing_octaves", 0.30))
    minimum_gain = float(profile.get("min_gain", -24.0))
    maximum_gain = float(profile.get("max_gain", 4.0))

    raw_prominences = []
    absolute_percentile_levels = []
    all_relative_values = []

    # 1. Сбор данных на основе динамического процентиля тишины
    for target_frequency in preset_frequencies:
        prominence_percentile, absolute_level, values = (
            get_percentile_prominence_dynamic(spectra, target_frequency, profile)
        )
        raw_prominences.append(prominence_percentile)
        absolute_percentile_levels.append(absolute_level)
        all_relative_values.append(values)

    # 2. Защита от узких пиков
    limited_prominences = []
    for index, target_frequency in enumerate(preset_frequencies):
        local_median = median_in_log_window(
            preset_frequencies, raw_prominences, target_frequency, peak_detection_window
        )
        limited_value = min(raw_prominences[index], local_median + narrow_peak_limit)
        limited_prominences.append(limited_value)

    # 3. Расчет целевой маски
    raw_masking_curve = []
    for prominence in limited_prominences:
        desired_mask_level = max(minimum_mask, prominence + masking_margin)
        raw_masking_curve.append(desired_mask_level)

    smooth_masking_curve = gaussian_smooth_log_curve(preset_frequencies, raw_masking_curve, curve_smoothing)

    # 4. Отсечка зон тишины с локальным пик-детектором (убирает полку на инфрабасе)
    raw_gains = []
    diagnostics = []

    for index, target_frequency in enumerate(preset_frequencies):
        speaker_level = interpolate_log_curve(speaker_response, target_frequency)
        desired_mask_level = smooth_masking_curve[index]
        prominence = limited_prominences[index]

        # Ищем локальную медиану вокруг текущей частоты в пределах 1 октавы
        local_env = median_in_log_window(preset_frequencies, limited_prominences, target_frequency, 1.0)

        # Частота считается активным шумом, если она явно выделяется на общем фоне
        # и уровень prominence имеет физический смысл (выше 3 дБ)
        if prominence < 3.0 or prominence < (local_env - 2.0):
            raw_gain = minimum_gain
        else:
            raw_gain = prominence + masking_margin + level_offset

        raw_gains.append(raw_gain)
        diagnostics.append({
            "frequency": target_frequency,
            "percentile_level": absolute_percentile_levels[index],
            "raw_prominence": raw_prominences[index],
            "limited_prominence": limited_prominences[index],
            "masking_level": desired_mask_level,
            "speaker_response": speaker_level,
            "raw_gain": raw_gain
        })

    # 5. Финальное сглаживание купола
    gains = gaussian_smooth_log_curve(preset_frequencies, raw_gains, curve_smoothing)
    for index, gain in enumerate(gains):
        if raw_gains[index] == minimum_gain:
            gains[index] = minimum_gain
        gains[index] = clamp(gain, minimum_gain, maximum_gain)
        diagnostics[index]["final_gain"] = gains[index]

    return preset_frequencies, gains, diagnostics, raw_prominences, limited_prominences

# ============================================================
# Сборка пресета и запуск GUI
# ============================================================

def build_audacity_preset(
    profile: dict,
    frequencies: list[float],
    gains: list[float]
) -> str:
    parts = ["FilterCurve:"]
    for index, frequency in enumerate(frequencies):
        parts.append(f'f{index}="{format_number(frequency)}"')

    parts.extend([
        f'FilterLength="{profile.get("filter_length", 8191)}"',
        f'InterpolateLin="{profile.get("interpolate_lin", 0)}"',
        f'InterpolationMethod="{profile.get("interpolation_method", "B-spline")}"'
    ])

    for index, gain in enumerate(gains):
        parts.append(f'v{index}="{format_number(gain)}"')

    return " ".join(parts)


def format_number(value: float) -> str:
    return f"{value:.12g}"


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        profile_file = filedialog.askopenfilename(
            title="Выберите профиль колонки",
            filetypes=[("JSON-профили", "*.json"), ("Все файлы", ".*")]
        )
        if not profile_file:
            return

        profile = load_profile(profile_file)

        input_files = filedialog.askopenfilenames(
            title="Выберите TXT-файлы со спектрами",
            filetypes=[("TXT-файлы", "*.txt"), ("Все файлы", ".*")]
        )
        if not input_files:
            return

        spectra = []
        for filename in input_files:
            frequencies, levels = read_spectrum(filename)
            spectra.append((frequencies, levels))

        preset_frequencies, gains, diagnostics, _, _ = create_gain_curve(
            spectra,
            profile
        )
        preset = build_audacity_preset(profile, preset_frequencies, gains)

        output_file = filedialog.asksaveasfilename(
            title="Сохранить пресет Audacity",
            initialfile=f"{profile['name'].replace(' ', '_')}_Dynamic_Preset.txt",
            defaultextension=".txt",
            filetypes=[("Audacity Filter Curve", "*.txt")]
        )
        if not output_file:
            return

        Path(output_file).write_text(preset + "\n", encoding="utf-8")
        messagebox.showinfo("Готово", "Пресет успешно сохранен!")

    except Exception as error:
        messagebox.showerror("Ошибка", str(error))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
