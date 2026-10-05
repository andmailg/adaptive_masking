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
# 0.08 соответствует примерно ±20% по частоте.
SPECTRUM_SMOOTHING_LOG_WINDOW = 0.08


# ============================================================
# Чтение чисел и TXT-файлов
# ============================================================

def parse_number(value: str) -> float:
    """
    Извлекает число из текстового поля.

    Поддерживает:
        12.5
        12,5
        -30
        1.2e-3

    Заголовки и строки без чисел будут пропущены.
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

    Поддерживаемые форматы:

        частота<TAB>уровень
        частота;уровень
        частота уровень

    Пример:

        20.0    -48.2
        25.0    -46.7
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
            # Заголовок или некорректная строка
            continue

        if frequency <= 0:
            continue

        if not math.isfinite(frequency):
            continue

        if not math.isfinite(level):
            continue

        frequencies.append(frequency)
        levels.append(level)

    if len(frequencies) < 3:
        raise ValueError(
            f"В файле недостаточно точек спектра:\n{filename}"
        )

    # Сортировка по частоте
    pairs = sorted(
        zip(frequencies, levels),
        key=lambda item: item[0]
    )

    # Объединение повторяющихся частот.
    # Если одна частота встречается несколько раз,
    # её уровни усредняются.
    grouped: dict[float, list[float]] = {}

    for frequency, level in pairs:
        grouped.setdefault(frequency, []).append(level)

    result_frequencies: list[float] = []
    result_levels: list[float] = []

    for frequency, values in grouped.items():
        result_frequencies.append(frequency)
        result_levels.append(
            sum(values) / len(values)
        )

    if len(result_frequencies) < 3:
        raise ValueError(
            f"После очистки в файле осталось мало частот:\n{filename}"
        )

    return result_frequencies, result_levels


# ============================================================
# Работа с профилем колонки
# ============================================================

def load_profile(filename: str) -> dict:
    """
    Загружает JSON-профиль колонки.
    """

    path = Path(filename)

    profile = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    required_fields = [
        "name",
        "preset_frequencies",
        "speaker_response",
        "baseline_band",
        "masking_margin_db",
        "minimum_mask_db",
        "min_gain",
        "max_gain"
    ]

    for field in required_fields:
        if field not in profile:
            raise ValueError(
                f"В профиле отсутствует поле: {field}"
            )

    if len(profile["preset_frequencies"]) < 3:
        raise ValueError(
            "В preset_frequencies должно быть минимум "
            "три частоты."
        )

    if len(profile["speaker_response"]) < 2:
        raise ValueError(
            "В speaker_response должно быть минимум "
            "две точки."
        )

    if len(profile["baseline_band"]) != 2:
        raise ValueError(
            "baseline_band должен содержать две частоты."
        )

    return profile


# ============================================================
# Математические функции
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float
) -> float:
    """
    Линейно интерполированный процентиль.

    percentile_value:
        0   — минимум
        50  — медиана
        90  — 90-й процентиль
        100 — максимум
    """

    if not values:
        raise ValueError("Невозможно вычислить процентиль пустого списка.")

    sorted_values = sorted(values)

    if percentile_value <= 0:
        return sorted_values[0]

    if percentile_value >= 100:
        return sorted_values[-1]

    position = (
        (len(sorted_values) - 1)
        * percentile_value
        / 100.0
    )

    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        + (
            sorted_values[upper_index]
            - sorted_values[lower_index]
        )
        * fraction
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float
) -> float:
    return max(minimum, min(value, maximum))


def interpolate_log_curve(
    points: list,
    target_frequency: float
) -> float:
    """
    Интерполяция кривой по логарифму частоты.

    points:

        [
            [частота, значение],
            [частота, значение]
        ]
    """

    normalized_points = sorted(
        [
            (
                float(point[0]),
                float(point[1])
            )
            for point in points
        ],
        key=lambda item: item[0]
    )

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

            ratio = (x - x1) / (x2 - x1)

            return y1 + (y2 - y1) * ratio

    return normalized_points[-1][1]


def interpolate_log_spectrum(
    frequencies: list[float],
    levels: list[float],
    target_frequency: float
) -> float:
    """
    Линейная интерполяция спектра по логарифму частоты.
    """

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

            y1 = levels[index]
            y2 = levels[index + 1]

            if x2 == x1:
                return (y1 + y2) / 2.0

            ratio = (x - x1) / (x2 - x1)

            return y1 + (y2 - y1) * ratio

    return levels[-1]


def smooth_spectrum_at(
    frequencies: list[float],
    levels: list[float],
    target_frequency: float
) -> float:
    """
    Усредняет спектр в логарифмическом окне около частоты.
    """

    selected_levels = []

    for frequency, level in zip(frequencies, levels):
        distance = abs(
            math.log10(frequency / target_frequency)
        )

        if distance <= SPECTRUM_SMOOTHING_LOG_WINDOW:
            selected_levels.append(level)

    if selected_levels:
        return sum(selected_levels) / len(selected_levels)

    return interpolate_log_spectrum(
        frequencies,
        levels,
        target_frequency
    )


def median_in_log_window(
    frequencies: list[float],
    values: list[float],
    target_frequency: float,
    window_octaves: float
) -> float:
    """
    Медиана значений в окне вокруг частоты.

    Используется для определения локального уровня,
    чтобы узкий пик не вызвал чрезмерное усиление.
    """

    selected_values = []

    for frequency, value in zip(frequencies, values):
        distance_octaves = abs(
            math.log2(frequency / target_frequency)
        )

        if distance_octaves <= window_octaves:
            selected_values.append(value)

    if not selected_values:
        return 0.0

    return statistics.median(selected_values)


def gaussian_smooth_log_curve(
    frequencies: list[float],
    values: list[float],
    sigma_octaves: float
) -> list[float]:
    """
    Сглаживает кривую гауссовым фильтром по логарифму частоты.

    sigma_octaves:
        ширина сглаживания в октавах.
        Например:
            0.20 — слабое сглаживание
            0.35 — умеренное
            0.50 — сильное
    """

    if sigma_octaves <= 0:
        return list(values)

    smoothed = []

    for target_frequency in frequencies:
        weighted_sum = 0.0
        weight_sum = 0.0

        for frequency, value in zip(frequencies, values):
            distance_octaves = math.log2(
                frequency / target_frequency
            )

            weight = math.exp(
                -0.5
                * (distance_octaves / sigma_octaves) ** 2
            )

            weighted_sum += value * weight
            weight_sum += weight

        if weight_sum == 0:
            smoothed.append(values[frequencies.index(target_frequency)])
        else:
            smoothed.append(weighted_sum / weight_sum)

    return smoothed


# ============================================================
# Расчёт фонового уровня и процентилей
# ============================================================

def get_baseline(
    frequencies: list[float],
    levels: list[float],
    profile: dict
) -> float:
    """
    Определяет фоновый уровень одного спектра.

    Предпочтительно используется диапазон baseline_band.
    """

    low_frequency, high_frequency = profile[
        "baseline_band"
    ]

    band_levels = [
        level
        for frequency, level in zip(frequencies, levels)
        if low_frequency <= frequency <= high_frequency
    ]

    if band_levels:
        return statistics.median(band_levels)

    return statistics.median(levels)


def get_percentile_prominence(
    spectra: list[tuple[list[float], list[float]]],
    target_frequency: float,
    profile: dict
) -> tuple[float, float, list[float]]:
    """
    Для одной контрольной частоты:

    1. вычисляет относительное превышение над фоном
       для каждого входного спектра;
    2. берёт высокий процентиль;
    3. возвращает также средний абсолютный уровень
       и список отдельных значений.

    Нормализация относительно собственного фона каждого
    спектра позволяет объединять записи с разной общей громкостью.
    """

    relative_prominences = []
    absolute_levels = []

    for frequencies, levels in spectra:
        baseline = get_baseline(
            frequencies,
            levels,
            profile
        )

        spectrum_level = smooth_spectrum_at(
            frequencies,
            levels,
            target_frequency
        )

        prominence = max(
            0.0,
            spectrum_level - baseline
        )

        relative_prominences.append(prominence)
        absolute_levels.append(spectrum_level)

    percentile_value = float(
        profile.get("spectrum_percentile", 90.0)
    )

    high_percentile = percentile(
        relative_prominences,
        percentile_value
    )

    absolute_percentile = percentile(
        absolute_levels,
        percentile_value
    )

    return (
        high_percentile,
        absolute_percentile,
        relative_prominences
    )


# ============================================================
# Основной расчёт маскирующей кривой
# ============================================================

def create_gain_curve(
    spectra: list[tuple[list[float], list[float]]],
    profile: dict
):
    """
    Создаёт кривую усиления для Audacity.

    Используются:

    - несколько исходных спектров;
    - высокий процентиль;
    - защита от узких пиков;
    - сглаживание кривой;
    - компенсация АЧХ колонки.
    """

    preset_frequencies = [
        float(value)
        for value in profile["preset_frequencies"]
    ]

    speaker_response = profile[
        "speaker_response"
    ]

    masking_margin = float(
        profile.get("masking_margin_db", 3.0)
    )

    minimum_mask = float(
        profile.get("minimum_mask_db", 2.0)
    )

    level_offset = float(
        profile.get("level_offset_db", 0.0)
    )

    spectrum_percentile = float(
        profile.get("spectrum_percentile", 90.0)
    )

    # Ограничение влияния узких пиков.
    #
    # Например, значение 4 дБ означает:
    # узкий пик не сможет дать больше чем примерно
    # local_median + 4 дБ.
    narrow_peak_limit = float(
        profile.get("narrow_peak_limit_db", 4.0)
    )

    # Радиус поиска локальной медианы для определения
    # узкого пика.
    peak_detection_window = float(
        profile.get("peak_detection_window_octaves", 0.35)
    )

    # Сглаживание итоговой кривой.
    curve_smoothing = float(
        profile.get("curve_smoothing_octaves", 0.30)
    )

    minimum_gain = float(
        profile.get("min_gain", -24.0)
    )

    maximum_gain = float(
        profile.get("max_gain", 4.0)
    )

    # Сначала вычисляем 90-й процентиль
    # относительного превышения для каждой частоты.
    raw_prominences = []
    absolute_percentile_levels = []
    all_relative_values = []

    for target_frequency in preset_frequencies:
        prominence_percentile, absolute_level, values = (
            get_percentile_prominence(
                spectra,
                target_frequency,
                profile
            )
        )

        raw_prominences.append(
            prominence_percentile
        )

        absolute_percentile_levels.append(
            absolute_level
        )

        all_relative_values.append(values)

    # Защита от узких пиков.
    #
    # Если значение заметно выше локальной медианы,
    # оно ограничивается. Широкий подъём спектра при этом
    # сохраняется, потому что локальная медиана тоже поднимается.
    limited_prominences = []

    for index, target_frequency in enumerate(
        preset_frequencies
    ):
        local_median = median_in_log_window(
            preset_frequencies,
            raw_prominences,
            target_frequency,
            peak_detection_window
        )

        limited_value = min(
            raw_prominences[index],
            local_median + narrow_peak_limit
        )

        limited_prominences.append(
            limited_value
        )

    # Формируем предварительную кривую маскирования.
    raw_masking_curve = []

    for prominence in limited_prominences:
        desired_mask_level = max(
            minimum_mask,
            prominence + masking_margin
        )

        raw_masking_curve.append(
            desired_mask_level
        )

    # Сглаживаем кривую маскирования по логарифму частоты.
    smooth_masking_curve = gaussian_smooth_log_curve(
        preset_frequencies,
        raw_masking_curve,
        curve_smoothing
    )

    # Компенсируем АЧХ колонки.
    raw_gains = []
    diagnostics = []

    for index, target_frequency in enumerate(
        preset_frequencies
    ):
        speaker_level = interpolate_log_curve(
            speaker_response,
            target_frequency
        )

        desired_mask_level = smooth_masking_curve[
            index
        ]

        raw_gain = (
            desired_mask_level
            - speaker_level
            + level_offset
        )

        raw_gains.append(raw_gain)

        diagnostics.append({
            "frequency": target_frequency,
            "percentile_level": (
                absolute_percentile_levels[index]
            ),
            "raw_prominence": (
                raw_prominences[index]
            ),
            "limited_prominence": (
                limited_prominences[index]
            ),
            "masking_level": (
                desired_mask_level
            ),
            "speaker_response": speaker_level,
            "raw_gain": raw_gain
        })

    # Убираем общий уровень.
    #
    # Это превращает результат именно в EQ-кривую,
    # а не в неконтролируемое общее усиление.
    reference_gain = statistics.median(
        raw_gains
    )

    gains = []

    for index, raw_gain in enumerate(raw_gains):
        gain = raw_gain - reference_gain

        gain = clamp(
            gain,
            minimum_gain,
            maximum_gain
        )

        gains.append(gain)

        diagnostics[index]["final_gain"] = gain

    # Дополнительное финальное сглаживание уже после
    # компенсации АЧХ колонки.
    #
    # Это особенно полезно, если АЧХ колонки задана
    # редкими измерительными точками.
    gains = gaussian_smooth_log_curve(
        preset_frequencies,
        gains,
        curve_smoothing
    )

    # Повторное ограничение после сглаживания.
    for index, gain in enumerate(gains):
        gains[index] = clamp(
            gain,
            minimum_gain,
            maximum_gain
        )

        diagnostics[index]["final_gain"] = gains[index]

    return (
        preset_frequencies,
        gains,
        diagnostics,
        raw_prominences,
        limited_prominences
    )


# ============================================================
# Формирование пресета Audacity
# ============================================================

def format_number(value: float) -> str:
    """
    Формат Audacity использует точку как десятичный разделитель.
    """

    return f"{value:.12g}"


def build_audacity_preset(
    profile: dict,
    frequencies: list[float],
    gains: list[float]
) -> str:

    parts = ["FilterCurve:"]

    for index, frequency in enumerate(frequencies):
        parts.append(
            f'f{index}="{format_number(frequency)}"'
        )

    filter_length = profile.get(
        "filter_length",
        8191
    )

    interpolate_lin = profile.get(
        "interpolate_lin",
        0
    )

    interpolation_method = profile.get(
        "interpolation_method",
        "B-spline"
    )

    parts.extend([
        f'FilterLength="{filter_length}"',
        f'InterpolateLin="{interpolate_lin}"',
        f'InterpolationMethod="{interpolation_method}"'
    ])

    for index, gain in enumerate(gains):
        parts.append(
            f'v{index}="{format_number(gain)}"'
        )

    return " ".join(parts)


# ============================================================
# Диагностический отчёт
# ============================================================

def build_report(
    profile: dict,
    input_files: list[str],
    diagnostics: list[dict]
) -> str:

    lines = [
        "Adaptive Audacity Masking Preset",
        "",
        f"Колонка: {profile['name']}",
        f"Количество спектров: {len(input_files)}",
        f"Процентиль: "
        f"{profile.get('spectrum_percentile', 90)}%",
        f"Запас маскирования: "
        f"{profile.get('masking_margin_db', 3.0):.2f} дБ",
        f"Минимальный уровень маскирования: "
        f"{profile.get('minimum_mask_db', 2.0):.2f} дБ",
        f"Ограничение узких пиков: "
        f"{profile.get('narrow_peak_limit_db', 4.0):.2f} дБ",
        f"Сглаживание: "
        f"{profile.get('curve_smoothing_octaves', 0.30):.2f} октавы",
        "",
        "Исходные спектры:"
    ]

    for filename in input_files:
        lines.append(
            f"  - {Path(filename).name}"
        )

    lines.extend([
        "",
        "Частота | P90 | Пик | Огранич. | "
        "Маскир. | АЧХ | Итог",
        "-" * 86
    ])

    for item in diagnostics:
        lines.append(
            f"{item['frequency']:7.0f} | "
            f"{item['percentile_level']:6.2f} | "
            f"{item['raw_prominence']:5.2f} | "
            f"{item['limited_prominence']:8.2f} | "
            f"{item['masking_level']:7.2f} | "
            f"{item['speaker_response']:5.2f} | "
            f"{item['final_gain']:5.2f}"
        )

    lines.extend([
        "",
        "Обозначения:",
        "P90       — 90-й процентиль уровня",
        "Пик       — необработанное превышение над фоном",
        "Огранич.  — значение после ограничения узкого пика",
        "Маскир.   — сглаженный целевой уровень",
        "АЧХ       — относительный уровень колонки",
        "Итог      — значение для Filter Curve EQ"
    ])

    return "\n".join(lines)


# ============================================================
# Графический запуск
# ============================================================

def main():
    root = tk.Tk()
    root.withdraw()

    try:
        # ----------------------------------------------------
        # Выбор профиля колонки
        # ----------------------------------------------------

        profile_file = filedialog.askopenfilename(
            title="Выберите профиль колонки",
            filetypes=[
                ("JSON-профили", "*.json"),
                ("Все файлы", "*.*")
            ]
        )

        if not profile_file:
            return

        profile = load_profile(profile_file)

        # ----------------------------------------------------
        # Выбор нескольких спектров
        # ----------------------------------------------------

        input_files = filedialog.askopenfilenames(
            title=(
                "Выберите один или несколько TXT-файлов "
                "со спектрами"
            ),
            filetypes=[
                ("TXT-файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )

        if not input_files:
            return

        spectra = []

        for filename in input_files:
            frequencies, levels = read_spectrum(
                filename
            )

            spectra.append(
                (frequencies, levels)
            )

        # ----------------------------------------------------
        # Расчёт кривой
        # ----------------------------------------------------

        (
            preset_frequencies,
            gains,
            diagnostics,
            raw_prominences,
            limited_prominences
        ) = create_gain_curve(
            spectra,
            profile
        )

        preset = build_audacity_preset(
            profile,
            preset_frequencies,
            gains
        )

        # ----------------------------------------------------
        # Сохранение
        # ----------------------------------------------------

        safe_name = (
            profile["name"]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        output_file = filedialog.asksaveasfilename(
            title="Сохранить пресет Audacity",
            initialfile=(
                safe_name
                + "_Adaptive_Masking_Preset.txt"
            ),
            defaultextension=".txt",
            filetypes=[
                ("Audacity Filter Curve", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )

        if not output_file:
            return

        output_path = Path(output_file)

        output_path.write_text(
            preset + "\n",
            encoding="utf-8"
        )

        report_path = output_path.with_name(
            output_path.stem + "_report.txt"
        )

        report_path.write_text(
            build_report(
                profile,
                list(input_files),
                diagnostics
            ),
            encoding="utf-8"
        )

        messagebox.showinfo(
            "Готово",
            f"Пресет создан.\n\n"
            f"Колонка: {profile['name']}\n"
            f"Спектров обработано: {len(input_files)}\n\n"
            f"Пресет:\n{output_path}\n\n"
            f"Отчёт:\n{report_path}"
        )

    except Exception as error:
        messagebox.showerror(
            "Ошибка",
            str(error)
        )

    finally:
        root.destroy()


if __name__ == "__main__":
    main()
