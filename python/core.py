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


def interpolate_speaker_response(speaker_data: dict, target_freq: float) -> float:
    """Линейно интерполирует АЧХ колонки для промежуточных микро-частот."""
    if not speaker_data:
        return -12.0
        
    if target_freq in speaker_data:
        return speaker_data[target_freq]

    frequencies = sorted(speaker_data.keys())
    if target_freq < frequencies[0]:
        return speaker_data[frequencies[0]]
    if target_freq > frequencies[-1]:
        return speaker_data[frequencies[-1]]

    for i in range(len(frequencies) - 1):
        f0, f1 = frequencies[i], frequencies[i + 1]
        if f0 <= target_freq <= f1:
            v0, v1 = speaker_data[f0], speaker_data[f1]
            return v0 + (v1 - v0) * (target_freq - f0) / (f1 - f0)
    return -12.0


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
    speaker_response: dict | None = None,
):
    """Строит монолитную форму психоакустического купола без провалов и ступенек."""
    base_peaks, peaks_details = get_all_spectrum_peaks_dynamic(
        spectrum_paths, min_db_threshold
    )
    
    dense_frequencies = set()

    if base_peaks:
        min_peak = min(base_peaks)
        max_peak = max(base_peaks)
        start_dome = max(40, min_peak - 10)
        end_dome = min(250, (max_peak * 3) + 20)
        for freq in range(start_dome, end_dome + 1, 1):  # Шаг в 1 Гц для максимальной плотности
            dense_frequencies.add(freq)
    else:
        for freq in range(55, 251, 1):
            dense_frequencies.add(freq)

    if speaker_response is None:
        if profile_path is None:
            raise ValueError("profile_path must be provided when speaker_response is not given")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        speaker_response = {
            int(k): v for k, v in profile.get("speaker_response", [])
        }

    preset_points = []
    raw_points = {}
    base_zone_gains = []

    for freq in sorted(list(dense_frequencies)):
        speaker_efficiency = interpolate_speaker_response(speaker_response, freq)
        center_bonus = 0.0
        is_pure_base = False

        for p in base_peaks:
            speaker_eff_at_p = interpolate_speaker_response(speaker_response, p)
            is_infra_for_speaker = (speaker_eff_at_p < -8.0)
            
            dist_f1 = abs(freq - p)
            dist_f2 = abs(freq - (p * 2))
            dist_f3 = abs(freq - (p * 3))
            
            # Применяем Cosine Bell плавное распределение
            if not is_infra_for_speaker:
                b1 = max(0.0, 6.0 * (1.0 + math.cos(min(dist_f1, 20) / 20.0 * math.pi)) / 2.0)
                if b1 > 0:
                    is_pure_base = True
                b2 = max(0.0, h2_gain * (1.0 + math.cos(min(dist_f2, 25) / 25.0 * math.pi)) / 2.0)
                center_bonus = max(center_bonus, b1, b2)
            else:
                b1 = 0.0 
                b2 = max(0.0, h2_gain * (1.0 + math.cos(min(dist_f2, 25) / 25.0 * math.pi)) / 2.0)
                b3 = max(0.0, h3_gain * (1.0 + math.cos(min(dist_f3, 30) / 30.0 * math.pi)) / 2.0)
                center_bonus = max(center_bonus, b1, b2, b3)

        calculated_gain = (0.0 - speaker_efficiency) + masking_margin + center_bonus
        raw_points[freq] = calculated_gain
        
        if is_pure_base or (not base_peaks and 55 <= freq <= 85):
            base_zone_gains.append(calculated_gain)

    unlimited_base_max = max(base_zone_gains) if base_zone_gains else max(raw_points.values()) if raw_points else 0.0
    global_reduction = max(0.0, unlimited_base_max - max_peak_limit)

    active_gains = {}
    for freq, raw_gain in raw_points.items():
        normalized_gain = raw_gain - global_reduction
        final_gain = max(min_gain, normalized_gain)
        preset_points.append((freq, round(final_gain, 1)))
        active_gains[freq] = final_gain

    first_active_freq = min(active_gains.keys()) if active_gains else 55
    first_active_gain = active_gains[first_active_freq] if active_gains else min_gain
    last_active_freq = max(active_gains.keys()) if active_gains else 250
    last_active_gain = active_gains[last_active_freq] if active_gains else min_gain

    # 1. СГЛАЖИВАНИЕ СЛЕВА
    preset_points.append((20, min_gain))
    if first_active_freq > 20 and first_active_gain > min_gain:
        log_start_l = math.log10(20)
        log_end_l = math.log10(first_active_freq)
        steps_l = 30
        for i in range(1, steps_l):
            ratio = i / steps_l
            log_f = log_start_l + ratio * (log_end_l - log_start_l)
            f = int(round(10**log_f))
            if 20 < f < first_active_freq:
                smooth_factor = (1.0 - math.cos(ratio * math.pi)) / 2.0
                f_gain = min_gain + smooth_factor * (first_active_gain - min_gain)
                preset_points.append((f, round(f_gain, 1)))

    # 2. СГЛАЖИВАНИЕ СПРАВА
    start_r_freq = last_active_freq + 1
    if start_r_freq < 20000 and last_active_gain > min_gain:
        log_start_r = math.log10(start_r_freq)
        log_end_r = math.log10(20000)
        steps_r = 100
        for i in range(0, steps_r + 1):
            ratio = i / steps_r
            log_f = log_start_r + ratio * (log_end_r - log_start_r)
            f = int(round(10**log_f))
            if f > last_active_freq and f <= 20000:
                smooth_factor = (1.0 - math.cos(ratio * math.pi)) / 2.0
                f_gain = last_active_gain - smooth_factor * (last_active_gain - min_gain)
                preset_points.append((f, round(f_gain, 1)))
    else:
        preset_points.append((20000, min_gain))

    unique_points = {}
    for freq, gain in preset_points:
        if freq not in unique_points or gain > unique_points[freq]:
            unique_points[freq] = gain

    preset_points = sorted(list(unique_points.items()), key=lambda x: x)
    return preset_points, sorted(list(base_peaks)), global_reduction, filter_length, peaks_details
