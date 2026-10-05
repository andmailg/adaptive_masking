import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def find_noise_peaks(spectrum_file_path: str) -> dict[int, float]:
    """Считывает TXT-файл спектра из Audacity и находит пики в целевых окнах баса."""
    peaks = {45: -120.0, 69: -120.0, 106: -120.0}
    path = Path(spectrum_file_path)
    
    with open(path, "r", encoding="utf-8") as file:
        next(file, None)  # Пропускаем заголовок Audacity
        for line in file:
            if not line.strip():
                continue
            try:
                freq_str, level_str = line.split()
                freq = float(freq_str)
                level = float(level_str)
                
                # ИСПРАВЛЕНО: Сравниваем level с конкретным элементом словаря peaks[key]
                if 35.0 <= freq <= 55.0 and level > peaks[45]:
                    peaks[45] = level
                elif 64.0 <= freq <= 78.0 and level > peaks[69]:
                    peaks[69] = level
                elif 95.0 <= freq <= 115.0 and level > peaks[106]:
                    peaks[106] = level
            except ValueError:
                continue
                
    return {f: round(l, 2) for f, l in peaks.items()}


def interpolate_speaker_response(speaker_data: dict, target_freq: float) -> float:
    """Линейно интерполирует АЧХ колонки для промежуточных микро-частот."""
    if target_freq in speaker_data:
        return speaker_data[target_freq]
        
    frequencies = sorted(speaker_data.keys())
    if target_freq < frequencies[0]:
        return speaker_data[frequencies[0]]
    if target_freq > frequencies[-1]:
        return speaker_data[frequencies[-1]]
        
    for i in range(len(frequencies) - 1):
        f0, f1 = frequencies[i], frequencies[i+1]
        if f0 <= target_freq <= f1:
            v0, v1 = speaker_data[f0], speaker_data[f1]
            return v0 + (v1 - v0) * (target_freq - f0) / (f1 - f0)
    return -12.0


def get_all_spectrum_peaks(file_paths: list[str]) -> set[int]:
    """Вспомогательная функция агрегации пиков из множества файлов."""
    detected_frequencies = set()
    for file_path in file_paths:
        if Path(file_path).exists():
            peaks = find_noise_peaks(file_path)
            detected_frequencies.add(max(peaks, key=peaks.get))
    return detected_frequencies


def build_high_density_preset(profile_path: str, spectrum_paths: list[str]) -> tuple[str, list[int], list[str]]:
    """Динамически уплотняет сетку частот и проверяет безопасность усиления."""
    base_peaks = get_all_spectrum_peaks(spectrum_paths)
    dense_frequencies = set()
    warnings = []
    
    for peak_freq in base_peaks:
        harmonic = peak_freq * 2
        for offset in range(-20, 25, 4):  # Высокоплотный шаг 4 Гц
            check_freq = harmonic + offset
            if 55 <= check_freq <= 160:
                dense_frequencies.add(check_freq)

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    speaker_response = {int(k): v for k, v in profile.get("speaker_response", [])}
    masking_margin = profile.get("masking_margin_db", 3.0)
    min_gain = profile.get("min_gain", -24.0)
    max_gain = profile.get("max_gain", 6.0)
    filter_length = profile.get("filter_length", 8191)

    preset_points = [(20, -27.0), (30, -24.0)]
    for peak_freq in base_peaks:
        preset_points.append((peak_freq, -27.0))

    for freq in dense_frequencies:
        speaker_efficiency = interpolate_speaker_response(speaker_response, freq)
        dist_to_harmonic = min([abs(freq - (p * 2)) for p in base_peaks])
        center_bonus = max(0.0, 5.0 - (dist_to_harmonic * 0.2))
        
        calculated_gain = (0.0 - speaker_efficiency) + masking_margin + center_bonus
        
        if calculated_gain > max_gain:
            warnings.append(
                f"Частота {freq} Гц: требуется {round(calculated_gain, 1)} дБ, "
                f"урезано до {max_gain} дБ"
            )
            
        final_gain = max(min_gain, min(max_gain, calculated_gain))
        preset_points.append((freq, round(final_gain, 1)))

    preset_points.extend([
        (200, -6.0), (315, -12.0), (500, -18.0), (1000, -24.0), (20000, -24.0)
    ])

    preset_points = list(set(preset_points))
    preset_points.sort(key=lambda x: x)

    parts = ["FilterCurve:"]
    for index, (freq, _) in enumerate(preset_points):
        parts.append(f'f{index}="{freq}.0"')
        
    parts.extend([
        f'FilterLength="{filter_length}"',
        'InterpolateLin="0"',
        'InterpolationMethod="B-spline"'
    ])
    
    for index, (_, level) in enumerate(preset_points):
        parts.append(f'v{index}="{level}"')
        
    return " ".join(parts), sorted(list(base_peaks)), warnings


class SecureMaskingApp(tk.Tk):
    """Графический интерфейс с системой безопасности динамиков."""
    def __init__(self):
        super().__init__()
        self.title("Высокоплотный генератор маскировки (Secure)")
        self.geometry("640x480")
        self.minsize(580, 420)

        self.json_path = tk.StringVar()
        self.spectrum_files = []
        self.generated_preset = ""

        self._create_widgets()
        
    def _create_widgets(self):
        file_frame = ttk.LabelFrame(self, text=" Файлы конфигурации ", padding=10)
        file_frame.pack(fill="x", padx=15, pady=15)

        ttk.Label(file_frame, text="Профиль колонки (JSON):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.json_path, width=42).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="Спектры шума (TXT):").grid(row=1, column=0, sticky="w", pady=5)
        self.files_label = ttk.Label(file_frame, text="Файлы не выбраны", foreground="gray", width=42, anchor="w")
        self.files_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(file_frame, text="Выбрать...", command=self._browse_spectra).grid(row=1, column=2, padx=5, pady=5)

        ttk.Button(self, text="Рассчитать безопасный пресет", command=self._process_data).pack(pady=5)

        output_frame = ttk.LabelFrame(self, text=" Мониторинг безопасности и гармоник ", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.log_text = tk.Text(output_frame, wrap="word", height=8, bg="#f8f9fa", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.save_btn = ttk.Button(self, text="Экспортировать пресет в файл...", command=self._save_file, state="disabled")
        self.save_btn.pack(pady=15)

    def _browse_json(self):
        filename = filedialog.askopenfilename(
            title="Открыть JSON-профиль колонки",
            filetypes=[("Профили акустики", "*.json"), ("Все файлы", "*.*")]
        )
        if filename:
            self.json_path.set(filename)

    def _browse_spectra(self):
        filenames = filedialog.askopenfilenames(
            title="Выберите файлы спектра соседа (TXT)",
            filetypes=[("Спектры Audacity", "*.txt"), ("Все файлы", "*.*")]
        )
        if filenames:
            self.spectrum_files = list(filenames)
            self.files_label.config(text=f"Выбрано файлов: {len(filenames)}", foreground="black")

    def _update_log(self, text: str, clear=False):
        self.log_text.config(state="normal")
        if clear:
            self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.config(state="disabled")

    def _process_data(self):
        if not self.json_path.get() or not self.spectrum_files:
            messagebox.showwarning("Внимание", "Выберите профиль JSON и файлы спектра!")
            return

        try:
            self.generated_preset, base_peaks, warnings = build_high_density_preset(
                self.json_path.get(), self.spectrum_files
            )

            self._update_log("== МОНИТОРИНГ ЗАВЕРШЕН УСПЕШНО ==\n", clear=True)
            self._update_log(f"Обнаруженные базовые резонансы соседа: {base_peaks} Гц\n\n")
            
            if warnings:
                self._update_log("⚠️ ВНИМАНИЕ: ОБНАРУЖЕНА ПЕРЕГРУЗКА ДИНАМИКОВ!\n")
                self._update_log("Математический расчет превысил безопасный лимит max_gain.\n")
                self._update_log("Сигнал был автоматически срезан для предотвращения хрипов:\n")
                for w in warnings:
                    self._update_log(f"  - {w}\n")
                self._update_log("\nРекомендация: Если маскировки не хватает, физически переместите ")
                self._update_log("колонку Marshall ближе к потолку / источнику шума.\n\n")
            else:
                self._update_log("✅ КОНТРОЛЬ БЕЗОПАСНОСТИ ПРОЙДЕН:\n")
                self._update_log("Все расчетные частоты лежат в пределах безопасного max_gain.\n\n")

            self._update_log("Пресет успешно собран. Можно выгружать файл.")
            self.save_btn.config(state="normal")
            
        except Exception as error:
            messagebox.showerror("Ошибка вычислений", f"Произошла ошибка: {str(error)}")

    def _save_file(self):
        if not self.generated_preset:
            return

        output_file = filedialog.asksaveasfilename(
            title="Сохранить безопасный пресет",
            initialfile="Marshall_III_Secure_Preset.txt",
            defaultextension=".txt",
            filetypes=[
                ("Audacity Filter Curve", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )

        if not output_file:
            return

        try:
            Path(output_file).write_text(
                self.generated_preset + "\n",
                encoding="utf-8"
            )
            messagebox.showinfo(
                "Успех",
                f"Безопасный пресет сохранен:\n"
                f"{os.path.basename(output_file)}"
            )
        except Exception as error:
            messagebox.showerror(
                "Ошибка записи",
                f"Не удалось сохранить файл пресета: {str(error)}"
            )


if __name__ == "__main__":
    app = SecureMaskingApp()
    app.mainloop()