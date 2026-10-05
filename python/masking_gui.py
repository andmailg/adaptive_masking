import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def find_average_noise_peaks(file_paths: list[str]) -> dict[int, float]:
    """Читает несколько файлов Audacity и возвращает усредненные пики шума."""
    total_peaks = {45: [], 69: [], 106: []}

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        current_max = {45: -120.0, 69: -120.0, 106: -120.0}

        with open(path, "r", encoding="utf-8") as file:
            next(file, None)  # Пропускаем заголовок
            for line in file:
                if not line.strip():
                    continue
                try:
                    freq_str, level_str = line.split()
                    freq = float(freq_str)
                    level = float(level_str)

                    if 35.0 <= freq <= 55.0 and level > current_max[45]:
                        current_max[45] = level
                    elif 64.0 <= freq <= 78.0 and level > current_max[69]:
                        current_max[69] = level
                    elif 95.0 <= freq <= 115.0 and level > current_max[106]:
                        current_max[106] = level
                except ValueError:
                    continue

        for freq in total_peaks:
            if current_max[freq] > -120.0:
                total_peaks[freq].append(current_max[freq])

    # Рассчитываем среднее арифметическое для каждого пика
    avg_peaks = {}
    for freq, levels in total_peaks.items():
        if levels:
            avg_peaks[freq] = round(sum(levels) / len(levels), 2)
        else:
            avg_peaks[freq] = -90.0  # Дефолтное значение тишины

    return avg_peaks


def build_fully_adaptive_preset(profile_path: str, spectrum_paths: list[str]) -> tuple[str, dict]:
    """Динамически связывает усредненные пики шума и параметры АЧХ из JSON."""
    neighbor_peaks = find_average_noise_peaks(spectrum_paths)
    primary_noise_freq = max(neighbor_peaks, key=neighbor_peaks.get)
    harmonic_freq = int(round(primary_noise_freq * 2))

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    speaker_response = dict(profile.get("speaker_response", []))
    masking_margin = profile.get("masking_margin_db", 3.0)
    min_gain = profile.get("min_gain", -24.0)
    max_gain = profile.get("max_gain", 6.0)
    filter_length = profile.get("filter_length", 8191)

    preset_points = [
        (20, -27.0),
        (30, -24.0),
        (int(round(primary_noise_freq)), -27.0)
    ]

    adaptive_masking_map = {
        harmonic_freq - 30: masking_margin + 1.5,
        harmonic_freq - 10: masking_margin + 5.0,
        harmonic_freq:      masking_margin + 4.5,
        harmonic_freq + 10: masking_margin + 1.0,
        harmonic_freq + 35: masking_margin - 2.0
    }

    for freq, boost in adaptive_masking_map.items():
        if freq < 50:
            continue
        closest_freq = min(speaker_response.keys(), key=lambda x: abs(x - freq))
        speaker_efficiency = speaker_response.get(closest_freq, -12.0)

        gain = (0.0 - speaker_efficiency) + boost
        gain = max(min_gain, min(max_gain, gain))
        preset_points.append((closest_freq, round(gain, 1)))

    preset_points.extend([
        (200, -6.0),
        (315, -12.0),
        (500, -18.0),
        (1000, -24.0),
        (20000, -24.0)
    ])

    preset_points = list(set(preset_points))
    preset_points.sort(key=lambda x: x)

    parts = ["FilterCurve:"]
    for index, (freq, _) in enumerate(preset_points):
        parts.append(f'f{index}="{freq}.0"')
    parts.extend([f'FilterLength="{filter_length}"', 'InterpolateLin="0"', 'InterpolationMethod="B-spline"'])
    for index, (_, level) in enumerate(preset_points):
        parts.append(f'v{index}="{level}"')

    return " ".join(parts), neighbor_peaks


class AdaptiveMaskingApp(tk.Tk):
    """Интерфейс Tkinter с поддержкой множественного выбора файлов спектров."""
    def __init__(self):
        super().__init__()
        self.title("Адаптивный мульти-генератор маскировки")
        self.geometry("620x460")
        self.minsize(580, 420)

        self.json_path = tk.StringVar()
        self.spectrum_files = []

        self._create_widgets()

    def _create_widgets(self):
        file_frame = ttk.LabelFrame(self, text=" Конфигурационные файлы ", padding=10)
        file_frame.pack(fill="x", padx=15, pady=15)

        ttk.Label(file_frame, text="Профиль колонки (JSON):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.json_path, width=42).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="Спектры шума (TXT):").grid(row=1, column=0, sticky="w", pady=5)
        self.files_label = ttk.Label(file_frame, text="Файлы не выбраны", foreground="gray", width=42, anchor="w")
        self.files_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(file_frame, text="Выбрать...", command=self._browse_spectra).grid(row=1, column=2, padx=5, pady=5)

        ttk.Button(self, text="Рассчитать адаптивный пресет", command=self._process_data).pack(pady=5)

        output_frame = ttk.LabelFrame(self, text=" Монитор усредненных параметров ", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.log_text = tk.Text(output_frame, wrap="word", height=7, bg="#f8f9fa", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.save_btn = ttk.Button(self, text="Экспортировать TXT-пресет...", command=self._save_file, state="disabled")
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
            title="Выберите один или несколько файлов спектра (TXT)",
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
            messagebox.showwarning("Внимание", "Выберите JSON-профиль и хотя бы один файл спектра!")
            return

        try:
            self.generated_preset, peaks = build_fully_adaptive_preset(self.json_path.get(), self.spectrum_files)

            primary_freq = max(peaks, key=peaks.get)
            harmonic_target = int(round(primary_freq * 2))

            self._update_log("== АНАЛИЗ УСРЕДНЕННОГО СПЕКТРА И АЧХ ==\n", clear=True)
            self._update_log(f"Усредненный пик гула:      {primary_freq} Гц ({peaks[primary_freq]} дБ)\n")
            self._update_log(f"Усредненный второй пик:    {69} Гц ({peaks[69]} дБ)\n")
            self._update_log(f"Усредненный третий пик:    {106} Гц ({peaks[106]} дБ)\n\n")
            self._update_log(f"Целевая гармоника маскировки: {harmonic_target} Гц\n")
            self._update_log("Данные успешно агрегированы. Пресет готов к записи в файл.")

            self.save_btn.config(state="normal")
        except Exception as error:
            messagebox.showerror("Ошибка", f"Ошибка во время адаптивного расчета: {str(error)}")

    def _save_file(self):
        if not self.generated_preset:
            return

        output_file = filedialog.asksaveasfilename(
            title="Экспорт пресета Audacity",
            initialfile="Marshall_III_Multi_Adaptive_Curve.txt",
            defaultextension=".txt",
            filetypes=[("Audacity Filter Curve", "*.txt"), ("Все файлы", "*.*")]
        )

        if not output_file:
            return

        try:
            Path(output_file).write_text(self.generated_preset + "\n", encoding="utf-8")
            messagebox.showinfo("Успех", f"Файл пресета сохранен:\n{os.path.basename(output_file)}")
        except Exception as error:
            messagebox.showerror("Ошибка записи", f"Не удалось сохранить пресет: {str(error)}")


if __name__ == "__main__":
    app = AdaptiveMaskingApp()
    app.mainloop()
