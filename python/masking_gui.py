import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def get_all_spectrum_peaks(file_paths: list[str]) -> set[int]:
    """Сканирует все файлы и находит точные частоты локальных пиков шума соседа."""
    detected_frequencies = set()

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        current_max_bass = -120.0
        best_bass_freq = 45  # дефолт

        current_max_mid = -120.0
        best_mid_freq = 69  # дефолт

        with open(path, "r", encoding="utf-8") as file:
            next(file, None)  # Пропускаем заголовок Audacity
            for line in file:
                if not line.strip():
                    continue
                try:
                    freq_str, level_str = line.split()
                    freq = float(freq_str)
                    level = float(level_str)

                    # Ловим точную частоту основного резонанса (в районе 35-55 Гц)
                    if 35.0 <= freq <= 55.0 and level > current_max_bass:
                        current_max_bass = level
                        best_bass_freq = int(round(freq))
                    
                    # Ловим точную частоту второго резонанса (в районе 60-80 Гц)
                    if 60.0 <= freq <= 80.0 and level > current_max_mid:
                        current_max_mid = level
                        best_mid_freq = int(round(freq))
                except ValueError:
                    continue

        detected_frequencies.add(best_bass_freq)
        detected_frequencies.add(best_mid_freq)

    return detected_frequencies


def interpolate_speaker_response(speaker_data: dict, target_freq: float) -> float:
    """Линейно интерполирует АЧХ колонки для промежуточных микро-частот."""
    if target_freq in speaker_data:
        return speaker_data[target_freq]
        
    frequencies = sorted(speaker_data.keys())
    if target_freq < frequencies[0]:
        return speaker_data[frequencies[0]]
    if target_freq > frequencies[-1]:
        return speaker_data[frequencies[-1]]
        
    # Ищем две ближайшие заводские точки для интерполяции
    for i in range(len(frequencies) - 1):
        f0, f1 = frequencies[i], frequencies[i+1]
        if f0 <= target_freq <= f1:
            v0, v1 = speaker_data[f0], speaker_data[f1]
            # Формула линейной интерполяции
            return v0 + (v1 - v0) * (target_freq - f0) / (f1 - f0)
    return -12.0


def build_high_density_preset(profile_path: str, spectrum_paths: list[str]) -> tuple[str, list[int]]:
    """Динамически уплотняет сетку частот баса на основе нескольких спектров."""
    # 1. Находим точные частоты резонансов по всем файлам
    base_peaks = get_all_spectrum_peaks(spectrum_paths)
    
    # Сюда мы будем собирать наши новые высокоплотные частотные точки
    dense_frequencies = set()
    
    # Для каждого найденного пика генерируем плотный шаг (через каждые 3-5 Гц) в зоне его гармоник
    for peak_freq in base_peaks:
        harmonic = peak_freq * 2  # Рассчитываем психоакустическую мишень (гармонику)
        
        # Создаем плотное облако точек вокруг гармоники соседа для плавности фильтра Audacity
        for offset in range(-20, 25, 4):  # Шаг уплотнения сетки — 4 Гц!
            check_freq = harmonic + offset
            if 55 <= check_freq <= 160:   # Ограничиваем только рабочим басовым регистром
                dense_frequencies.add(check_freq)

    # 2. Загружаем JSON профиль Marshall Emberton III
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    speaker_response = dict(profile.get("speaker_response", []))
    masking_margin = profile.get("masking_margin_db", 3.0)
    min_gain = profile.get("min_gain", -24.0)
    max_gain = profile.get("max_gain", 6.0)
    filter_length = profile.get("filter_length", 8191)

    # Базовые точки спада инфразвука
    preset_points = [(20, -27.0), (30, -24.0)]
    for peak_freq in base_peaks:
        preset_points.append((peak_freq, -27.0)) # Гасим физически невоспроизводимые частоты

    # 3. Расчет Gain для высокоплотной сетки
    for freq in dense_frequencies:
        # Автоматически вычисляем АЧХ колонки в этой кастомной точке
        speaker_efficiency = interpolate_speaker_response(speaker_response, freq)
        
        # Рассчитываем адаптивный буст. Ближе к центру гармоники делаем звук плотнее
        dist_to_harmonic = min([abs(freq - (p * 2)) for p in base_peaks])
        center_bonus = max(0.0, 5.0 - (dist_to_harmonic * 0.2)) # Пиковый бонус для центра полосы
        
        gain = (0.0 - speaker_efficiency) + masking_margin + center_bonus
        gain = max(min_gain, min(max_gain, gain))
        preset_points.append((freq, round(gain, 1)))

    # 4. Стандартный мягкий спад для средних и высоких частот (фильтр шума водопада)
    preset_points.extend([
        (200, -6.0), (315, -12.0), (500, -18.0), (1000, -24.0), (20000, -24.0)
    ])

    # Очистка от дубликатов частот и финальная сортировка
    preset_points = list(set(preset_points))
    preset_points.sort(key=lambda x: x)

    # 5. Сборка строки FilterCurve для Audacity
    parts = ["FilterCurve:"]
    for index, (freq, _) in enumerate(preset_points):
        parts.append(f'f{index}="{freq}.0"')
    parts.extend([f'FilterLength="{filter_length}"', 'InterpolateLin="0"', 'InterpolationMethod="B-spline"'])
    for index, (_, level) in enumerate(preset_points):
        parts.append(f'v{index}="{level}"')

    return " ".join(parts), sorted(list(base_peaks))


class HighDensityMaskingApp(tk.Tk):
    """Графический интерфейс с поддержкой мульти-анализа и уплотнения сетки."""
    def __init__(self):
        super().__init__()
        self.title("Высокоплотный адаптивный генератор маскировки")
        self.geometry("620x460")
        self.minsize(580, 420)

        self.json_path = tk.StringVar()
        self.spectrum_files = []
        self.generated_preset = ""

        self._create_widgets()

    def _create_widgets(self):
        file_frame = ttk.LabelFrame(self, text=" Исходные файлы конфигурации ", padding=10)
        file_frame.pack(fill="x", padx=15, pady=15)

        ttk.Label(file_frame, text="Профиль колонки (JSON):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.json_path, width=42).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="Спектры шума (TXT):").grid(row=1, column=0, sticky="w", pady=5)
        self.files_label = ttk.Label(file_frame, text="Файлы не выбраны", foreground="gray", width=42, anchor="w")
        self.files_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(file_frame, text="Выбрать...", command=self._browse_spectra).grid(row=1, column=2, padx=5, pady=5)

        ttk.Button(self, text="Рассчитать пресет с плотным шагом баса", command=self._process_data).pack(pady=5)

        output_frame = ttk.LabelFrame(self, text=" Монитор плотности сетки частот ", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.log_text = tk.Text(output_frame, wrap="word", height=7, bg="#f8f9fa", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.save_btn = ttk.Button(self, text="Экспортировать уплотненный пресет в файл...", command=self._save_file, state="disabled")
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
            self.files_label.config(text=f"Выбрано файлов для анализа: {len(filenames)}", foreground="black")

    def _update_log(self, text: str, clear=False):
        self.log_text.config(state="normal")
        if clear:
            self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.config(state="disabled")

    def _process_data(self):
        if not self.json_path.get() or not self.spectrum_files:
            messagebox.showwarning("Внимание", "Необходимо выбрать профиль JSON и файлы спектра!")
            return

        try:
            self.generated_preset, base_peaks = build_high_density_preset(self.json_path.get(), self.spectrum_files)

            self._update_log("== МУЛЬТИ-АНАЛИЗ И УПЛОТНЕНИЕ СЕТКИ ЗАВЕРШЕНО ==\n", clear=True)
            self._update_log(f"Найдено стабильных резонансов соседа: {len(base_peaks)}\n")
            self._update_log(f"Точные частоты обнаруженного гула: {base_peaks} Гц\n")
            
            harmonics = [p * 2 for p in base_peaks]
            self._update_log(f"Расчетные мишени психоакустических гармоник: {harmonics} Гц\n\n")
            self._update_log("Модификация: Алгоритм автоматически уплотнил шаг сетки до 4 Гц вокруг ")
            self._update_log("каждой мишени. Промежуточные параметры АЧХ Marshall Emberton III рассчитаны ")
            self._update_log("методом линейной интерполяции. Пресет готов к выгрузке.")

            self.save_btn.config(state="normal")
        except Exception as error:
            messagebox.showerror("Ошибка вычислений", f"Произошла ошибка: {str(error)}")

    def _save_file(self):
        if not self.generated_preset:
            return

        output_file = filedialog.asksaveasfilename(
            title="Сохранить уплотненный пресет Audacity",
            initialfile="Marshall_III_High_Density_Preset.txt",
            defaultextension=".txt",
            filetypes=[
                ("Audacity Filter Curve", ".txt"),
                ("Все файлы", ".*")
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
                f"Высокоплотный пресет успешно сохранен:\n"
                f"{os.path.basename(output_file)}"
            )
        except Exception as error:
            messagebox.showerror(
                "Ошибка записи",
                f"Не удалось сохранить файл пресета: {str(error)}"
            )


if __name__ == "__main__":
    app = HighDensityMaskingApp()
    app.mainloop()
