import json
import math
import os
from pathlib import Path
import random
import struct
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import imageio_ffmpeg as im_ffmpeg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pydub import AudioSegment

# Автоматический поиск FFmpeg и прописка путей для pydub
AudioSegment.converter = im_ffmpeg.get_ffmpeg_exe()


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


def interpolate_speaker_response(
    speaker_data: dict, target_freq: float
) -> float:
    """Линейно интерполирует АЧХ колонки для промежуточных микро-частот."""
    if target_freq in speaker_data:
        return speaker_data[target_freq]

    frequencies = sorted(speaker_data.keys())

    if not frequencies:
        return -12.0
    # ИСПРАВЛЕНО: Сравниваем с первым элементом списка частот, а не со всем списком
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
    profile_path: str,
    spectrum_paths: list[str],
    max_peak_limit: float,
    min_db_threshold: float,
    min_gain: float,
    masking_margin: float,
    filter_length: int,
):
    """Строит форму купола с двухсторонним сглаживанием по чистой АЧХ из JSON."""
    base_peaks, peaks_details = get_all_spectrum_peaks_dynamic(
        spectrum_paths, min_db_threshold
    )
    dense_frequencies = set()

    for peak_freq in base_peaks:
        harmonic = peak_freq * 2
        for offset in range(-20, 25, 4):
            check_freq = harmonic + offset
            if 55 <= check_freq <= 160:
                dense_frequencies.add(check_freq)

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Защищенное извлечение только АЧХ, игнорируя любые другие лишние ключи
    speaker_response = {
        int(k): v for k, v in profile.get("speaker_response", [])
    }

    preset_points = []
    raw_points = {}

    for freq in dense_frequencies:
        speaker_efficiency = interpolate_speaker_response(
            speaker_response, freq
        )
        dist_to_harmonic = (
            min([abs(freq - (p * 2)) for p in base_peaks]) if base_peaks else 999
        )
        center_bonus = max(0.0, 5.0 - (dist_to_harmonic * 0.2))

        calculated_gain = (
            (0.0 - speaker_efficiency) + masking_margin + center_bonus
        )
        raw_points[freq] = calculated_gain

    unlimited_max_peak = max(raw_points.values()) if raw_points else 0.0

    global_reduction = 0.0
    if unlimited_max_peak > max_peak_limit:
        global_reduction = unlimited_max_peak - max_peak_limit

    active_gains = {}
    for freq, raw_gain in raw_points.items():
        normalized_gain = raw_gain - global_reduction
        final_gain = max(min_gain, normalized_gain)
        preset_points.append((freq, round(final_gain, 1)))
        active_gains[freq] = final_gain

    if active_gains:
        first_active_freq = min(active_gains.keys())
        first_active_gain = active_gains[first_active_freq]
        last_active_freq = max(active_gains.keys())
        last_active_gain = active_gains[last_active_freq]
    else:
        first_active_freq = 60
        first_active_gain = min_gain
        last_active_freq = 160
        last_active_gain = min_gain

    # 1. СГЛАЖИВАНИЕ СЛЕВА: Плавная S-образная дуга инфрабаса
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

    # 2. СГЛАЖИВАНИЕ СПРАВА: Плавная S-образная дуга хвоста АЧХ
    if last_active_freq < 20000 and last_active_gain > min_gain:
        log_start_r = math.log10(last_active_freq)
        log_end_r = math.log10(20000)
        steps_r = 60

        for i in range(1, steps_r):
            ratio = i / steps_r
            log_f = log_start_r + ratio * (log_end_r - log_start_r)
            f = int(round(10**log_f))

            if last_active_freq < f < 20000:
                smooth_factor = (1.0 - math.cos(ratio * math.pi)) / 2.0
                f_gain = last_active_gain - smooth_factor * (last_active_gain - min_gain)
                preset_points.append((f, round(f_gain, 1)))

        preset_points.append((20000, min_gain))

    unique_points = {}
    for freq, gain in preset_points:
        if freq not in unique_points or gain > unique_points[freq]:
            unique_points[freq] = gain

    preset_points = sorted(list(unique_points.items()), key=lambda x: x)

    return (
        preset_points,
        sorted(list(base_peaks)),
        global_reduction,
        filter_length,
        peaks_details,
    )


class AdvancedMaskingStudio(tk.Tk):
    """Графическая студия генерации пресетов и лимитирования звука."""

    def __init__(self):
        super().__init__()
        self.title("Advanced Noise Masking Studio (Dynamic)")
        self.geometry("1000x670")
        self.minsize(950, 600)

        self.json_path = tk.StringVar()
        self.spectrum_files = []
        self.calculated_points = []

        # Конфигурация переменных интерфейса
        self.max_peak_limit_var = tk.DoubleVar(value=0.0)
        self.min_db_threshold_var = tk.DoubleVar(value=-70.0)
        self.min_gain_var = tk.DoubleVar(value=-24.0)
        self.masking_margin_var = tk.DoubleVar(value=3.0)  # Перенесено в UI
        self.filter_length_var = tk.IntVar(value=8191)    # Перенесено в UI

        self._create_widgets()

    def _create_widgets(self):
        left_panel = ttk.Frame(self, padding=10)
        left_panel.pack(side="left", fill="both", expand=True)

        file_frame = ttk.LabelFrame(
            left_panel, text=" Исходные конфигурации ", padding=10
        )
        file_frame.pack(fill="x", pady=5)

        ttk.Label(file_frame, text="Профиль (JSON):").grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Entry(file_frame, textvariable=self.json_path, width=28).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(
            row=0, column=2, padx=2, pady=5
        )

        ttk.Label(file_frame, text="Спектры (TXT):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.files_label = ttk.Label(
            file_frame,
            text="Файлы не выбраны",
            foreground="gray",
            width=28,
            anchor="w",
        )
        self.files_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(
            file_frame, text="Выбрать...", command=self._browse_spectra
        ).grid(row=1, column=2, padx=2, pady=5)

        ttk.Label(file_frame, text="Нижний порог пиков (дБ):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(
            file_frame, textvariable=self.min_db_threshold_var, width=12
        ).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(file_frame, text="Нижний пол купола (дБ):").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Entry(file_frame, textvariable=self.min_gain_var, width=12).grid(
            row=3, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Label(file_frame, text="Запас маскировки (дБ):").grid(
            row=4, column=0, sticky="w", pady=5
        )
        ttk.Entry(file_frame, textvariable=self.masking_margin_var, width=12).grid(
            row=4, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Label(file_frame, text="Размер окна фильтра:").grid(
            row=5, column=0, sticky="w", pady=5
        )
        ttk.Entry(file_frame, textvariable=self.filter_length_var, width=12).grid(
            row=5, column=1, padx=5, pady=5, sticky="w"
        )
        ttk.Label(file_frame, text="Лимит купола (дБ):").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Entry(
            file_frame, textvariable=self.max_peak_limit_var, width=12
        ).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(
            left_panel,
            text="Рассчитать и нормализовать АЧХ",
            command=self._process_data,
        ).pack(fill="x", pady=10)

        output_frame = ttk.LabelFrame(
            left_panel, text=" Лог и диагностика нормализации ", padding=10
        )
        output_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(
            output_frame,
            wrap="word",
            height=10,
            bg="#f8f9fa",
            state="disabled",
            yscrollcommand=scrollbar.set,
        )
        self.log_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.log_text.yview)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=10)

        self.save_preset_btn = ttk.Button(
            btn_frame,
            text="Экспорт пресета .txt",
            command=self._save_preset_file,
            state="disabled",
        )
        self.save_preset_btn.pack(side="left", fill="x", expand=True, padx=2)

        self.generate_audio_btn = ttk.Button(
            btn_frame,
            text="Сгенерировать шум .mp3 (10 мин)",
            command=self._generate_audio_file,
            state="disabled",
        )
        self.generate_audio_btn.pack(side="right", fill="x", expand=True, padx=2)

        self.right_panel = ttk.LabelFrame(
            self, text=" Визуализация кривой фильтра ", padding=10
        )
        self.right_panel.pack(
            side="right", fill="both", expand=True, padx=10, pady=15
        )

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _browse_json(self):
        """Открывает диалоговое окно для выбора JSON-профиля."""
        filename = filedialog.askopenfilename(
            title="Открыть JSON-профиль",
            filetypes=[("JSON-профили", "*.json")],
        )
        if filename:
            self.json_path.set(filename)

    def _browse_spectra(self):
        """Открывает окно выбора нескольких текстовых файлов спектра."""
        filename = filedialog.askopenfilenames(
            title="Выберите файлы спектра",
            filetypes=[("Спектры Audacity", "*.txt")],
        )
        if filename:
            self.spectrum_files = list(filename)
            self.files_label.config(
                text=f"Выбрано файлов: {len(filename)}", foreground="black"
            )

    def _update_log(self, text: str, clear=False):
        """Обновляет текстовое поле логирования и диагностики."""
        self.log_text.config(state="normal")
        if clear:
            self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.config(state="disabled")

    def _plot_curve(self, points):
        """Отрисовывает результирующую кривую на графике matplotlib."""
        self.ax.clear()
        self.ax.set_title("Результирующий психоакустический купол")
        self.ax.set_xlabel("Частота (Гц)")
        self.ax.set_ylabel("Усиление (дБ)")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.5)
        self.ax.set_xscale("log")

        freqs = [p[0] for p in points]
        gains = [p[1] for p in points]

        self.ax.plot(
            freqs,
            gains,
            color="#1f77b4",
            linewidth=2,
            marker=".",
            markersize=3,
            label="Filter Curve",
        )
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-30, max(gains) + 5 if gains else 15)
        self.canvas.draw()

    def _process_data(self):
        """Обрабатывает спектры и выводит полную диагностику в лог."""
        if not self.json_path.get() or not self.spectrum_files:
            messagebox.showwarning(
                "Внимание", "Выберите JSON-профиль и файлы спектра!"
            )
            return
        try:
            peak_limit = float(self.max_peak_limit_var.get())
            min_db_threshold = float(self.min_db_threshold_var.get())
            min_gain = float(self.min_gain_var.get())
            masking_margin = float(self.masking_margin_var.get())
            filter_length_val = int(self.filter_length_var.get())
        except ValueError:
            messagebox.showerror(
                "Ошибка ввода", "Введите корректные числовые значения лимитов!"
            )
            return

        try:
            res = build_normalized_preset_dynamic(
                self.json_path.get(),
                self.spectrum_files,
                peak_limit,
                min_db_threshold,
                min_gain,
                masking_margin,
                filter_length_val,
            )
            (
                self.calculated_points,
                base_peaks,
                reduction,
                self.filter_length_val,
                peaks_details,
            ) = res

            self._update_log(
                "== ГЛОБАЛЬНАЯ ДИНАМИЧЕСКАЯ НОРМАЛИЗАЦИЯ ==\n", clear=True
            )
            self._update_log(
                f"Задан отсекающий порог громкости: {min_db_threshold} дБ\n"
            )
            self._update_log(
                f"Задан нижний пол купола (min_gain): {min_gain} дБ\n"
            )
            self._update_log(
                f"Уникальные частоты выше порога ({len(base_peaks)} шт): "
                f"{base_peaks} Гц\n\n"
            )
            self._update_log(
                "--- НАЙДЕННЫЕ ПИКИ (для калибровки порога) ---\n"
            )

            if not peaks_details:
                self._update_log("⚠️ Пиков выше указанного порога не найдено.\n")
            else:
                for idx, peak in enumerate(peaks_details, 1):
                    self._update_log(
                        f"{idx}. {peak['freq']} Гц -> {peak['level']} дБ "
                        f"({peak['source']})\n"
                    )

            self._update_log("\n-----------------------------------------\n")
            self._update_log(f"Заданный лимит вершины купола: {peak_limit} дБ\n")

            if reduction > 0:
                self._update_log(
                    f"📈 Величина ослабления Gain: -{round(reduction, 1)} дБ\n"
                )
            else:
                self._update_log("✅ Снижение уровня не потребовалось (0.0 дБ).\n")

            self._plot_curve(self.calculated_points)
            self.save_preset_btn.config(state="normal")
            self.generate_audio_btn.config(state="normal")
        except Exception as error:
            messagebox.showerror(
                "Ошибка вычислений", f"Произошла ошибка: {str(error)}"
            )

    def _generate_text_preset(self) -> str:
        """Формирует строку пресета FilterCurve для Audacity."""
        parts = ["FilterCurve:"]
        for index, (freq, _) in enumerate(self.calculated_points):
            parts.append(f'f{index}="{freq}.0"')

        parts.extend([
            f'FilterLength="{self.filter_length_val}"',
            'InterpolateLin="0"',
            'InterpolationMethod="B-spline"',
        ])

        for index, (_, level) in enumerate(self.calculated_points):
            parts.append(f'v{index}="{level}"')

        return " ".join(parts)
        
    def _save_preset_file(self):
        """Сохраняет сгенерированный пресет в текстовый файл."""
        if not self.calculated_points:
            return

        output_file = filedialog.asksaveasfilename(
            title="Сохранить нормализованный пресет",
            initialfile="Dynamic_Noise_Preset.txt",
            defaultextension=".txt",
            filetypes=[("Audacity Filter Curve", ".txt"), ("Все файлы", ".*")],
        )

        if output_file:
            try:
                preset_str = self._generate_text_preset()
                Path(output_file).write_text(
                    preset_str + "\n", encoding="utf-8"
                )
                messagebox.showinfo(
                    "Успех", "Тextовый пресет успешно экспортирован!"
                )
            except Exception as e:
                messagebox.showerror(
                    "Ошибка", f"Не удалось записать файл: {str(e)}"
                )

    def _generate_audio_file(self):
        """Инициализирует асинхронный процесс кодирования маскирующего аудио."""
        if not self.calculated_points:
            return

        output_audio = filedialog.asksaveasfilename(
            title="Сохранить маскирующий аудиофайл",
            initialfile="Dynamic_10Min_Masking.mp3",
            defaultextension=".mp3",
            filetypes=[("MP3 Audio", "*.mp3")],
        )
        if not output_audio:
            return

        self.generate_audio_btn.config(state="disabled")

        progress_win = tk.Toplevel(self)
        progress_win.title("Синтез аудио")
        progress_win.geometry("320x130")
        progress_win.resizable(False, False)
        progress_win.transient(self)
        progress_win.grab_set()

        status_label = ttk.Label(
            progress_win,
            text="Генерация базового коричневого шума...\n"
                 "Пожалуйста, подождите.",
            justify="center",
            anchor="center",
        )
        status_label.pack(pady=(15, 10), padx=10, fill="x")

        progress_bar = ttk.Progressbar(
            progress_win, orient="horizontal", mode="determinate", length=250
        )
        progress_bar.pack(pady=5, padx=20, fill="x")

        render_thread = threading.Thread(
            target=self._async_render_audio,
            args=(output_audio, progress_win, progress_bar, status_label),
            daemon=True,
        )
        render_thread.start()

    def _async_render_audio(
        self, output_path, progress_win, progress_bar, status_label
    ):
        """Выполняет тяжелый синтез аудио в отдельном фоновом потоке."""
        try:
            sample_rate = 44100
            duration_ms = 10000
            num_samples = int(sample_rate * (duration_ms / 1000.0))
            samples = []
            current_sum = 0.0

            for _ in range(num_samples):
                white_sample = random.uniform(-1.0, 1.0)
                current_sum += white_sample
                current_sum *= 0.995
                samples.append(current_sum)

            max_val = max(abs(s) for s in samples)
            if max_val > 0:
                samples = [int((s / max_val) * 26000) for s in samples]

            byte_data = struct.pack(f"<{len(samples)}h", *samples)
            base_noise = AudioSegment(
                data=byte_data,
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )

            max_boost = (
                max(p for p in self.calculated_points)
                if self.calculated_points
                else 0
            )
            filtered_chunk = base_noise.apply_gain(max_boost - 3.0)
            total_loops = int((10 * 60 * 1000) / duration_ms)
            final_audio = filtered_chunk
            chunk_with_fade = filtered_chunk.fade_in(500).fade_out(500)
            loops_built = 1

            progress_bar["max"] = total_loops

            while loops_built < total_loops:
                final_audio = final_audio.append(chunk_with_fade, crossfade=500)
                loops_built += 1

                if loops_built % 5 == 0:
                    percent = int((loops_built / total_loops) * 100)
                    progress_win.after(
                        0,
                        lambda lb=loops_built, p=percent: [
                            progress_bar.configure(value=lb),
                            status_label.configure(
                                text=f"Склеивание аудио-блоков: {p}%"
                            ),
                        ],
                    )

            progress_win.after(
                0,
                lambda: status_label.configure(
                    text="Кодирование в MP3 и запись на диск..."
                ),
            )
            final_audio = final_audio.fade_in(3000).fade_out(3000)
            final_audio.export(output_path, format="mp3", bitrate="192k")

            progress_win.after(
                0,
                lambda: [
                    progress_win.destroy(),
                    self.generate_audio_btn.config(state="normal"),
                    messagebox.showinfo(
                        "Успех", "10-минутный маскирующий аудиофайл сохранен!"
                    ),
                ],
            )
        except Exception as error:
            progress_win.after(
                0,
                lambda err=str(error): [
                    progress_win.destroy(),
                    self.generate_audio_btn.config(state="normal"),
                    messagebox.showerror(
                        "Ошибка аудио-экспорта",
                        f"Не удалось сгенерировать аудио: {err}",
                    ),
                ],
            )


if __name__ == "__main__":
    app = AdvancedMaskingStudio()
    app.mainloop()
