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

AudioSegment.converter = im_ffmpeg.get_ffmpeg_exe()


def get_all_spectrum_peaks_dynamic(
    file_paths: list[str], min_db_threshold: float
) -> tuple[set[int], list[dict]]:
    detected_frequencies = set()
    all_found_peaks_details = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        file_peaks = []
        with open(path, "r", encoding="utf-8") as file:
            next(file, None)
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
    if target_freq in speaker_data:
        return speaker_data[target_freq]

    frequencies = sorted(speaker_data.keys())
    if not frequencies:
        return -12.0
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
    base_peaks, peaks_details = get_all_spectrum_peaks_dynamic(
        spectrum_paths, min_db_threshold
    )
    dense_frequencies = set()

    if base_peaks:
        min_peak = min(base_peaks)
        max_peak = max(base_peaks)
        start_dome = max(40, min_peak - 10)
        end_dome = min(250, (max_peak * 3) + 20)
        for freq in range(start_dome, end_dome + 1, 2):
            dense_frequencies.add(freq)
    else:
        for freq in range(55, 161, 2):
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

    for freq in dense_frequencies:
        speaker_efficiency = interpolate_speaker_response(speaker_response, freq)
        center_bonus = 0.0
        
        for p in base_peaks:
            speaker_eff_at_p = interpolate_speaker_response(speaker_response, p)
            is_infra_for_speaker = (speaker_eff_at_p < -8.0)
            
            dist_f1 = abs(freq - p)
            dist_f2 = abs(freq - (p * 2))
            dist_f3 = abs(freq - (p * 3))
            
            if not is_infra_for_speaker:
                b1 = max(0.0, 6.0 - (dist_f1 * 0.15))
                b2 = max(0.0, (h2_gain * 0.43) - (dist_f2 * 0.1))
                center_bonus = max(center_bonus, b1, b2)
            else:
                b1 = 0.0 
                b2 = max(0.0, h2_gain - (dist_f2 * 0.12))
                b3 = max(0.0, h3_gain - (dist_f3 * 0.08))
                center_bonus = max(center_bonus, b1, b2, b3)

        calculated_gain = (0.0 - speaker_efficiency) + masking_margin + center_bonus
        raw_points[freq] = calculated_gain

    unlimited_max_peak = max(raw_points.values()) if raw_points else 0.0
    global_reduction = max(0.0, unlimited_max_peak - max_peak_limit)

    active_gains = {}
    for freq, raw_gain in raw_points.items():
        normalized_gain = raw_gain - global_reduction
        final_gain = max(min_gain, normalized_gain)
        preset_points.append((freq, round(final_gain, 1)))
        active_gains[freq] = final_gain

    first_active_freq = min(active_gains.keys()) if active_gains else 55
    first_active_gain = active_gains[first_active_freq] if active_gains else min_gain
    last_active_freq = max(active_gains.keys()) if active_gains else 160
    last_active_gain = active_gains[last_active_freq] if active_gains else min_gain

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

    start_r_freq = last_active_freq + 1
    if start_r_freq < 20000 and last_active_gain > min_gain:
        log_start_r = math.log10(start_r_freq)
        log_end_r = math.log10(20000)
        steps_r = 60
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


class AdvancedMaskingStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Noise Masking Studio (Real-time)")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        self.json_path = tk.StringVar()
        self.spectrum_files = []
        self.calculated_points = []
        self.profiles_file = Path(__file__).parent / "speaker_profiles.json"
        self.device_names = []
        self._load_devices()

        self.max_peak_limit_var = tk.DoubleVar(value=0.0)
        self.min_db_threshold_var = tk.DoubleVar(value=-70.0)
        self.min_gain_var = tk.DoubleVar(value=-24.0)
        self.masking_margin_var = tk.DoubleVar(value=3.0)
        self.filter_length_var = tk.IntVar(value=8191)
        
        self.h2_gain_var = tk.DoubleVar(value=7.0)
        self.h3_gain_var = tk.DoubleVar(value=4.5)

        self._create_widgets()

    def _create_widgets(self):
        left_panel = ttk.Frame(self, padding=10)
        left_panel.pack(side="left", fill="both", expand=True)

        file_frame = ttk.LabelFrame(left_panel, text=" Конфигурации и параметры ", padding=10)
        file_frame.pack(fill="x", pady=5)

        ttk.Label(file_frame, text="Профиль (JSON):").grid(row=0, column=0, sticky="w", pady=3)
        self.profile_combo = ttk.Combobox(file_frame, textvariable=self.json_path, values=self.device_names, width=20, state="readonly")
        self.profile_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_param_changed)
        
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(row=0, column=2, padx=2, pady=3)
        ttk.Button(file_frame, text="+ Устройство", command=self._add_device).grid(row=0, column=3, padx=2, pady=3)

        ttk.Label(file_frame, text="Спектры (TXT):").grid(row=1, column=0, sticky="w", pady=3)
        self.files_label = ttk.Label(file_frame, text="Файлы не выбраны", foreground="gray", width=25, anchor="w")
        self.files_label.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        ttk.Button(file_frame, text="Выбрать...", command=self._browse_spectra).grid(row=1, column=2, padx=2, pady=3)

        # Текстовые поля с трассировкой изменений для живого обновления
        fields = [
            ("Нижний порог пиков (дБ):", self.min_db_threshold_var),
            ("Нижний пол купола (дБ):", self.min_gain_var),
            ("Запас маскировки (дБ):", self.masking_margin_var),
            ("Размер окна фильтра:", self.filter_length_var),
            ("Лимит купола (дБ):", self.max_peak_limit_var)
        ]
        
        for idx, (label_text, var) in enumerate(fields, start=2):
            ttk.Label(file_frame, text=label_text).grid(row=idx, column=0, sticky="w", pady=3)
            entry = ttk.Entry(file_frame, textvariable=var, width=12)
            entry.grid(row=idx, column=1, padx=5, pady=3, sticky="w")
            var.trace_add("write", self._on_param_changed)

        # Блок интерактивных ползунков для гармоник
        slider_frame = ttk.LabelFrame(left_panel, text=" Тонкая калибровка гармоник суб-баса (Живой пересчет) ", padding=10)
        slider_frame.pack(fill="x", pady=5)

        ttk.Label(slider_frame, text="Усиление 2-й гармоники:").grid(row=0, column=0, sticky="w", pady=5)
        self.h2_scale = ttk.Scale(slider_frame, from_=0.0, to=15.0, variable=self.h2_gain_var, orient="horizontal", length=180, command=self._on_slider_scroll)
        self.h2_scale.grid(row=0, column=1, padx=5, pady=5)
        self.h2_lbl = ttk.Label(slider_frame, text="7.0 дБ", width=8)
        self.h2_lbl.grid(row=0, column=2, padx=2, pady=5)

        ttk.Label(slider_frame, text="Усиление 3-й гармоники:").grid(row=1, column=0, sticky="w", pady=5)
        self.h3_scale = ttk.Scale(slider_frame, from_=0.0, to=15.0, variable=self.h3_gain_var, orient="horizontal", length=180, command=self._on_slider_scroll)
        self.h3_scale.grid(row=1, column=1, padx=5, pady=5)
        self.h3_lbl = ttk.Label(slider_frame, text="4.5 дБ", width=8)
        self.h3_lbl.grid(row=1, column=2, padx=2, pady=5)

        output_frame = ttk.LabelFrame(left_panel, text=" Лог и диагностика нормализации ", padding=10)
        output_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(output_frame, wrap="word", height=8, bg="#f8f9fa", state="disabled", yscrollcommand=scrollbar.set)
        self.log_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.log_text.yview)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=5)

        self.save_preset_btn = ttk.Button(btn_frame, text="Экспорт пресета .txt", command=self._save_preset_file, state="disabled")
        self.save_preset_btn.pack(side="left", fill="x", expand=True, padx=2)

        self.generate_audio_btn = ttk.Button(btn_frame, text="Сгенерировать шум .mp3", command=self._generate_audio_file, state="disabled")
        self.generate_audio_btn.pack(side="right", fill="x", expand=True, padx=2)

        self.right_panel = ttk.LabelFrame(self, text=" Визуализация кривой фильтра ", padding=10)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _load_devices(self):
        self.device_names = []
        if self.profiles_file.exists():
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.device_names = sorted(data.get("profiles", {}).keys())
        if self.device_names:
            self.json_path.set(self.device_names[0])

    def _browse_json(self):
        filename = filedialog.askopenfilename(title="Открыть JSON-профиль", filetypes=[("JSON-профили", "*.json")])
        if filename:
            self.json_path.set(filename)
            self._process_data(quiet=True)

    def _add_device(self):
        add_win = tk.Toplevel(self)
        add_win.title("Добавить устройство")
        add_win.geometry("380x320")
        add_win.resizable(False, False)
        add_win.transient(self)
        add_win.grab_set()

        ttk.Label(add_win, text="Название устройства:").pack(pady=(10, 2))
        name_entry = ttk.Entry(add_win, width=40)
        name_entry.pack(pady=2)

        ttk.Label(add_win, text="АЧХ (частота,дБ, строка):").pack(pady=(10, 2))
        text_area = tk.Text(add_win, height=10, width=45)
        text_area.pack(pady=2, padx=10)

        def save_device():
            name = name_entry.get().strip()
            raw = text_area.get("1.0", tk.END).strip()
            if not name: return
            if not raw: return

            pairs = []
            for line in raw.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    parts = line.replace(",", " ").replace("\t", " ").split()
                    pairs.append([int(float(parts[0])), float(parts[1])])
                except Exception: continue

            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["profiles"][name] = pairs
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            self.device_names.append(name)
            self.profile_combo["values"] = self.device_names
            self.json_path.set(name)
            add_win.destroy()
            self._process_data(quiet=True)

        ttk.Button(add_win, text="Сохранить", command=save_device).pack(pady=10)

    def _browse_spectra(self):
        filename = filedialog.askopenfilenames(title="Выберите файлы спектра", filetypes=[("Спектры Audacity", "*.txt")])
        if filename:
            self.spectrum_files = list(filename)
            self.files_label.config(text=f"Выбрано файлов: {len(filename)}", foreground="black")
            self._process_data(quiet=False)

    def _on_param_changed(self, *args):
        # Метод тихой автоперерисовки при изменении полей
        self._process_data(quiet=True)

    def _on_slider_scroll(self, *args):
        # Метод для обработки прокрутки ползунков
        h2 = self.h2_gain_var.get()
        h3 = self.h3_gain_var.get()
        self.h2_lbl.config(text=f"{h2:.1f} дБ")
        self.h3_lbl.config(text=f"{h3:.1f} дБ")
        self._process_data(quiet=True)

    def _update_log(self, text: str, clear=False):
        self.log_text.config(state="normal")
        if clear: self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.config(state="disabled")

    def _plot_curve(self, points):
        self.ax.clear()
        self.ax.set_title("Результирующий психоакустический купол")
        self.ax.set_xlabel("Частота (Гц)")
        self.ax.set_ylabel("Усиление (дБ)")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.5)
        self.ax.set_xscale("log")

        freqs = [p[0] for p in points]
        gains = [p[1] for p in points]

        self.ax.plot(freqs, gains, color="#1f77b4", linewidth=2, marker=".", markersize=3)
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-30, max(gains) + 5 if gains else 15)
        self.canvas.draw()

    def _process_data(self, quiet=False):
        if not self.json_path.get() or not self.spectrum_files:
            return
        try:
            peak_limit = float(self.max_peak_limit_var.get())
            min_db_threshold = float(self.min_db_threshold_var.get())
            min_gain = float(self.min_gain_var.get())
            masking_margin = float(self.masking_margin_var.get())
            filter_length_val = int(self.filter_length_var.get())
            h2_val = float(self.h2_gain_var.get())
            h3_val = float(self.h3_gain_var.get())
        except ValueError:
            return

        selected_device = self.json_path.get()
        profile_path = None
        speaker_response = None

        if selected_device in self.device_names:
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                profiles_data = json.load(f)
            speaker_response = {int(k): v for k, v in profiles_data["profiles"][selected_device]}
        else:
            profile_path = selected_device
            if profile_path and not Path(profile_path).is_absolute():
                profile_path = str(Path(__file__).parent / profile_path)

        try:
            res = build_normalized_preset_dynamic(
                profile_path, self.spectrum_files, peak_limit, min_db_threshold,
                min_gain, masking_margin, filter_length_val, h2_val, h3_val,
                speaker_response if profile_path is None else None
            )
            (self.calculated_points, base_peaks, reduction, self.filter_length_val, peaks_details) = res

            if not quiet:
                self._update_log("== ГЛОБАЛЬНАЯ ДИНАМИЧЕСКАЯ НОРМАЛИЗАЦИЯ ==\n", clear=True)
                self._update_log(f"Задан отсекающий порог громкости: {min_db_threshold} дБ\n")
                self._update_log(f"Задан нижний пол купола (min_gain): {min_gain} дБ\n")
                self._update_log(f"Уникальные частоты выше порога ({len(base_peaks)} шт): {base_peaks} Гц\n\n")
                self._update_log("--- НАЙДЕННЫЕ ПИКИ ---\n")
                if not peaks_details:
                    self._update_log("⚠️ Пиков выше указанного порога не найдено.\n")
                else:
                    for idx, peak in enumerate(peaks_details, 1):
                        self._update_log(f"{idx}. {peak['freq']} Гц -> {peak['level']} дБ ({peak['source']})\n")
                self._update_log("\n-----------------------------------------\n")
                self._update_log(f"Заданный лимит вершины купола: {peak_limit} дБ\n")
                if reduction > 0:
                    self._update_log(f"📈 Величина ослабления Gain: -{round(reduction, 1)} дБ\n")
                else:
                    self._update_log("✅ Снижение уровня не потребовалось (0.0 дБ).\n")

            self._plot_curve(self.calculated_points)
            self.save_preset_btn.config(state="normal")
            self.generate_audio_btn.config(state="normal")
        except Exception:
            pass

    def _generate_text_preset(self) -> str:
        parts = ["FilterCurve:"]
        for index, (freq, _) in enumerate(self.calculated_points):
            parts.append(f'f{index}="{freq}.0"')
        parts.extend([f'FilterLength="{self.filter_length_val}"', 'InterpolateLin="0"', 'InterpolationMethod="B-spline"'])
        for index, (_, level) in enumerate(self.calculated_points):
            parts.append(f'v{index}="{level}"')
        return " ".join(parts)

    def _save_preset_file(self):
        if not self.calculated_points: return
        output_file = filedialog.asksaveasfilename(title="Сохранить нормализованный пресет", initialfile="Dynamic_Noise_Preset.txt", defaultextension=".txt", filetypes=[("Audacity Filter Curve", ".txt")])
        if output_file:
            try:
                preset_str = self._generate_text_preset()
                Path(output_file).write_text(preset_str + "\n", encoding="utf-8")
                messagebox.showinfo("Успех", "Текстовый пресет успешно экспортирован!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось записать файл: {str(e)}")

    def _generate_audio_file(self):
        if not self.calculated_points: return
        output_audio = filedialog.asksaveasfilename(title="Сохранить маскирующий аудиофайл", initialfile="Dynamic_10Min_Masking.mp3", defaultextension=".mp3", filetypes=[("MP3 Audio", "*.mp3")])
        if not output_audio: return

        self.generate_audio_btn.config(state="disabled")
        progress_win = tk.Toplevel(self)
        progress_win.title("Синтез аудио")
        progress_win.geometry("320x130")
        progress_win.transient(self)
        progress_win.grab_set()

        status_label = ttk.Label(progress_win, text="Генерация базового коричневого шума...\nПожалуйста, подождите.", justify="center", anchor="center")
        status_label.pack(pady=(15, 10), padx=10, fill="x")

        progress_bar = ttk.Progressbar(progress_win, orient="horizontal", mode="determinate", length=250)
        progress_bar.pack(pady=5, padx=20, fill="x")

        render_thread = threading.Thread(target=self._async_render_audio, args=(output_audio, progress_win, progress_bar, status_label), daemon=True)
        render_thread.start()

    def _async_render_audio(self, output_path, progress_win, progress_bar, status_label):
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
            base_noise = AudioSegment(data=byte_data, sample_width=2, frame_rate=sample_rate, channels=1)

            max_boost = max(p[1] for p in self.calculated_points) if self.calculated_points else 0.0
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
                    progress_win.after(0, lambda lb=loops_built, p=percent: [
                        progress_bar.configure(value=lb),
                        status_label.configure(text=f"Склеивание аудио-блоков: {p}%"),
                    ])

            progress_win.after(0, lambda: status_label.configure(text="Кодирование в MP3 и запись на диск..."))
            final_audio = final_audio.fade_in(3000).fade_out(3000)
            final_audio.export(output_path, format="mp3", bitrate="192k")

            progress_win.after(0, lambda: [
                progress_win.destroy(),
                self.generate_audio_btn.config(state="normal"),
                messagebox.showinfo("Успех", "10-минутный маскирующий аудиофайл сохранен!"),
            ])
        except Exception as error:
            progress_win.after(0, lambda err=str(error): [
                progress_win.destroy(),
                self.generate_audio_btn.config(state="normal"),
                messagebox.showerror("Ошибка аудио-экспорта", f"Не удалось сгенерировать аудио: {err}"),
            ])


if __name__ == "__main__":
    app = AdvancedMaskingStudio()
    app.mainloop()
