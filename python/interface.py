import json
import random
import struct
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import imageio_ffmpeg as im_ffmpeg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pydub import AudioSegment

# Импортируем функцию из математического ядра (core.py)
from core import build_normalized_preset_dynamic

AudioSegment.converter = im_ffmpeg.get_ffmpeg_exe()


class AdvancedMaskingStudio(tk.Tk):
    """Графическая студия генерации пресетов с интерактивным превью."""

    def __init__(self):
        super().__init__()
        self.title("Advanced Noise Masking Studio (Monolithic Fixed)")
        self.geometry("1020x720")
        self.minsize(980, 660)

        self.json_path = tk.StringVar()
        self.spectrum_files = []
        self.calculated_points = []
        self.profiles_file = Path(__file__).parent / "speaker_profiles.json"
        self.device_names = []
        self._init_default_profiles_file()
        self._load_devices()

        self.max_peak_limit_var = tk.StringVar(value="0.0")
        self.min_db_threshold_var = tk.StringVar(value="-70.0")
        self.min_gain_var = tk.StringVar(value="-24.0")
        self.masking_margin_var = tk.StringVar(value="3.0")
        self.filter_length_var = tk.StringVar(value="8191")

        self.h2_gain_var = tk.DoubleVar(value=7.0)
        self.h3_gain_var = tk.DoubleVar(value=4.5)

        self._create_widgets()
        self._setup_traces()

    def _init_default_profiles_file(self):
        if not self.profiles_file.exists():
            default_data = {
                "profiles": {
                    "Marshall Emberton III": [
                        [20, -25.0], [40, -18.0], [55, -12.0], [65, -5.0], 
                        [80, -2.0], [100, 0.0], [150, -1.0], [200, -2.0], 
                        [500, -1.5], [1000, 0.0], [5000, 1.0], [20000, -3.0]
                    ]
                }
            }
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=4)

    def _setup_traces(self):
        self.max_peak_limit_var.trace_add("write", lambda *args: self._process_data(quiet=True))
        self.min_db_threshold_var.trace_add("write", lambda *args: self._process_data(quiet=True))
        self.min_gain_var.trace_add("write", lambda *args: self._process_data(quiet=True))
        self.masking_margin_var.trace_add("write", lambda *args: self._process_data(quiet=True))
        self.filter_length_var.trace_add("write", lambda *args: self._process_data(quiet=True))
    def _create_widgets(self):
        left_panel = ttk.Frame(self, padding=10)
        left_panel.pack(side="left", fill="both", expand=True)

        file_frame = ttk.LabelFrame(left_panel, text=" Исходные конфигурации ", padding=10)
        file_frame.pack(fill="x", pady=4)

        ttk.Label(file_frame, text="Профиль (JSON):").grid(row=0, column=0, sticky="w", pady=4)
        self.profile_combo = ttk.Combobox(
            file_frame, textvariable=self.json_path, values=self.device_names, width=20, state="readonly"
        )
        self.profile_combo.grid(row=0, column=1, padx=5, pady=4, sticky="w")
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_device_select)
        ttk.Button(file_frame, text="Обзор...", command=self._browse_json).grid(row=0, column=2, padx=2, pady=4)
        ttk.Button(file_frame, text="+ Устройство", command=self._add_device).grid(row=0, column=3, padx=2, pady=4)

        ttk.Label(file_frame, text="Спектры (TXT):").grid(row=1, column=0, sticky="w", pady=4)
        self.files_label = ttk.Label(file_frame, text="Файлы не выбраны", foreground="gray", width=24, anchor="w")
        self.files_label.grid(row=1, column=1, padx=5, pady=4, sticky="w")
        ttk.Button(file_frame, text="Выбрать...", command=self._browse_spectra).grid(row=1, column=2, padx=2, pady=4)

        ttk.Label(file_frame, text="Нижний порог пиков (дБ):").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(file_frame, textvariable=self.min_db_threshold_var, width=12).grid(row=2, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(file_frame, text="Нижний пол купола (дБ):").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(file_frame, textvariable=self.min_gain_var, width=12).grid(row=3, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(file_frame, text="Запас маскировки (дБ):").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(file_frame, textvariable=self.masking_margin_var, width=12).grid(row=4, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(file_frame, text="Размер окна фильтра:").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(file_frame, textvariable=self.filter_length_var, width=12).grid(row=5, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(file_frame, text="Лимит купола (дБ):").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(file_frame, textvariable=self.max_peak_limit_var, width=12).grid(row=6, column=1, padx=5, pady=4, sticky="w")

        harm_frame = ttk.LabelFrame(left_panel, text=" Тонкая калибровка гармоник суб-баса ", padding=10)
        harm_frame.pack(fill="x", pady=6)

        ttk.Label(harm_frame, text="Усиление 2-й гармоники:").grid(row=0, column=0, sticky="w", pady=5)
        self.h2_slider = ttk.Scale(
            harm_frame, from_=0.0, to=15.0, variable=self.h2_gain_var, orient="horizontal", length=140,
            command=lambda v: [self.h2_lbl.config(text=f"{float(v):.1f} дБ"), self._process_data(quiet=True)]
        )
        self.h2_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.h2_lbl = ttk.Label(harm_frame, text=f"{self.h2_gain_var.get():.1f} дБ", width=8)
        self.h2_lbl.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(harm_frame, text="Усиление 3-й гармоники:").grid(row=1, column=0, sticky="w", pady=5)
        self.h3_slider = ttk.Scale(
            harm_frame, from_=0.0, to=15.0, variable=self.h3_gain_var, orient="horizontal", length=140,
            command=lambda v: [self.h3_lbl.config(text=f"{float(v):.1f} дБ"), self._process_data(quiet=True)]
        )
        self.h3_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.h3_lbl = ttk.Label(harm_frame, text=f"{self.h3_gain_var.get():.1f} дБ", width=8)
        self.h3_lbl.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Button(left_panel, text="Рассчитать и нормализовать АЧХ", command=lambda: self._process_data(quiet=False)).pack(fill="x", pady=6)

        output_frame = ttk.LabelFrame(left_panel, text=" Лог и диагностика нормализации ", padding=10)
        output_frame.pack(fill="both", expand=True, pady=4)

        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(
            output_frame, wrap="word", height=6, bg="#f8f9fa", state="disabled", yscrollcommand=scrollbar.set
        )
        self.log_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.log_text.yview)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=6)

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
        """Загружает список устройств, исправляя опечатку с передачей списка."""
        self.device_names = []
        if self.profiles_file.exists():
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.device_names = sorted(data.get("profiles", {}).keys())
        if self.device_names:
            # Исправлено: берем строго первую строку, исключая tuple-передачу
            self.json_path.set(self.device_names[0])

    def _on_device_select(self, event=None):
        self._process_data(quiet=False)

    def _browse_json(self):
        filename = filedialog.askopenfilename(title="Открыть JSON-профиль", filetypes=[("JSON-профили", "*.json")])
        if filename:
            self.json_path.set(filename)
            self._process_data(quiet=False)
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
            if not name or not raw:
                messagebox.showwarning("Ошибка", "Заполните все поля!")
                return

            pairs = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    parts = line.replace(",", " ").replace("\t", " ").split()
                    pairs.append([int(float(parts[0])), float(parts[1])])
                except (ValueError, IndexError):
                    messagebox.showwarning("Ошибка", f"Некорректная строка: {line}")
                    return

            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["profiles"][name] = pairs
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            self._load_devices()
            self.profile_combo["values"] = self.device_names
            self.json_path.set(name)
            add_win.destroy()
            self._process_data(quiet=False)

        ttk.Button(add_win, text="Сохранить", command=save_device).pack(pady=10)

    def _browse_spectra(self):
        filename = filedialog.askopenfilenames(title="Выберите файлы спектра", filetypes=[("Спектры Audacity", "*.txt")])
        if filename:
            self.spectrum_files = list(filename)
            self.files_label.config(text=f"Выбрано файлов: {len(filename)}", foreground="black")
            self._process_data(quiet=False)

    def _update_log(self, text: str, clear=False):
        self.log_text.config(state="normal")
        if clear:
            self.log_text.delete("1.0", tk.END)
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

        self.ax.plot(freqs, gains, color="#1f77b4", linewidth=2, marker=".", markersize=3, label="Кривая фильтра")
        self.ax.set_xlim(20, 20000)
        
        min_y = -30
        max_y = max(gains) + 5 if gains else 15
        self.ax.set_ylim(min_y, max(max_y, 15))
        self.canvas.draw()

    def _process_data(self, quiet=False):
        if not self.json_path.get() or not self.spectrum_files:
            return
        try:
            peak_limit = float(self.max_peak_limit_var.get() or 0.0)
            min_db_threshold = float(self.min_db_threshold_var.get() or -70.0)
            min_gain = float(self.min_gain_var.get() or -24.0)
            masking_margin = float(self.masking_margin_var.get() or 3.0)
            filter_length_val = int(self.filter_length_var.get() or 8191)
            h2_gain = self.h2_gain_var.get()
            h3_gain = self.h3_gain_var.get()
        except ValueError:
            return

        selected_device = self.json_path.get()
        # Защитная распаковка tuple-строк в чистые имена путей
        if isinstance(selected_device, (tuple, list)) and len(selected_device) > 0:
            selected_device = selected_device[0]

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
                min_gain, masking_margin, filter_length_val, h2_gain, h3_gain, speaker_response
            )
            (self.calculated_points, base_peaks, reduction, self.filter_length_val, peaks_details) = res

            if not quiet:
                self._update_log("== ГЛОБАЛЬНАЯ ДИНАМИЧЕСКАЯ НОРМАЛИЗАЦИЯ ==\n", clear=True)
                self._update_log(f"Уникальные частоты выше порога ({len(base_peaks)} шт): {base_peaks} Гц\n\n")
                self._update_log("--- НАЙДЕННЫЕ ПИКИ СУБ-БАСА ---\n")
                for idx, peak in enumerate(peaks_details[:15], 1):
                    self._update_log(f"{idx}. {peak['freq']} Гц -> {peak['level']} дБ ({peak['source']})\n")
                self._update_log("\n-----------------------------------------\n")
                if reduction > 0:
                    self._update_log(f"📈 Величина ослабления основного Gain (shape preservation): -{round(reduction, 1)} дБ\n")
                else:
                    self._update_log("✅ Снижение основного уровня не потребовалось (0.0 дБ).\n")

            self._plot_curve(self.calculated_points)
            self.save_preset_btn.config(state="normal")
            self.generate_audio_btn.config(state="normal")
        except Exception as error:
            if not quiet:
                messagebox.showerror("Ошибка вычислений", f"Произошла ошибка: {str(error)}")

    def _generate_text_preset(self) -> str:
        parts = ["FilterCurve:"]
        for index, (freq, _) in enumerate(self.calculated_points):
            parts.append(f'f{index}="{freq}.0"')
        parts.extend([f'FilterLength="{self.filter_length_val}"', 'InterpolateLin="0"', 'InterpolationMethod="B-spline"'])
        for index, (_, level) in enumerate(self.calculated_points):
            parts.append(f'v{index}="{level}"')
        return " ".join(parts)
        
    def _save_preset_file(self):
        if not self.calculated_points:
            return
        output_file = filedialog.asksaveasfilename(
            title="Сохранить нормализованный пресет", initialfile="Dynamic_Noise_Preset.txt",
            defaultextension=".txt", filetypes=[("Audacity Filter Curve", ".txt"), ("Все файлы", ".*")]
        )
        if output_file:
            try:
                Path(output_file).write_text(self._generate_text_preset() + "\n", encoding="utf-8")
                messagebox.showinfo("Успех", "Текстовый пресет успешно экспортирован!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось записать файл: {str(e)}")

    def _generate_audio_file(self):
        if not self.calculated_points:
            return
        output_audio = filedialog.asksaveasfilename(
            title="Сохранить маскирующий аудиофайл", initialfile="Dynamic_10Min_Masking.mp3",
            defaultextension=".mp3", filetypes=[("MP3 Audio", "*.mp3")]
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

        status_label = ttk.Label(progress_win, text="Генерация базового коричневого шума...\nПожалуйста, подождите.", justify="center")
        status_label.pack(pady=(15, 10), padx=10, fill="x")

        progress_bar = ttk.Progressbar(progress_win, orient="horizontal", mode="determinate", length=250)
        progress_bar.pack(pady=5, padx=20, fill="x")

        threading.Thread(
            target=self._async_render_audio, args=(output_audio, progress_win, progress_bar, status_label), daemon=True
        ).start()

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
                    progress_win.after(
                        0,
                        lambda lb=loops_built, p=percent: [
                            progress_bar.configure(value=lb),
                            status_label.configure(
                                text=f"Склеивание аудио-блоков: {p}%"
                            ),
                        ],
                    )

            # Перевод статуса в режим кодирования
            progress_win.after(
                0,
                lambda: status_label.configure(
                    text="Кодирование в MP3 и запись на диск..."
                ),
            )

            # Наложение финального фейда и экспорт на диск
            final_audio = final_audio.fade_in(3000).fade_out(3000)
            final_audio.export(output_path, format="mp3", bitrate="192k")

            # Успешное закрытие окна прогресса
            progress_win.after(
                0,
                lambda: [
                    progress_win.destroy(),
                    self.generate_audio_btn.config(state="normal"),
                    messagebox.showinfo(
                        "Успех",
                        "10-минутный маскирующий аудиофайл сохранен!",
                    ),
                ],
            )
        except Exception as error:
            # Обработка критических ошибок сборки аудио
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
