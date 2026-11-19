"""Jendela utama dan logika antarmuka aplikasi Mengolah Citra UPN."""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from camera import CameraPreviewWindow, CameraSession, LiveFrameProcessor
import filters
from histogram import HistogramPanel
import theme


FILTER_GROUPS = [
    ("Basic", ["Grayscale", "Negative", "Thresholding"]),
    ("Color", ["Brightness", "Saturation/Hue", "Equalization"]),
    ("Spatial", ["Blur", "Sharpening", "Edge Detection", "Rotate"]),
    ("Effects", ["Mirroring", "Sketch", "Vignette"]),
    ("Noise", ["Add Gaussian Noise", "Add Salt & Pepper", "Denoising"]),
]

PROCESSING_MODES = ("Single Filter", "Filter Chaining")

# Konfigurasi tiap filter: fungsi target pada filters.py dan daftar kontrol parameter dinamis
FILTER_SPECS: dict[str, dict[str, object]] = {
    "Grayscale": {"func": filters.grayscale, "params": []},
    "Negative": {"func": filters.negative, "params": []},
    "Thresholding": {
        "func": filters.thresholding,
        "params": [
            {
                "key": "threshold",
                "label": "Threshold",
                "kind": "int",
                "min": 0,
                "max": 255,
                "default": 127,
            }
        ],
    },
    "Brightness": {
        "func": filters.brightness,
        "params": [
            {
                "key": "value",
                "label": "Brightness",
                "kind": "int",
                "min": -100,
                "max": 100,
                "default": 0,
            }
        ],
    },
    "Saturation/Hue": {
        "func": filters.saturation_hue,
        "params": [
            {
                "key": "hue",
                "label": "Hue",
                "kind": "int",
                "min": -90,
                "max": 90,
                "default": 0,
            },
            {
                "key": "saturation",
                "label": "Saturation (%)",
                "kind": "int",
                "min": 0,
                "max": 200,
                "default": 100,
            },
        ],
    },
    "Equalization": {"func": filters.equalization, "params": []},
    "Blur": {
        "func": filters.blur,
        "params": [
            {
                "key": "method",
                "label": "Method",
                "kind": "dropdown",
                "options": ["Gaussian", "Average"],
                "default": "Gaussian",
            },
            {
                "key": "kernel_size",
                "label": "Kernel Size",
                "kind": "odd",
                "min": 3,
                "max": 31,
                "default": 5,
            },
        ],
    },
    "Sharpening": {
        "func": filters.sharpening,
        "params": [
            {
                "key": "strength",
                "label": "Strength",
                "kind": "float",
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "decimals": 1,
                "default": 1.0,
            }
        ],
    },
    "Edge Detection": {
        "func": filters.edge_detection,
        "params": [
            {
                "key": "method",
                "label": "Method",
                "kind": "dropdown",
                "options": ["Canny", "Sobel"],
                "default": "Canny",
                "refresh": True,
            },
            {
                "key": "threshold",
                "label": "Threshold",
                "kind": "int",
                "min": 0,
                "max": 255,
                "default": 100,
                "visible_when": ("method", "Canny"),
            },
        ],
    },
    "Rotate": {
        "func": filters.rotate,
        "params": [
            {
                "key": "angle",
                "label": "Angle",
                "kind": "int",
                "min": -180,
                "max": 180,
                "default": 0,
            }
        ],
    },
    "Mirroring": {
        "func": filters.mirroring,
        "params": [
            {
                "key": "direction",
                "label": "Direction",
                "kind": "dropdown",
                "options": ["Horizontal", "Vertical"],
                "default": "Horizontal",
            }
        ],
    },
    "Sketch": {
        "func": filters.sketch,
        "params": [
            {
                "key": "kernel_size",
                "label": "Blur Kernel",
                "kind": "odd",
                "min": 3,
                "max": 31,
                "default": 21,
            }
        ],
    },
    "Vignette": {
        "func": filters.vignette,
        "params": [
            {
                "key": "strength",
                "label": "Strength",
                "kind": "float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "decimals": 2,
                "default": 0.6,
            },
            {
                "key": "radius",
                "label": "Radius",
                "kind": "float",
                "min": 0.2,
                "max": 1.5,
                "step": 0.05,
                "decimals": 2,
                "default": 0.8,
            },
        ],
    },
    "Add Gaussian Noise": {
        "func": filters.add_gaussian_noise,
        "params": [
            {
                "key": "sigma",
                "label": "Sigma",
                "kind": "int",
                "min": 5,
                "max": 80,
                "default": 25,
            }
        ],
    },
    "Add Salt & Pepper": {
        "func": filters.add_salt_pepper_noise,
        "params": [
            {
                "key": "amount",
                "label": "Amount (%)",
                "kind": "int",
                "min": 1,
                "max": 20,
                "default": 5,
            }
        ],
    },
    "Denoising": {
        "func": filters.denoising,
        "params": [
            {
                "key": "method",
                "label": "Method",
                "kind": "dropdown",
                "options": ["Median", "Gaussian", "Bilateral"],
                "default": "Median",
                "refresh": True,
            },
            {
                "key": "kernel_size",
                "label": "Kernel Size",
                "kind": "odd",
                "min": 3,
                "max": 31,
                "default": 5,
            },
            {
                "key": "sigma",
                "label": "Sigma",
                "kind": "int",
                "min": 5,
                "max": 150,
                "default": 75,
                "visible_when": ("method", "Bilateral"),
            },
        ],
    },
}


class MengolahCitraApp(ctk.CTk):
    """Kelas utama jendela aplikasi pengolahan citra."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("light")

        self.title(theme.APP_TITLE)
        self.geometry(f"{theme.WINDOW_WIDTH}x{theme.WINDOW_HEIGHT}")
        self.minsize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
        self.configure(fg_color=theme.COLORS["bg_window"])

        self._font_family = theme.resolve_font_family()
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self._current_filename: str | None = None
        self._current_color_mode: str | None = None

        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self._display_sources: dict[str, Image.Image] = {
            "before": self._create_dummy_image(variant="before"),
            "after": self._create_dummy_image(variant="after"),
        }
        self._image_versions: dict[str, int] = {"before": 0, "after": 0}
        self._last_rendered: dict[str, tuple[int, int, int] | None] = {
            "before": None,
            "after": None,
        }
        self._image_labels: dict[str, tk.Label] = {}
        self._image_boxes: dict[str, ctk.CTkFrame] = {}
        self._resize_jobs: dict[str, str | None] = {"before": None, "after": None}

        self._selected_filter: str = "Brightness"
        self._processing_mode: str = "Single Filter"
        self._chaining_started: bool = False
        self._chaining_applied_filters: set[str] = set()
        self._is_processing: bool = False
        self._process_job_id: int = 0
        self._process_queue: queue.Queue[
            tuple[int, str, str, np.ndarray | None, str | None]
        ] = queue.Queue()

        self._filter_buttons: dict[str, ctk.CTkButton] = {}
        self._toolbar_buttons: dict[str, ctk.CTkButton] = {}
        self._mode_buttons: dict[str, ctk.CTkButton] = {}
        self._param_values: dict[str, object] = {}
        self._param_widgets: dict[str, dict[str, object]] = {}
        self._rotate_preset_buttons: dict[int, ctk.CTkButton] = {}
        self._camera_window: CameraPreviewWindow | None = None
        self._camera_session: CameraSession | None = None
        self._live_active: bool = False
        self._live_session: CameraSession | None = None
        self._live_processor: LiveFrameProcessor | None = None
        self._live_poll_job: str | None = None
        self._live_last_seq: int = -1
        self._live_last_hist_gen: int = -1
        self._live_last_hist_time: float = 0.0
        self._live_hist_interval: float = 0.15
        self._live_after_frame: np.ndarray | None = None
        self._live_after_gen: int = -1
        self._live_after_err: str | None = None
        self._live_after_mode: str = "Single Filter"
        self._live_after_steps: int = 0
        self._live_chain: list[dict[str, object]] = []
        self._poll_job: str | None = None
        self._is_closing: bool = False

        self.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self._build_fonts()
        self._build_layout()

    def _build_fonts(self) -> None:
        """Menyiapkan tipografi aplikasi sesuai batas maksimal dua bobot."""
        self._font_section = ctk.CTkFont(
            family=self._font_family,
            size=theme.FONT_SIZES["section"],
            weight=theme.FONT_WEIGHTS["medium"],
        )
        self._font_label = ctk.CTkFont(
            family=self._font_family,
            size=theme.FONT_SIZES["label"],
            weight=theme.FONT_WEIGHTS["regular"],
        )
        self._font_label_medium = ctk.CTkFont(
            family=self._font_family,
            size=theme.FONT_SIZES["label"],
            weight=theme.FONT_WEIGHTS["medium"],
        )
        self._font_body = ctk.CTkFont(
            family=self._font_family,
            size=theme.FONT_SIZES["body"],
            weight=theme.FONT_WEIGHTS["regular"],
        )
        self._font_body_medium = ctk.CTkFont(
            family=self._font_family,
            size=theme.FONT_SIZES["body"],
            weight=theme.FONT_WEIGHTS["medium"],
        )

    def _build_layout(self) -> None:
        """Menyusun struktur utama: toolbar, area citra, histogram, kontrol, dan status bar."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Toolbar
        self.grid_rowconfigure(1, weight=1)  # Konten utama
        self.grid_rowconfigure(2, weight=0)  # Status bar

        self._build_toolbar()
        self._build_main_workspace()
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        """Membuat toolbar atas dengan tombol teks datar tanpa border."""
        toolbar_wrapper = ctk.CTkFrame(
            self,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=0,
            border_width=0,
        )
        toolbar_wrapper.grid(row=0, column=0, sticky="ew")
        toolbar_wrapper.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(
            toolbar_wrapper,
            fg_color="transparent",
            corner_radius=0,
            height=44,
        )
        toolbar.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=theme.SPACING["sm"],
            pady=theme.SPACING["xs"],
        )

        toolbar_items = [
            ("Open File", self._on_open_file, 92),
            ("Open Camera", self._on_open_camera, 98),
            ("Live Processing", self._toggle_live_processing, 118),
            ("Capture", self._on_capture_live, 76),
            ("Save File", self._on_save_file, 92),
            ("Reset", self._on_reset, 76),
        ]
        for idx, (text, command, width) in enumerate(toolbar_items):
            btn = ctk.CTkButton(
                toolbar,
                text=text,
                font=self._font_body_medium,
                fg_color="transparent",
                hover_color=theme.COLORS["bg_toolbar_hover"],
                text_color=theme.COLORS["text_primary"],
                border_width=0,
                corner_radius=theme.CORNER_RADIUS,
                height=30,
                width=width,
                cursor="hand2",
                command=command,
            )
            btn.grid(
                row=0,
                column=idx,
                padx=(0, theme.SPACING["xs"]),
            )
            self._toolbar_buttons[text] = btn

        separator = ctk.CTkFrame(
            toolbar_wrapper,
            fg_color=theme.COLORS["border"],
            height=theme.BORDER_WIDTH,
            corner_radius=0,
        )
        separator.grid(row=1, column=0, sticky="ew")

    def _build_main_workspace(self) -> None:
        """Membangun area tengah berisi panel citra, histogram, dan panel filter/parameter."""
        workspace = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        workspace.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["sm"],
            pady=theme.SPACING["sm"],
        )
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(0, weight=4, minsize=280)  # Panel Before/After
        workspace.grid_rowconfigure(1, weight=1, minsize=115)  # Panel Histogram
        workspace.grid_rowconfigure(2, weight=0)               # Filter & Parameter

        self._build_image_and_histogram_area(workspace)
        self._build_controls_area(workspace)

    def _build_image_and_histogram_area(self, parent: ctk.CTkFrame) -> None:
        """Membangun dua kolom simetris untuk panel Sebelum/Sesudah dan histogramnya."""
        compare_frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        compare_frame.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nsew",
            pady=(0, theme.SPACING["sm"]),
        )
        compare_frame.grid_columnconfigure(0, weight=1, uniform="compare_cols")
        compare_frame.grid_columnconfigure(1, weight=1, uniform="compare_cols")
        compare_frame.grid_rowconfigure(0, weight=4, minsize=280)
        compare_frame.grid_rowconfigure(1, weight=1, minsize=115)

        self._create_image_panel(
            compare_frame,
            key="before",
            title="Before",
            column=0,
            padx=(0, theme.SPACING["xs"]),
        )
        self._create_image_panel(
            compare_frame,
            key="after",
            title="After",
            column=1,
            padx=(theme.SPACING["xs"], 0),
        )

        self._hist_before = HistogramPanel(compare_frame, title="Original Histogram")
        self._hist_before.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, theme.SPACING["xs"]),
            pady=(theme.SPACING["xs"], 0),
        )
        self._hist_before.plot_dummy(variant="before")

        self._hist_after = HistogramPanel(compare_frame, title="Processed Histogram")
        self._hist_after.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(theme.SPACING["xs"], 0),
            pady=(theme.SPACING["xs"], 0),
        )
        self._hist_after.plot_dummy(variant="after")

    def _create_image_panel(
        self,
        parent: ctk.CTkFrame,
        key: str,
        title: str,
        column: int,
        padx: tuple[int, int],
    ) -> None:
        """Membuat satu panel citra dengan label sudut atas dan area pratinjau proporsional."""
        card = ctk.CTkFrame(
            parent,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=padx,
            pady=(0, theme.SPACING["xs"]),
        )
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        label = ctk.CTkLabel(
            card,
            text=title,
            font=self._font_label,
            text_color=theme.COLORS["text_secondary"],
            anchor="w",
        )
        label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["xs"], 4),
        )

        image_box = ctk.CTkFrame(
            card,
            fg_color=theme.COLORS["bg_image_area"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=0,
        )
        image_box.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["xs"],
            pady=(0, theme.SPACING["xs"]),
        )
        image_box.grid_propagate(False)
        image_box.grid_columnconfigure(0, weight=1)
        image_box.grid_rowconfigure(0, weight=1)

        img_label = tk.Label(
            image_box,
            bg=theme.COLORS["bg_image_area"],
            borderwidth=0,
            highlightthickness=0,
        )
        img_label.grid(row=0, column=0, sticky="nsew", padx=theme.SPACING["xs"], pady=theme.SPACING["xs"])

        self._image_boxes[key] = image_box
        self._image_labels[key] = img_label

        image_box.bind("<Configure>", lambda _e, k=key: self._schedule_image_resize(k))

    def _schedule_image_resize(self, key: str) -> None:
        """Menjadwalkan pembaruan skala pratinjau citra saat ukuran panel berubah."""
        if self._is_closing:
            return
        pending = self._resize_jobs.get(key)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
        self._resize_jobs[key] = self.after(30, lambda k=key: self._render_image_preview(k))

    def _render_image_preview(self, key: str) -> None:
        """Menampilkan pratinjau citra secara proporsional di tengah area citra."""
        self._resize_jobs[key] = None
        if self._is_closing:
            return
        box = self._image_boxes[key]
        label = self._image_labels[key]
        try:
            if not self.winfo_exists() or not box.winfo_exists() or not label.winfo_exists():
                return
        except tk.TclError:
            return
        source_img = self._display_sources[key]

        box_w = max(box.winfo_width() - theme.SPACING["sm"], 1)
        box_h = max(box.winfo_height() - theme.SPACING["sm"], 1)
        if box_w <= 10 or box_h <= 10:
            return

        src_w, src_h = source_img.size
        scale = min(box_w / src_w, box_h / src_h)
        target_w = max(int(round(src_w * scale)), 1)
        target_h = max(int(round(src_h * scale)), 1)

        render_sig = (target_w, target_h, self._image_versions[key])
        if self._last_rendered[key] == render_sig:
            return

        resample_mode = (
            Image.Resampling.BILINEAR
            if self._live_active
            else Image.Resampling.LANCZOS
        )
        resized = source_img.resize((target_w, target_h), resample_mode)
        photo = ImageTk.PhotoImage(resized)
        self._photo_refs[key] = photo
        self._last_rendered[key] = render_sig
        label.configure(image=photo, text="")

    @staticmethod
    def _bgr_to_display_pil(img_bgr: np.ndarray, max_preview_dim: int = 2000) -> Image.Image:
        """Mengonversi array BGR ke PIL RGB serta membatasi ukuran pratinjau citra besar."""
        height, width = img_bgr.shape[:2]
        longest = max(width, height)
        preview_bgr = img_bgr
        if longest > max_preview_dim:
            scale = max_preview_dim / longest
            new_w = max(int(round(width * scale)), 1)
            new_h = max(int(round(height * scale)), 1)
            preview_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb, mode="RGB")

    def _on_open_file(self) -> None:
        """Membuka dialog pemilihan berkas citra dan memuatnya ke panel Sebelum/Sesudah."""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Open File",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
            ],
        )
        if not file_path:
            return

        if self._live_active:
            self.stop_live_processing(restore_ui=False)
        self.load_image_from_path(file_path)

    @staticmethod
    def _normalize_image_array(raw_img: np.ndarray) -> tuple[np.ndarray, str]:
        """Menormalisasi array citra menjadi BGR uint8 3-channel beserta mode warnanya."""
        if not isinstance(raw_img, np.ndarray) or raw_img.size == 0 or raw_img.ndim not in (2, 3):
            raise ValueError("Invalid or corrupted image")

        img = raw_img
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), "Grayscale"

        channels = img.shape[2]
        if channels == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR), "Grayscale"
        if channels == 3:
            is_gray = (
                np.array_equal(img[:, :, 0], img[:, :, 1])
                and np.array_equal(img[:, :, 1], img[:, :, 2])
            )
            return img.copy(), ("Grayscale" if is_gray else "RGB")
        if channels == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            is_gray = (
                np.array_equal(bgr[:, :, 0], bgr[:, :, 1])
                and np.array_equal(bgr[:, :, 1], bgr[:, :, 2])
            )
            return bgr, ("Grayscale" if is_gray else "RGB")
        raise ValueError("Unsupported channel count")

    def load_image_from_path(self, file_path: str) -> bool:
        """Membaca citra dari disk dalam format BGR dan memperbarui tampilan bila valid."""
        try:
            if not os.path.isfile(file_path):
                raise ValueError("File not found")

            raw_bytes = np.fromfile(file_path, dtype=np.uint8)
            if raw_bytes.size == 0:
                raise ValueError("Empty file")

            decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError("Invalid or corrupted image")

            return self.load_image_array(decoded, filename=os.path.basename(file_path))
        except (OSError, ValueError, cv2.error):
            self._set_status_message("Error: failed to open image file", is_error=True)
            return False

    def load_image_array(self, raw_img: np.ndarray, filename: str) -> bool:
        """Memuat array citra sebagai sumber asli baru dan memperbarui seluruh panel."""
        if self._live_active:
            self.stop_live_processing(restore_ui=False)
            self._update_live_controls_ui()
        try:
            bgr, color_mode = self._normalize_image_array(raw_img)
        except (ValueError, cv2.error):
            self._set_status_message("Error: invalid image data", is_error=True)
            return False

        # Batalkan pekerjaan latar belakang lama bila ada, lalu simpan citra baru
        self._process_job_id += 1
        self._is_processing = False
        self._process_btn.configure(state="normal")

        # Simpan citra asli dan salinan aktif secara terpisah pada resolusi penuh
        self.original_image = bgr
        self.current_image = bgr.copy()
        self._chaining_started = False
        self._chaining_applied_filters.clear()
        self._update_filter_button_styles()
        self._current_filename = filename
        self._current_color_mode = color_mode

        display_pil = self._bgr_to_display_pil(self.original_image)
        self._display_sources["before"] = display_pil
        self._display_sources["after"] = display_pil.copy()
        self._image_versions["before"] += 1
        self._image_versions["after"] += 1

        self._render_image_preview("before")
        self._render_image_preview("after")
        self._hist_before.update_histogram(self.original_image)
        self._hist_after.update_histogram(self.current_image)

        height, width = self.original_image.shape[:2]
        self._info_label.configure(
            text=f"{self._current_filename}   {width} x {height}   {color_mode}"
        )
        self._set_status_message("Ready", is_error=False)
        return True

    def _on_open_camera(self) -> None:
        """Membuka jendela Capture webcam atau memfokuskan jendela yang sudah aktif."""
        if self._live_active:
            self.stop_live_processing(restore_ui=True)

        if self._camera_window is not None and self._camera_window.winfo_exists():
            self._camera_window.lift()
            self._camera_window.focus()
            return

        session = CameraSession(camera_index=0)
        try:
            session.open()
        except RuntimeError:
            session.close()
            self._camera_session = None
            self._camera_window = None
            self._set_status_message("Error: camera not found or unavailable", is_error=True)
            return

        self._camera_session = session
        self._camera_window = CameraPreviewWindow(
            master=self,
            session=session,
            on_capture=self._on_camera_captured,
            on_cancel=self._on_camera_cancelled,
        )

    def _toggle_live_processing(self) -> None:
        """Memulai atau menghentikan Live Processing di workspace utama."""
        if self._live_active:
            self.stop_live_processing(restore_ui=True)
        else:
            self.start_live_processing()

    def start_live_processing(self) -> bool:
        """Menjalankan kamera langsung ke panel Before/After dan histogram utama."""
        if self._live_active:
            return True

        if self._camera_window is not None and self._camera_window.winfo_exists():
            self._camera_window.close_window()
            self._camera_window = None
        if self._camera_session is not None:
            self._camera_session.close()
            self._camera_session = None

        # Batalkan proses statis yang mungkin masih berjalan tanpa mengubah citra tersimpan
        self._process_job_id += 1
        self._is_processing = False

        session = CameraSession(camera_index=0)
        try:
            session.open()
        except RuntimeError:
            session.close()
            self._set_status_message("Error: camera not found or unavailable", is_error=True)
            return False

        self._live_chain.clear()
        processor = LiveFrameProcessor(session)
        processor.set_pipeline(self.get_live_pipeline())
        processor.start()

        self._live_session = session
        self._live_processor = processor
        self._live_active = True
        self._live_last_seq = -1
        self._live_last_hist_gen = -1
        self._live_last_hist_time = 0.0
        self._live_after_frame = None
        self._live_after_gen = -1
        self._live_after_err = None
        self._live_after_mode = self._processing_mode
        self._live_after_steps = 0

        self._update_live_controls_ui()
        self._set_status_message("Starting live camera...", is_error=False)
        self._live_poll_job = self.after(33, self._poll_live_camera)
        return True

    def stop_live_processing(self, restore_ui: bool = True) -> None:
        """Menghentikan Live Processing, melepas kamera, dan memulihkan workspace."""
        was_active = self._live_active
        self._live_active = False
        self._live_after_frame = None
        self._live_after_gen = -1
        self._live_after_err = None
        self._live_after_steps = 0

        if self._live_poll_job is not None:
            try:
                self.after_cancel(self._live_poll_job)
            except tk.TclError:
                pass
            self._live_poll_job = None

        if self._live_processor is not None:
            self._live_processor.close()
            self._live_processor = None

        if self._live_session is not None:
            self._live_session.close()
            self._live_session = None

        self._live_chain.clear()

        if restore_ui and not self._is_closing and self.winfo_exists():
            self._update_live_controls_ui()
            if was_active:
                self._restore_workspace_after_live()

    def _restore_workspace_after_live(self) -> None:
        """Memulihkan tampilan Before/After, histogram, dan status bar setelah Live berhenti."""
        if self.original_image is not None and self.current_image is not None:
            self._display_sources["before"] = self._bgr_to_display_pil(self.original_image)
            self._display_sources["after"] = self._bgr_to_display_pil(self.current_image)
            self._image_versions["before"] += 1
            self._image_versions["after"] += 1
            self._render_image_preview("before")
            self._render_image_preview("after")
            self._hist_before.update_histogram(self.original_image)
            self._hist_after.update_histogram(self.current_image)
            height, width = self.original_image.shape[:2]
            fname = self._current_filename or "image"
            mode = self._current_color_mode or "RGB"
            self._info_label.configure(text=f"{fname}   {width} x {height}   {mode}")
        else:
            self._display_sources["before"] = self._create_dummy_image(variant="before")
            self._display_sources["after"] = self._create_dummy_image(variant="after")
            self._image_versions["before"] += 1
            self._image_versions["after"] += 1
            self._render_image_preview("before")
            self._render_image_preview("after")
            self._hist_before.plot_dummy(variant="before")
            self._hist_after.plot_dummy(variant="after")
            self._info_label.configure(text="sample_image.jpg   1280 x 720   RGB")

        self._set_status_message("Ready", is_error=False)

    def _poll_live_camera(self) -> None:
        """Memperbarui panel Before/After dan histogram secara terkendali saat Live aktif."""
        self._live_poll_job = None
        if not self._live_active or self._is_closing:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        if self._live_session is None or self._live_processor is None:
            return

        _raw_peek, cam_err = self._live_session.get_state()
        if cam_err is not None:
            self.stop_live_processing(restore_ui=True)
            self._set_status_message(cam_err, is_error=True)
            return

        self._live_processor.trigger_if_idle()

        raw_bgr, proc_bgr, proc_err, seq, gen, _latency_ms = self._live_processor.get_result()
        result_mode, result_steps = self._live_processor.get_result_meta()
        if raw_bgr is not None and proc_bgr is not None and seq != self._live_last_seq:
            self._live_last_seq = seq
            self._live_after_frame = proc_bgr.copy()
            self._live_after_gen = gen
            self._live_after_err = proc_err
            self._live_after_mode = result_mode
            self._live_after_steps = result_steps

            self._display_sources["before"] = self._bgr_to_display_pil(raw_bgr)
            self._display_sources["after"] = self._bgr_to_display_pil(proc_bgr)
            self._image_versions["before"] += 1
            self._image_versions["after"] += 1
            self._render_image_preview("before")
            self._render_image_preview("after")

            h, w = proc_bgr.shape[:2]
            is_gray = (
                np.array_equal(proc_bgr[:, :, 0], proc_bgr[:, :, 1])
                and np.array_equal(proc_bgr[:, :, 1], proc_bgr[:, :, 2])
            )
            mode_label = "Grayscale" if is_gray else "RGB"
            self._info_label.configure(text=f"Live Camera   {w} x {h}   {mode_label}")

            now = time.perf_counter()
            if (
                gen != self._live_last_hist_gen
                or (now - self._live_last_hist_time) >= self._live_hist_interval
            ):
                self._hist_before.update_histogram(raw_bgr)
                self._hist_after.update_histogram(proc_bgr)
                self._live_last_hist_time = now
                self._live_last_hist_gen = gen

            if proc_err is not None:
                self._set_status_message(f"Live filter error: {proc_err}", is_error=True)
            elif self._processing_mode == "Filter Chaining":
                count = len(self._live_chain)
                msg = (
                    f"Live Processing (Filter Chaining): {count} filter(s)"
                    if count > 0
                    else "Live Processing (Filter Chaining): chain is empty"
                )
                self._set_status_message(msg, is_error=False)
            else:
                self._set_status_message(
                    f"Live Processing: {self._selected_filter}",
                    is_error=False,
                )

        if self._live_active and not self._is_closing:
            self._live_poll_job = self.after(33, self._poll_live_camera)

    def get_live_pipeline(
        self,
    ) -> tuple[str, list[tuple[str, object, dict[str, object]]], str | None]:
        """Mengembalikan snapshot mode dan daftar filter aktif untuk pemrosesan kamera live."""
        if self._processing_mode == "Filter Chaining":
            steps = [
                (str(item["name"]), item["func"], dict(item["params"]))
                for item in self._live_chain
            ]
            return "Filter Chaining", steps, None

        if self._selected_filter not in FILTER_SPECS:
            return "Single Filter", [], "No filter selected"

        try:
            params = self._collect_filter_params()
        except (TypeError, ValueError) as exc:
            return "Single Filter", [], str(exc)

        return (
            "Single Filter",
            [(self._selected_filter, FILTER_SPECS[self._selected_filter]["func"], params)],
            None,
        )

    def add_selected_filter_to_live_chain(self) -> tuple[bool, str]:
        """Menambahkan filter terpilih beserta parameternya ke rantai pemrosesan langsung."""
        if self._selected_filter not in FILTER_SPECS:
            return False, "No filter selected"

        if any(item["name"] == self._selected_filter for item in self._live_chain):
            return False, f"'{self._selected_filter}' is already in the live chain"

        try:
            params = self._collect_filter_params()
        except (TypeError, ValueError) as exc:
            return False, str(exc)

        self._live_chain.append(
            {
                "name": self._selected_filter,
                "func": FILTER_SPECS[self._selected_filter]["func"],
                "params": params,
            }
        )
        self._update_live_controls_ui()
        self._notify_live_pipeline_changed()
        return True, self._selected_filter

    def _on_add_live_filter(self) -> None:
        """Handler tombol Add Filter pada workspace utama saat mode Live + Chaining."""
        added, info = self.add_selected_filter_to_live_chain()
        if not added:
            self._set_status_message(info, is_error=True)

    def clear_live_chain(self) -> None:
        """Mengosongkan daftar rantai filter kamera langsung."""
        self._live_chain.clear()
        self._update_live_controls_ui()
        self._notify_live_pipeline_changed()

    def get_live_chain_names(self) -> list[str]:
        """Mengembalikan urutan nama filter pada rantai pemrosesan langsung."""
        return [str(item["name"]) for item in self._live_chain]

    def _notify_live_pipeline_changed(self) -> None:
        """Menyinkronkan perubahan filter, parameter, atau mode ke worker Live Processing."""
        if self._live_active and self._live_processor is not None:
            self._live_processor.set_pipeline(self.get_live_pipeline())

    def _on_capture_live(self) -> None:
        """Menangkap frame hasil pemrosesan Live Processing ke alur kerja utama aplikasi."""
        if not self._live_active or self._live_processor is None or self._live_session is None:
            self._set_status_message("Error: live processing is not active", is_error=True)
            return

        _raw_peek, cam_err = self._live_session.get_state()
        if cam_err is not None:
            self._set_status_message(cam_err, is_error=True)
            return

        if self._processing_mode == "Single Filter":
            if self._selected_filter not in FILTER_SPECS:
                self._set_status_message(
                    "Error: please select a filter before capturing",
                    is_error=True,
                )
                return
        elif len(self._live_chain) == 0:
            self._set_status_message(
                "Error: please add a filter to the live chain first",
                is_error=True,
            )
            return

        current_gen = self._live_processor.current_pipeline_gen
        if self._live_after_frame is None or self._live_after_gen != current_gen:
            if self._live_after_err is not None and self._live_after_gen == current_gen:
                self._set_status_message(f"Error: {self._live_after_err}", is_error=True)
            else:
                self._set_status_message(
                    "Error: processed frame is not ready for current settings",
                    is_error=True,
                )
            return

        if self._live_after_err is not None:
            self._set_status_message(f"Error: {self._live_after_err}", is_error=True)
            return

        if self._live_after_steps <= 0 or self._live_after_mode != self._processing_mode:
            self._set_status_message("Error: no processed filter output available", is_error=True)
            return

        captured_bgr = self._live_after_frame.copy()
        self.stop_live_processing(restore_ui=False)
        loaded = self.load_image_array(captured_bgr, filename="live_capture.png")
        self._update_live_controls_ui()
        if not loaded:
            self._restore_workspace_after_live()
            self._set_status_message("Error: failed to load captured frame", is_error=True)

    def _on_camera_captured(self, frame_bgr: np.ndarray) -> None:
        """Memuat frame hasil tangkapan kamera ke alur kerja utama aplikasi."""
        self._camera_window = None
        self._camera_session = None
        self.load_image_array(frame_bgr, filename="camera_capture.png")

    def _on_camera_cancelled(self) -> None:
        """Membersihkan referensi jendela kamera setelah pengguna membatalkan pratinjau."""
        self._camera_window = None
        if self._camera_session is not None:
            self._camera_session.close()
            self._camera_session = None

    def _on_app_close(self) -> None:
        """Memastikan resource kamera dan callback tertunda dibersihkan sebelum menutup jendela."""
        self._is_closing = True
        self._process_job_id += 1
        self._is_processing = False

        self.stop_live_processing(restore_ui=False)

        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None

        for key, job in list(self._resize_jobs.items()):
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
                self._resize_jobs[key] = None

        if self._camera_window is not None:
            self._camera_window.close_window()
            self._camera_window = None
        if self._camera_session is not None:
            self._camera_session.close()
            self._camera_session = None
        self.destroy()

    def _on_save_file(self) -> None:
        """Membuka dialog simpan berkas dan menyimpan citra hasil beresolusi penuh."""
        if self.current_image is None:
            self._set_status_message("Error: please open an image first", is_error=True)
            return

        stem = os.path.splitext(self._current_filename or "image")[0]
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save File",
            defaultextension=".png",
            initialfile=f"{stem}_processed.png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
            ],
        )
        if not file_path:
            return

        self.save_image_to_path(file_path)

    def save_image_to_path(self, file_path: str) -> bool:
        """Mengenkode dan menyimpan salinan current_image pada resolusi aslinya."""
        if self.current_image is None:
            self._set_status_message("Error: please open an image first", is_error=True)
            return False

        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            ext = ".png"
            file_path = f"{file_path}.png"

        try:
            img_to_save = self.current_image.copy()
            if ext == ".png":
                ok, encoded = cv2.imencode(".png", img_to_save)
            elif ext in (".jpg", ".jpeg"):
                ok, encoded = cv2.imencode(
                    ".jpg",
                    img_to_save,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
            else:
                raise ValueError("Unsupported save format")

            if not ok:
                raise ValueError("Failed to encode image")

            with open(file_path, "wb") as out_file:
                out_file.write(encoded.tobytes())
        except (OSError, ValueError, cv2.error):
            self._set_status_message("Error: failed to save image file", is_error=True)
            return False

        self._set_status_message(f"Saved: {os.path.basename(file_path)}", is_error=False)
        return True

    def _on_reset(self) -> None:
        """Mengembalikan citra hasil ke citra asli tanpa mengubah mode atau filter terpilih."""
        if self.original_image is None:
            self._set_status_message("Error: please open an image first", is_error=True)
            return

        # Invalidasi hasil worker yang sedang berjalan agar tidak menimpa hasil Reset
        self._process_job_id += 1
        self._is_processing = False
        self._process_btn.configure(state="normal")

        self.current_image = self.original_image.copy()
        self._chaining_started = False
        self._chaining_applied_filters.clear()
        self._update_filter_button_styles()

        self._display_sources["after"] = self._bgr_to_display_pil(self.current_image)
        self._image_versions["after"] += 1
        self._render_image_preview("after")
        self._hist_after.update_histogram(self.current_image)

        self._set_status_message("Ready", is_error=False)

    def _set_status_message(self, message: str, is_error: bool = False) -> None:
        """Memperbarui teks status di sudut kanan status bar."""
        color = (
            theme.COLORS["error_text"]
            if is_error
            else theme.COLORS["text_secondary"]
        )
        self._state_label.configure(text=message, text_color=color)

    @staticmethod
    def _create_dummy_image(variant: str = "before") -> Image.Image:
        """Membuat gambar contoh netral berasio 16:9 untuk tata letak statis Fase 2."""
        width, height = 640, 360
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

        # Lapisan gradien halus bergaya lanskap minimalis untuk contoh uji
        r = 165 + 45 * (1.0 - y) + 22 * x
        g = 172 + 38 * (1.0 - y) - 10 * x
        b = 168 + 32 * (1.0 - y) - 18 * x

        # Siluet bukit sederhana agar pratinjau terlihat seperti citra alami
        horizon = 0.56 + 0.06 * np.sin(x * np.pi * 2.2)
        hill_mask = y > horizon
        r = np.where(hill_mask, 82 + 28 * (1.0 - y), r)
        g = np.where(hill_mask, 108 + 34 * (1.0 - y), g)
        b = np.where(hill_mask, 96 + 24 * (1.0 - y), b)

        if variant == "after":
            # Simulasi hasil sedikit lebih cerah dan kontras
            r = r * 1.08 + 8
            g = g * 1.06 + 6
            b = b * 1.04 + 4

        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")

        draw = ImageDraw.Draw(img)
        sun_center = (490, 96) if variant == "before" else (490, 92)
        sun_radius = 28
        draw.ellipse(
            (
                sun_center[0] - sun_radius,
                sun_center[1] - sun_radius,
                sun_center[0] + sun_radius,
                sun_center[1] + sun_radius,
            ),
            fill=(228, 224, 210),
        )
        return img

    def _build_controls_area(self, parent: ctk.CTkFrame) -> None:
        """Membangun area bawah berisi daftar filter (kiri) dan parameter + Process (kanan)."""
        controls_frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        controls_frame.grid(row=2, column=0, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=15)
        controls_frame.grid_columnconfigure(1, weight=8)
        controls_frame.grid_rowconfigure(0, weight=1)

        self._build_filter_panel(controls_frame)
        self._build_parameter_panel(controls_frame)

    def _build_filter_panel(self, parent: ctk.CTkFrame) -> None:
        """Membangun panel FILTER dengan 5 kelompok kategori dan 15 tombol filter."""
        filter_card = ctk.CTkFrame(
            parent,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
        )
        filter_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, theme.SPACING["xs"]),
        )
        filter_card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            filter_card,
            text="FILTER",
            font=self._font_section,
            text_color=theme.COLORS["text_primary"],
            anchor="w",
        )
        header.grid(
            row=0,
            column=0,
            sticky="w",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["xs"], theme.SPACING["xs"]),
        )

        grid_frame = ctk.CTkFrame(filter_card, fg_color="transparent", corner_radius=0)
        grid_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["sm"],
            pady=(0, theme.SPACING["sm"]),
        )

        for col_idx, (group_name, filter_names) in enumerate(FILTER_GROUPS):
            grid_frame.grid_columnconfigure(col_idx, weight=1, uniform="filter_cols")

            group_label = ctk.CTkLabel(
                grid_frame,
                text=group_name,
                font=self._font_label,
                text_color=theme.COLORS["text_secondary"],
                anchor="w",
            )
            group_label.grid(
                row=0,
                column=col_idx,
                sticky="w",
                padx=(0 if col_idx == 0 else 4, 0 if col_idx == 4 else 4),
                pady=(0, 4),
            )

            for row_offset, filter_name in enumerate(filter_names, start=1):
                btn = ctk.CTkButton(
                    grid_frame,
                    text=filter_name,
                    font=self._font_label,
                    fg_color=theme.COLORS["bg_panel"],
                    hover_color=theme.COLORS["bg_window"],
                    text_color=theme.COLORS["text_primary"],
                    border_width=theme.BORDER_WIDTH,
                    border_color=theme.COLORS["border"],
                    corner_radius=theme.CORNER_RADIUS,
                    height=30,
                    cursor="hand2",
                    command=lambda name=filter_name: self.select_filter(name),
                )
                btn.grid(
                    row=row_offset,
                    column=col_idx,
                    sticky="ew",
                    padx=(0 if col_idx == 0 else 4, 0 if col_idx == 4 else 4),
                    pady=(0, 6 if row_offset < len(filter_names) else 0),
                )
                self._filter_buttons[filter_name] = btn

        self._update_filter_button_styles()

    def _build_parameter_panel(self, parent: ctk.CTkFrame) -> None:
        """Membangun panel PARAMETER dinamis, pemilih mode, dan tombol Process."""
        param_card = ctk.CTkFrame(
            parent,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
        )
        param_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(theme.SPACING["xs"], 0),
        )
        param_card.grid_columnconfigure(0, weight=1)
        param_card.grid_rowconfigure(1, weight=1)

        header_row = ctk.CTkFrame(param_card, fg_color="transparent", corner_radius=0)
        header_row.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["xs"], 4),
        )
        header_row.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            header_row,
            text="PARAMETER",
            font=self._font_section,
            text_color=theme.COLORS["text_primary"],
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="w")

        mode_frame = ctk.CTkFrame(header_row, fg_color="transparent", corner_radius=0)
        mode_frame.grid(row=0, column=1, sticky="e")

        for idx, mode_name in enumerate(PROCESSING_MODES):
            mode_btn = ctk.CTkButton(
                mode_frame,
                text=mode_name,
                font=self._font_label,
                fg_color=theme.COLORS["bg_panel"],
                hover_color=theme.COLORS["bg_window"],
                text_color=theme.COLORS["text_secondary"],
                border_width=theme.BORDER_WIDTH,
                border_color=theme.COLORS["border"],
                corner_radius=theme.CORNER_RADIUS,
                height=24,
                width=92,
                cursor="hand2",
                command=lambda m=mode_name: self.set_processing_mode(m),
            )
            mode_btn.grid(row=0, column=idx, padx=(0 if idx == 0 else 4, 0))
            self._mode_buttons[mode_name] = mode_btn

        self._update_mode_button_styles()

        # Area kontrol parameter dinamis sesuai filter yang dipilih
        self._param_body = ctk.CTkFrame(param_card, fg_color="transparent", corner_radius=0)
        self._param_body.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["sm"],
            pady=(2, theme.SPACING["xs"]),
        )
        self._param_body.grid_columnconfigure(0, weight=1)

        # Tombol Process di kanan bawah (satu-satunya tombol beraksen penuh)
        action_row = ctk.CTkFrame(param_card, fg_color="transparent", corner_radius=0)
        action_row.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=theme.SPACING["sm"],
            pady=(0, theme.SPACING["sm"]),
        )
        action_row.grid_columnconfigure(0, weight=1)

        self._live_chain_controls = ctk.CTkFrame(
            action_row,
            fg_color="transparent",
            corner_radius=0,
        )
        self._live_chain_controls.grid_columnconfigure(2, weight=1)

        self._add_live_btn = ctk.CTkButton(
            self._live_chain_controls,
            text="Add Filter",
            font=self._font_label_medium,
            fg_color=theme.COLORS["bg_panel"],
            hover_color=theme.COLORS["bg_window"],
            text_color=theme.COLORS["accent"],
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["accent"],
            corner_radius=theme.CORNER_RADIUS,
            height=30,
            width=86,
            cursor="hand2",
            command=self._on_add_live_filter,
        )
        self._add_live_btn.grid(row=0, column=0, padx=(0, 6))

        self._clear_live_btn = ctk.CTkButton(
            self._live_chain_controls,
            text="Clear Chain",
            font=self._font_label,
            fg_color=theme.COLORS["bg_panel"],
            hover_color=theme.COLORS["bg_window"],
            text_color=theme.COLORS["text_primary"],
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
            corner_radius=theme.CORNER_RADIUS,
            height=30,
            width=86,
            cursor="hand2",
            command=self.clear_live_chain,
        )
        self._clear_live_btn.grid(row=0, column=1, padx=(0, theme.SPACING["xs"]))

        self._live_chain_label = ctk.CTkLabel(
            self._live_chain_controls,
            text="Chain: Empty",
            font=self._font_label,
            text_color=theme.COLORS["text_secondary"],
            anchor="w",
        )
        self._live_chain_label.grid(row=0, column=2, sticky="ew")

        self._process_btn = ctk.CTkButton(
            action_row,
            text="Process",
            font=self._font_body_medium,
            fg_color=theme.COLORS["accent"],
            hover_color=theme.COLORS["accent_hover"],
            text_color=theme.COLORS["accent_text"],
            border_width=0,
            corner_radius=theme.CORNER_RADIUS,
            height=34,
            width=112,
            cursor="hand2",
            command=self._on_process,
        )
        self._process_btn.grid(row=0, column=1, sticky="e")

        self.select_filter(self._selected_filter, force_reset=True)
        self._update_live_controls_ui()

    def _update_live_controls_ui(self) -> None:
        """Menyesuaikan kontrol toolbar dan panel parameter dengan status Live Processing."""
        live_btn = self._toolbar_buttons.get("Live Processing")
        if live_btn is not None:
            if self._live_active:
                live_btn.configure(
                    text="Stop Live",
                    text_color=theme.COLORS["accent"],
                    border_width=theme.BORDER_WIDTH,
                    border_color=theme.COLORS["accent"],
                )
            else:
                live_btn.configure(
                    text="Live Processing",
                    text_color=theme.COLORS["text_primary"],
                    border_width=0,
                )

        capture_btn = self._toolbar_buttons.get("Capture")
        if capture_btn is not None:
            if self._live_active:
                capture_btn.configure(
                    state="normal",
                    text_color=theme.COLORS["accent"],
                    border_width=theme.BORDER_WIDTH,
                    border_color=theme.COLORS["accent"],
                )
            else:
                capture_btn.configure(
                    state="disabled",
                    text_color=theme.COLORS["text_primary"],
                    border_width=0,
                )

        for key in ("Save File", "Reset"):
            btn = self._toolbar_buttons.get(key)
            if btn is not None:
                btn.configure(state="disabled" if self._live_active else "normal")

        if self._live_active:
            self._process_btn.configure(state="disabled")
            if self._processing_mode == "Filter Chaining":
                names = self.get_live_chain_names()
                summary = " -> ".join(names) if names else "Empty"
                self._live_chain_label.configure(text=f"Chain: {summary}")
                self._live_chain_controls.grid(
                    row=0,
                    column=0,
                    sticky="ew",
                    padx=(0, theme.SPACING["xs"]),
                )
            else:
                self._live_chain_controls.grid_forget()
        else:
            self._live_chain_controls.grid_forget()
            self._process_btn.configure(
                state="disabled" if self._is_processing else "normal"
            )

    def _update_filter_button_styles(self) -> None:
        """Menyoroti filter terpilih serta riwayat filter aktif pada mode Filter Chaining."""
        if self._processing_mode == "Filter Chaining":
            highlighted = self._chaining_applied_filters | {self._selected_filter}
        else:
            highlighted = {self._selected_filter}

        for name, btn in self._filter_buttons.items():
            is_highlighted = name in highlighted
            btn.configure(
                font=self._font_label_medium if is_highlighted else self._font_label,
                text_color=(
                    theme.COLORS["accent"]
                    if is_highlighted
                    else theme.COLORS["text_primary"]
                ),
                border_color=(
                    theme.COLORS["accent"]
                    if is_highlighted
                    else theme.COLORS["border"]
                ),
            )

    def _update_mode_button_styles(self) -> None:
        """Memperbarui indikator visual mode pemrosesan aktif."""
        for mode_name, btn in self._mode_buttons.items():
            is_selected = mode_name == self._processing_mode
            btn.configure(
                font=self._font_label_medium if is_selected else self._font_label,
                text_color=(
                    theme.COLORS["accent"]
                    if is_selected
                    else theme.COLORS["text_secondary"]
                ),
                border_color=(
                    theme.COLORS["accent"]
                    if is_selected
                    else theme.COLORS["border"]
                ),
            )

    def set_processing_mode(self, mode: str) -> None:
        """Mengganti mode pemrosesan tanpa mengubah citra atau menjalankan filter."""
        if mode not in PROCESSING_MODES:
            return
        if mode != self._processing_mode:
            self._processing_mode = mode
            self._chaining_started = False
            self._chaining_applied_filters.clear()
            self._live_chain.clear()
            self._update_mode_button_styles()
            self._update_filter_button_styles()
            self._update_live_controls_ui()
            self._notify_live_pipeline_changed()

    def select_filter(self, filter_name: str, force_reset: bool = False) -> None:
        """Memilih satu filter dan mengatur ulang kontrol parameter ke nilai default."""
        if filter_name not in FILTER_SPECS:
            return
        if not force_reset and filter_name == self._selected_filter and self._param_widgets:
            return

        self._selected_filter = filter_name
        self._update_filter_button_styles()

        spec = FILTER_SPECS[filter_name]
        self._param_values = {
            param["key"]: param["default"] for param in spec["params"]
        }
        self._render_parameter_controls()
        self._notify_live_pipeline_changed()

    def _render_parameter_controls(self) -> None:
        """Membangun ulang kontrol parameter dinamis sesuai filter yang aktif."""
        for child in self._param_body.winfo_children():
            child.destroy()
        self._param_widgets.clear()
        self._rotate_preset_buttons.clear()

        spec = FILTER_SPECS[self._selected_filter]
        param_defs = spec["params"]

        if not param_defs:
            empty_label = ctk.CTkLabel(
                self._param_body,
                text="No parameters for this filter.",
                font=self._font_label,
                text_color=theme.COLORS["text_secondary"],
                anchor="w",
            )
            empty_label.grid(row=0, column=0, sticky="w", pady=(theme.SPACING["xs"], 0))
            return

        row_idx = 0
        for pdef in param_defs:
            condition = pdef.get("visible_when")
            if condition is not None:
                dep_key, dep_val = condition
                if self._param_values.get(dep_key) != dep_val:
                    continue

            key = pdef["key"]
            kind = pdef["kind"]

            if kind == "dropdown":
                row_frame = ctk.CTkFrame(self._param_body, fg_color="transparent", corner_radius=0)
                row_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 6))
                row_frame.grid_columnconfigure(1, weight=1)

                lbl = ctk.CTkLabel(
                    row_frame,
                    text=pdef["label"],
                    font=self._font_label,
                    text_color=theme.COLORS["text_secondary"],
                    width=84,
                    anchor="w",
                )
                lbl.grid(row=0, column=0, sticky="w", padx=(0, theme.SPACING["xs"]))

                var = ctk.StringVar(value=str(self._param_values[key]))
                menu = ctk.CTkOptionMenu(
                    row_frame,
                    values=pdef["options"],
                    variable=var,
                    font=self._font_label,
                    dropdown_font=self._font_label,
                    fg_color=theme.COLORS["bg_window"],
                    button_color=theme.COLORS["border"],
                    button_hover_color=theme.COLORS["bg_toolbar_hover"],
                    text_color=theme.COLORS["text_primary"],
                    dropdown_fg_color=theme.COLORS["bg_panel"],
                    dropdown_hover_color=theme.COLORS["bg_toolbar_hover"],
                    dropdown_text_color=theme.COLORS["text_primary"],
                    corner_radius=theme.CORNER_RADIUS,
                    height=26,
                    cursor="hand2",
                    command=lambda val, k=key, r=bool(pdef.get("refresh")): self._on_dropdown_change(k, val, r),
                )
                menu.grid(row=0, column=1, sticky="ew")
                self._param_widgets[key] = {"widget": menu, "var": var}
                row_idx += 1
                continue

            # Kontrol slider (int, odd, float) dengan nilai numerik di kanan
            row_frame = ctk.CTkFrame(self._param_body, fg_color="transparent", corner_radius=0)
            row_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 6))
            row_frame.grid_columnconfigure(1, weight=1)

            lbl = ctk.CTkLabel(
                row_frame,
                text=pdef["label"],
                font=self._font_label,
                text_color=theme.COLORS["text_secondary"],
                width=84,
                anchor="w",
            )
            lbl.grid(row=0, column=0, sticky="w", padx=(0, theme.SPACING["xs"]))

            min_val = pdef["min"]
            max_val = pdef["max"]
            if kind == "int":
                steps = int(max_val - min_val)
            elif kind == "odd":
                steps = int((max_val - min_val) // 2)
            else:
                steps = int(round((max_val - min_val) / pdef["step"]))

            val_label = ctk.CTkLabel(
                row_frame,
                text=self._format_param_value(pdef, self._param_values[key]),
                font=self._font_label_medium,
                text_color=theme.COLORS["text_primary"],
                width=42,
                anchor="e",
            )
            val_label.grid(row=0, column=2, sticky="e")

            slider = ctk.CTkSlider(
                row_frame,
                from_=min_val,
                to=max_val,
                number_of_steps=max(steps, 1),
                height=14,
                fg_color=theme.COLORS["border"],
                progress_color=theme.COLORS["accent"],
                button_color=theme.COLORS["accent"],
                button_hover_color=theme.COLORS["accent_hover"],
                cursor="hand2",
                command=lambda raw, d=pdef, vl=val_label: self._on_slider_change(d, raw, vl),
            )
            slider.set(float(self._param_values[key]))
            slider.grid(row=0, column=1, sticky="ew", padx=(0, theme.SPACING["xs"]))
            self._param_widgets[key] = {"widget": slider, "value_label": val_label}
            row_idx += 1

        if self._selected_filter == "Rotate":
            self._build_rotate_presets(row_idx)

    def _build_rotate_presets(self, start_row: int) -> None:
        """Menambahkan 4 tombol preset sudut dalam tata letak 2 kolom untuk filter Rotate."""
        presets_frame = ctk.CTkFrame(self._param_body, fg_color="transparent", corner_radius=0)
        presets_frame.grid(row=start_row, column=0, sticky="ew", pady=(2, 0))
        presets_frame.grid_columnconfigure(0, weight=1, uniform="rotate_preset_cols")
        presets_frame.grid_columnconfigure(1, weight=1, uniform="rotate_preset_cols")

        presets = [
            ("Rotate 90°", 90),
            ("Rotate 180°", 180),
            ("Rotate -90°", -90),
            ("Rotate -180°", -180),
        ]
        for idx, (label_text, angle_val) in enumerate(presets):
            r = idx // 2
            c = idx % 2
            btn = ctk.CTkButton(
                presets_frame,
                text=label_text,
                font=self._font_label,
                fg_color=theme.COLORS["bg_panel"],
                hover_color=theme.COLORS["bg_window"],
                text_color=theme.COLORS["text_primary"],
                border_width=theme.BORDER_WIDTH,
                border_color=theme.COLORS["border"],
                corner_radius=theme.CORNER_RADIUS,
                height=26,
                cursor="hand2",
                command=lambda val=angle_val: self._apply_rotate_preset(val),
            )
            btn.grid(
                row=r,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 4 if c == 0 else 0),
                pady=(0, 4 if r == 0 else 0),
            )
            self._rotate_preset_buttons[angle_val] = btn

    def _apply_rotate_preset(self, angle: int) -> None:
        """Mengatur slider dan tampilan nilai Angle sesuai preset tanpa langsung memproses."""
        angle_entry = self._param_widgets.get("angle")
        if not angle_entry:
            return
        slider = angle_entry["widget"]
        val_label = angle_entry["value_label"]
        self._param_values["angle"] = int(angle)
        slider.set(float(angle))
        val_label.configure(text=str(int(angle)))
        self._notify_live_pipeline_changed()

    @staticmethod
    def _snap_odd_kernel(value: float, min_val: int, max_val: int) -> int:
        """Membulatkan nilai kernel agar selalu bilangan bulat ganjil pada rentang valid."""
        k = int(round(value))
        k = max(min_val, min(max_val, k))
        if k % 2 == 0:
            k = k + 1 if k < max_val else k - 1
        return k

    @staticmethod
    def _format_param_value(pdef: dict[str, object], value: object) -> str:
        """Memformat tampilan nilai parameter pada sisi kanan slider."""
        kind = pdef["kind"]
        if kind in ("int", "odd"):
            return str(int(round(float(value))))
        decimals = int(pdef.get("decimals", 1))
        return f"{float(value):.{decimals}f}"

    def _on_slider_change(
        self,
        pdef: dict[str, object],
        raw_value: float,
        value_label: ctk.CTkLabel,
    ) -> None:
        """Memperbarui nilai parameter saat slider digeser."""
        key = pdef["key"]
        kind = pdef["kind"]
        if kind == "int":
            parsed: object = int(round(raw_value))
        elif kind == "odd":
            parsed = self._snap_odd_kernel(raw_value, int(pdef["min"]), int(pdef["max"]))
        else:
            decimals = int(pdef.get("decimals", 2))
            parsed = round(float(raw_value), decimals)

        self._param_values[key] = parsed
        value_label.configure(text=self._format_param_value(pdef, parsed))
        self._notify_live_pipeline_changed()

    def _on_dropdown_change(self, key: str, value: str, refresh: bool) -> None:
        """Menyimpan pilihan dropdown dan menyegarkan parameter kondisional bila perlu."""
        self._param_values[key] = value
        if refresh:
            self._render_parameter_controls()
        self._notify_live_pipeline_changed()

    def _collect_filter_params(self) -> dict[str, object]:
        """Mengumpulkan serta memvalidasi argumen kata kunci sesuai tanda tangan filters.py."""
        spec = FILTER_SPECS[self._selected_filter]
        kwargs: dict[str, object] = {}
        for pdef in spec["params"]:
            condition = pdef.get("visible_when")
            if condition is not None:
                dep_key, dep_val = condition
                if self._param_values.get(dep_key) != dep_val:
                    continue

            key = pdef["key"]
            kind = pdef["kind"]
            val = self._param_values.get(key, pdef["default"])

            if kind == "odd":
                k = int(round(float(val)))
                min_v, max_v = int(pdef["min"]), int(pdef["max"])
                if k < min_v or k > max_v or k % 2 == 0:
                    raise ValueError(
                        f"Kernel size must be an odd integer between {min_v} and {max_v}."
                    )
                kwargs[key] = k
            elif kind == "int":
                kwargs[key] = int(round(float(val)))
            elif kind == "float":
                kwargs[key] = float(val)
            else:
                kwargs[key] = str(val)

        return kwargs

    def _on_process(self) -> None:
        """Memvalidasi kondisi dan menjalankan filter terpilih di thread latar belakang."""
        if self._is_processing or self._live_active:
            return

        if self.original_image is None:
            self._set_status_message("Error: please open an image first", is_error=True)
            return

        if self._selected_filter not in FILTER_SPECS:
            self._set_status_message("Error: please select a filter first", is_error=True)
            return

        try:
            params = self._collect_filter_params()
        except (TypeError, ValueError) as exc:
            self._set_status_message(f"Error: {exc}", is_error=True)
            return

        if (
            self._processing_mode == "Filter Chaining"
            and self._chaining_started
            and self.current_image is not None
        ):
            source_copy = self.current_image.copy()
        else:
            source_copy = self.original_image.copy()

        filter_name = self._selected_filter
        mode_snapshot = self._processing_mode
        filter_func = FILTER_SPECS[filter_name]["func"]
        self._is_processing = True
        self._process_job_id += 1
        job_id = self._process_job_id

        self._process_btn.configure(state="disabled")
        self._set_status_message("Processing...", is_error=False)

        worker = threading.Thread(
            target=self._run_filter_worker,
            args=(job_id, filter_name, mode_snapshot, filter_func, source_copy, params),
            daemon=True,
        )
        worker.start()
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
        self._poll_job = self.after(20, self._poll_process_queue)

    def _run_filter_worker(
        self,
        job_id: int,
        filter_name: str,
        mode_snapshot: str,
        filter_func,
        source_copy: np.ndarray,
        params: dict[str, object],
    ) -> None:
        """Menjalankan fungsi pengolahan citra di worker thread tanpa menyentuh Tkinter."""
        result: np.ndarray | None = None
        error_msg: str | None = None
        try:
            result = filter_func(source_copy, **params)
        except Exception as exc:
            error_msg = str(exc) or "Image processing failed"

        self._process_queue.put((job_id, filter_name, mode_snapshot, result, error_msg))

    def _poll_process_queue(self) -> None:
        """Memeriksa antrean hasil pemrosesan dari thread utama GUI."""
        self._poll_job = None
        if self._is_closing:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        while True:
            try:
                job_id, filter_name, mode_snapshot, result, error_msg = (
                    self._process_queue.get_nowait()
                )
            except queue.Empty:
                break

            self._on_process_complete(
                job_id, filter_name, mode_snapshot, result, error_msg
            )

        if self._is_processing and not self._is_closing:
            self._poll_job = self.after(20, self._poll_process_queue)

    def _on_process_complete(
        self,
        job_id: int,
        filter_name: str,
        mode_snapshot: str,
        result: np.ndarray | None,
        error_msg: str | None,
    ) -> None:
        """Menerima hasil pemrosesan di thread utama GUI dan memperbarui panel Sesudah."""
        if self._is_closing or job_id != self._process_job_id:
            return

        self._is_processing = False
        self._process_btn.configure(state="normal")

        if error_msg is not None or result is None:
            self._set_status_message(f"Error: {error_msg}", is_error=True)
            return

        if (
            not isinstance(result, np.ndarray)
            or result.ndim != 3
            or result.shape[2] != 3
            or result.dtype != np.uint8
        ):
            self._set_status_message("Error: invalid processed image output", is_error=True)
            return

        self.current_image = result
        if mode_snapshot == "Filter Chaining" and self._processing_mode == "Filter Chaining":
            self._chaining_started = True
            self._chaining_applied_filters.add(filter_name)
            self._update_filter_button_styles()

        self._display_sources["after"] = self._bgr_to_display_pil(self.current_image)
        self._image_versions["after"] += 1
        self._render_image_preview("after")
        self._hist_after.update_histogram(self.current_image)
        self._set_status_message("Ready", is_error=False)

    def _build_status_bar(self) -> None:
        """Membangun status bar bawah berisi informasi citra dummy."""
        status_wrapper = ctk.CTkFrame(
            self,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=0,
            border_width=0,
        )
        status_wrapper.grid(row=2, column=0, sticky="ew")
        status_wrapper.grid_columnconfigure(0, weight=1)

        separator = ctk.CTkFrame(
            status_wrapper,
            fg_color=theme.COLORS["border"],
            height=theme.BORDER_WIDTH,
            corner_radius=0,
        )
        separator.grid(row=0, column=0, sticky="ew")

        status_inner = ctk.CTkFrame(
            status_wrapper,
            fg_color="transparent",
            corner_radius=0,
            height=32,
        )
        status_inner.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=theme.SPACING["sm"],
            pady=6,
        )
        status_inner.grid_columnconfigure(0, weight=1)

        self._info_label = ctk.CTkLabel(
            status_inner,
            text="sample_image.jpg   1280 x 720   RGB",
            font=self._font_body,
            text_color=theme.COLORS["text_secondary"],
            anchor="w",
        )
        self._info_label.grid(row=0, column=0, sticky="w")

        self._state_label = ctk.CTkLabel(
            status_inner,
            text="Ready",
            font=self._font_label,
            text_color=theme.COLORS["text_secondary"],
            anchor="e",
        )
        self._state_label.grid(row=0, column=1, sticky="e")
