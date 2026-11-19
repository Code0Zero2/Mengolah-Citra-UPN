"""Modul akses webcam, pemrosesan frame langsung, dan pengambilan gambar (capture)."""

import threading
import time
import tkinter as tk
from typing import Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import theme


def normalize_camera_frame(frame: np.ndarray) -> np.ndarray | None:
    """Memastikan frame kamera berbentuk BGR 3-channel bertipe uint8 yang valid."""
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        return None

    arr = frame
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 3:
        return arr.copy()
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return None


class CameraSession:
    """Pengelola siklus hidup VideoCapture dan pembacaan frame di thread latar belakang."""

    def __init__(self, camera_index: int = 0, poll_interval_sec: float = 0.03) -> None:
        self.camera_index = camera_index
        self.poll_interval_sec = poll_interval_sec

        self._cap = None
        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

        self._latest_frame: np.ndarray | None = None
        self._error_message: str | None = None

    def open(self) -> None:
        """Membuka perangkat kamera dan memulai thread akuisisi frame."""
        with self._cap_lock:
            if self._running:
                return
            cap = None
            try:
                cap = cv2.VideoCapture(self.camera_index)
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Camera not found or cannot be opened.")
                self._cap = cap
            except Exception as exc:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._cap = None
                raise RuntimeError("Camera not found or cannot be opened.") from exc

            with self._frame_lock:
                self._latest_frame = None
                self._error_message = None

            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _reader_loop(self) -> None:
        """Membaca frame kamera secara berkala tanpa menyentuh widget Tkinter."""
        consecutive_failures = 0
        while self._running:
            with self._cap_lock:
                if not self._running or self._cap is None:
                    break
                try:
                    ok, raw_frame = self._cap.read()
                except Exception:
                    ok, raw_frame = False, None

            if not self._running:
                break

            if ok and raw_frame is not None:
                normalized = normalize_camera_frame(raw_frame)
                if normalized is not None:
                    consecutive_failures = 0
                    with self._frame_lock:
                        self._latest_frame = normalized
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1

            if consecutive_failures >= 10:
                with self._frame_lock:
                    self._error_message = "Camera disconnected or failed to capture frame."
                self._running = False
                self._stop_event.set()
                self._release_capture_locked()
                break

            if self._stop_event.wait(self.poll_interval_sec):
                break

    def _release_capture_locked(self) -> None:
        """Melepas objek VideoCapture secara aman di bawah kunci I/O."""
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def get_state(self) -> tuple[np.ndarray | None, str | None]:
        """Mengambil salinan frame terbaru dan pesan error (jika ada)."""
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            err = self._error_message
        return frame, err

    def capture_frame(self) -> np.ndarray | None:
        """Mengambil satu frame BGR uint8 terbaru yang valid untuk diproses."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def close(self) -> None:
        """Menghentikan pembacaan frame dan melepas resource kamera."""
        self._running = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=1.5)
        self._thread = None
        self._release_capture_locked()


class LiveFrameProcessor:
    """Worker thread tunggal untuk memproses frame kamera terbaru pada mode Live Processing."""

    def __init__(self, session: CameraSession) -> None:
        self._session = session
        self._pipeline_lock = threading.Lock()
        self._pipeline: tuple[str, list[tuple[str, Callable, dict]], str | None] = (
            "Single Filter",
            [],
            None,
        )
        self._pipeline_gen: int = 0

        self._result_lock = threading.Lock()
        self._raw_frame: np.ndarray | None = None
        self._processed_frame: np.ndarray | None = None
        self._error_msg: str | None = None
        self._result_seq: int = 0
        self._result_gen: int = 0
        self._result_mode: str = "Single Filter"
        self._result_steps_count: int = 0
        self._last_latency_ms: float = 0.0

        self._trigger = threading.Event()
        self._stop_event = threading.Event()
        self._busy = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)

    @property
    def is_busy(self) -> bool:
        """Menunjukkan apakah worker sedang memproses satu frame."""
        return self._busy

    @property
    def current_pipeline_gen(self) -> int:
        """Mengembalikan nomor generasi pipeline filter terbaru."""
        with self._pipeline_lock:
            return self._pipeline_gen

    def start(self) -> None:
        """Memulai thread pekerja pemrosesan live."""
        if not self._thread.is_alive():
            self._thread.start()

    def set_pipeline(
        self,
        pipeline: tuple[str, list[tuple[str, Callable, dict]], str | None],
    ) -> None:
        """Menyimpan snapshot pipeline terbaru lalu memicu pemrosesan bila sedang idle."""
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._pipeline_gen += 1
        self._trigger.set()

    def trigger_if_idle(self) -> None:
        """Meminta pemrosesan frame terbaru jika worker tidak sedang sibuk."""
        if not self._busy and not self._stop_event.is_set():
            self._trigger.set()

    def get_result(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, int, int, float]:
        """Mengambil frame asli, hasil, error, nomor urut, generasi pipeline, dan latensi."""
        with self._result_lock:
            raw = self._raw_frame.copy() if self._raw_frame is not None else None
            proc = self._processed_frame.copy() if self._processed_frame is not None else None
            return (
                raw,
                proc,
                self._error_msg,
                self._result_seq,
                self._result_gen,
                self._last_latency_ms,
            )

    def get_result_meta(self) -> tuple[str, int]:
        """Mengembalikan mode pemrosesan dan jumlah langkah filter dari hasil terbaru."""
        with self._result_lock:
            return self._result_mode, self._result_steps_count

    def _worker_loop(self) -> None:
        """Memproses frame kamera mentah terbaru secara berurutan tanpa menyentuh Tkinter."""
        while not self._stop_event.is_set():
            triggered = self._trigger.wait(timeout=0.25)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue
            self._trigger.clear()

            raw_frame = self._session.capture_frame()
            if raw_frame is None:
                continue

            with self._pipeline_lock:
                mode, steps, pipeline_err = self._pipeline
                gen = self._pipeline_gen

            if pipeline_err is not None:
                with self._result_lock:
                    self._raw_frame = raw_frame
                    self._processed_frame = raw_frame
                    self._error_msg = pipeline_err
                    self._result_seq += 1
                    self._result_gen = gen
                    self._result_mode = mode
                    self._result_steps_count = len(steps)
                    self._last_latency_ms = 0.0
                continue

            if not steps:
                with self._result_lock:
                    self._raw_frame = raw_frame
                    self._processed_frame = raw_frame
                    self._error_msg = None
                    self._result_seq += 1
                    self._result_gen = gen
                    self._result_mode = mode
                    self._result_steps_count = 0
                    self._last_latency_ms = 0.0
                continue

            self._busy = True
            t0 = time.perf_counter()
            working = raw_frame.copy()
            err_msg: str | None = None
            try:
                for _name, func, params in steps:
                    out = func(working, **params)
                    normalized = normalize_camera_frame(out)
                    if normalized is None:
                        raise ValueError("Invalid filter output frame")
                    working = normalized
            except Exception as exc:
                err_msg = str(exc) or "Filter processing failed"
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._busy = False

            with self._result_lock:
                self._raw_frame = raw_frame
                if err_msg is not None:
                    self._processed_frame = raw_frame
                    self._error_msg = err_msg
                else:
                    self._processed_frame = working
                    self._error_msg = None
                self._result_seq += 1
                self._result_gen = gen
                self._result_mode = mode
                self._result_steps_count = len(steps)
                self._last_latency_ms = elapsed_ms

    def close(self) -> None:
        """Menghentikan worker thread pemrosesan live dengan aman."""
        self._stop_event.set()
        self._trigger.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.5)


class CameraPreviewWindow(ctk.CTkToplevel):
    """Jendela pratinjau webcam khusus untuk pengambilan gambar (Capture)."""

    def __init__(
        self,
        master,
        session: CameraSession,
        on_capture: Callable[[np.ndarray], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self._session = session
        self._on_capture_cb = on_capture
        self._on_cancel_cb = on_cancel

        self._font_family = theme.resolve_font_family()
        self._photo_ref: ImageTk.PhotoImage | None = None
        self._latest_pil: Image.Image | None = None
        self._poll_job: str | None = None
        self._closed = False

        self.title("Camera Capture")
        self.geometry("720x540")
        self.minsize(560, 440)
        self.configure(fg_color=theme.COLORS["bg_window"])
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._handle_cancel)

        self._build_ui()
        self._poll_preview()

    def _build_ui(self) -> None:
        """Menyusun area pratinjau kamera serta tombol aksi Capture dan Cancel."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        preview_card = ctk.CTkFrame(
            self,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
        )
        preview_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["sm"], theme.SPACING["xs"]),
        )
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(0, weight=1)

        self._preview_box = ctk.CTkFrame(
            preview_card,
            fg_color=theme.COLORS["bg_image_area"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=0,
        )
        self._preview_box.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["xs"],
            pady=theme.SPACING["xs"],
        )
        self._preview_box.grid_propagate(False)
        self._preview_box.grid_columnconfigure(0, weight=1)
        self._preview_box.grid_rowconfigure(0, weight=1)

        self._preview_label = tk.Label(
            self._preview_box,
            text="Starting camera...",
            fg=theme.COLORS["text_secondary"],
            bg=theme.COLORS["bg_image_area"],
            font=(self._font_family, theme.FONT_SIZES["body"]),
            borderwidth=0,
            highlightthickness=0,
        )
        self._preview_label.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["xs"],
            pady=theme.SPACING["xs"],
        )
        self._preview_box.bind("<Configure>", lambda _e: self._render_current_pil())

        footer = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        footer.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["xs"], theme.SPACING["sm"]),
        )
        footer.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            footer,
            text="Camera active",
            font=ctk.CTkFont(
                family=self._font_family,
                size=theme.FONT_SIZES["label"],
                weight=theme.FONT_WEIGHTS["regular"],
            ),
            text_color=theme.COLORS["text_secondary"],
            anchor="w",
        )
        self._status_label.grid(row=0, column=0, sticky="w")

        self._cancel_btn = ctk.CTkButton(
            footer,
            text="Cancel",
            font=ctk.CTkFont(
                family=self._font_family,
                size=theme.FONT_SIZES["body"],
                weight=theme.FONT_WEIGHTS["medium"],
            ),
            fg_color=theme.COLORS["bg_panel"],
            hover_color=theme.COLORS["bg_window"],
            text_color=theme.COLORS["text_primary"],
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
            corner_radius=theme.CORNER_RADIUS,
            height=34,
            width=96,
            cursor="hand2",
            command=self._handle_cancel,
        )
        self._cancel_btn.grid(row=0, column=1, padx=(theme.SPACING["xs"], theme.SPACING["xs"]))

        self._capture_btn = ctk.CTkButton(
            footer,
            text="Capture",
            font=ctk.CTkFont(
                family=self._font_family,
                size=theme.FONT_SIZES["body"],
                weight=theme.FONT_WEIGHTS["medium"],
            ),
            fg_color=theme.COLORS["accent"],
            hover_color=theme.COLORS["accent_hover"],
            text_color=theme.COLORS["accent_text"],
            border_width=0,
            corner_radius=theme.CORNER_RADIUS,
            height=34,
            width=104,
            cursor="hand2",
            state="disabled",
            command=self._handle_capture,
        )
        self._capture_btn.grid(row=0, column=2)

    def _poll_preview(self) -> None:
        """Mengambil frame terbaru dari session dan memperbarui pratinjau di thread GUI."""
        if self._closed:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        frame_bgr, error_msg = self._session.get_state()
        if error_msg is not None:
            self._status_label.configure(text=error_msg, text_color=theme.COLORS["error_text"])
            self._capture_btn.configure(state="disabled")
            if self._latest_pil is None:
                self._preview_label.configure(image="", text=error_msg)
            return

        if frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._latest_pil = Image.fromarray(rgb, mode="RGB")
            self._render_current_pil()
            if self._capture_btn.cget("state") == "disabled":
                self._capture_btn.configure(state="normal")
            self._status_label.configure(
                text="Camera active",
                text_color=theme.COLORS["text_secondary"],
            )

        if not self._closed:
            self._poll_job = self.after(33, self._poll_preview)

    def _render_current_pil(self) -> None:
        """Menampilkan frame pratinjau secara proporsional sesuai ukuran area kamera."""
        if self._closed or self._latest_pil is None:
            return
        try:
            if not self.winfo_exists() or not self._preview_box.winfo_exists():
                return
        except tk.TclError:
            return

        box_w = max(self._preview_box.winfo_width() - theme.SPACING["sm"], 1)
        box_h = max(self._preview_box.winfo_height() - theme.SPACING["sm"], 1)
        if box_w <= 10 or box_h <= 10:
            return

        src_w, src_h = self._latest_pil.size
        scale = min(box_w / src_w, box_h / src_h)
        target_w = max(int(round(src_w * scale)), 1)
        target_h = max(int(round(src_h * scale)), 1)

        resized = self._latest_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        self._photo_ref = photo
        self._preview_label.configure(image=photo, text="")

    def _handle_capture(self) -> None:
        """Menangkap frame terakhir yang valid, menutup kamera, dan mengirim hasil ke app."""
        if self._closed:
            return

        captured = self._session.capture_frame()
        if captured is None:
            self._status_label.configure(
                text="Error: no valid camera frame available",
                text_color=theme.COLORS["error_text"],
            )
            return

        self.close_window()
        self._on_capture_cb(captured)

    def _handle_cancel(self) -> None:
        """Membatalkan pratinjau kamera dan melepas resource tanpa mengubah citra utama."""
        if self._closed:
            return
        self.close_window()
        self._on_cancel_cb()

    def close_window(self) -> None:
        """Menghentikan polling, melepas resource kamera, dan menutup jendela pratinjau."""
        if self._closed:
            return
        self._closed = True
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None

        self._session.close()
        self._photo_ref = None
        try:
            if self.winfo_exists():
                self.destroy()
        except tk.TclError:
            pass
