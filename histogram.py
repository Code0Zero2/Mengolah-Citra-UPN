"""Modul pembuatan dan rendering histogram citra dengan Matplotlib (TkAgg)."""

import matplotlib
matplotlib.use("TkAgg")

import cv2
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

import theme


def compute_histogram_data(img_bgr: np.ndarray) -> dict[str, object]:
    """Menghitung data histogram ter-normalisasi untuk citra BGR berwarna atau grayscale."""
    if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Citra histogram tidak valid.")

    if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        plane = img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]
        hist_gray = cv2.calcHist([plane], [0], None, [256], [0, 256]).ravel()
        peak = float(hist_gray.max()) if hist_gray.size else 1.0
        norm_gray = hist_gray / max(peak, 1.0)
        return {"mode": "gray", "gray": norm_gray, "raw_gray": hist_gray}

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Format channel citra histogram harus 1 atau 3.")

    is_gray = (
        np.array_equal(img_bgr[:, :, 0], img_bgr[:, :, 1])
        and np.array_equal(img_bgr[:, :, 1], img_bgr[:, :, 2])
    )
    if is_gray:
        hist_gray = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).ravel()
        peak = float(hist_gray.max()) if hist_gray.size else 1.0
        norm_gray = hist_gray / max(peak, 1.0)
        return {"mode": "gray", "gray": norm_gray, "raw_gray": hist_gray}

    # Urutan channel OpenCV adalah BGR: 0=Biru, 1=Hijau, 2=Merah
    hist_b = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).ravel()
    hist_g = cv2.calcHist([img_bgr], [1], None, [256], [0, 256]).ravel()
    hist_r = cv2.calcHist([img_bgr], [2], None, [256], [0, 256]).ravel()
    peak = max(float(hist_r.max()), float(hist_g.max()), float(hist_b.max()), 1.0)

    return {
        "mode": "rgb",
        "r": hist_r / peak,
        "g": hist_g / peak,
        "b": hist_b / peak,
        "raw_r": hist_r,
        "raw_g": hist_g,
        "raw_b": hist_b,
    }


class HistogramPanel(ctk.CTkFrame):
    """Panel widget CustomTkinter yang menanamkan canvas histogram Matplotlib."""

    def __init__(self, master, title: str = "Histogram", **kwargs) -> None:
        super().__init__(
            master,
            fg_color=theme.COLORS["bg_panel"],
            corner_radius=theme.CORNER_RADIUS,
            border_width=theme.BORDER_WIDTH,
            border_color=theme.COLORS["border"],
            **kwargs,
        )
        self._font_family = theme.resolve_font_family()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(
                family=self._font_family,
                size=theme.FONT_SIZES["label"],
                weight=theme.FONT_WEIGHTS["regular"],
            ),
            text_color=theme.COLORS["text_secondary"],
            anchor="w",
        )
        self._title_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=theme.SPACING["sm"],
            pady=(theme.SPACING["xs"], 0),
        )

        self._figure = Figure(figsize=(4.2, 1.10), dpi=100)
        self._figure.patch.set_facecolor(theme.COLORS["bg_panel"])
        self._ax = self._figure.add_subplot(111)
        self._configure_axes()
        self._last_hist_data: dict[str, object] | None = None

        self._mpl_canvas = FigureCanvasTkAgg(self._figure, master=self)
        canvas_widget = self._mpl_canvas.get_tk_widget()
        canvas_widget.configure(
            background=theme.COLORS["bg_panel"],
            highlightthickness=0,
            borderwidth=0,
        )
        canvas_widget.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.SPACING["xs"],
            pady=(0, theme.SPACING["xs"]),
        )

    def _configure_axes(self) -> None:
        """Mengatur tampilan sumbu histogram agar minimalis tanpa kotak tebal."""
        ax = self._ax
        ax.set_facecolor("none")
        ax.set_xlim(0, 255)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_yticks([])
        ax.tick_params(
            axis="x",
            colors=theme.COLORS["text_secondary"],
            labelsize=8,
            length=2,
            pad=2,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(theme.COLORS["border"])
        ax.spines["bottom"].set_linewidth(0.8)
        ax.grid(
            axis="x",
            color=theme.HISTOGRAM_COLORS["grid"],
            linestyle="-",
            linewidth=0.5,
            alpha=0.7,
        )
        self._figure.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.22)

    def plot_dummy(self, variant: str = "before") -> None:
        """Menampilkan kurva histogram RGB dummy untuk pratinjau tata letak statis."""
        ax = self._ax
        ax.clear()
        self._configure_axes()

        x = np.arange(256, dtype=np.float32)
        if variant == "before":
            r_curve = np.exp(-((x - 155) ** 2) / (2 * 32 ** 2)) * 0.85 + 0.12 * np.exp(-((x - 75) ** 2) / (2 * 22 ** 2))
            g_curve = np.exp(-((x - 128) ** 2) / (2 * 36 ** 2)) * 0.90
            b_curve = np.exp(-((x - 95) ** 2) / (2 * 30 ** 2)) * 0.78 + 0.15 * np.exp(-((x - 190) ** 2) / (2 * 25 ** 2))
        else:
            r_curve = np.exp(-((x - 170) ** 2) / (2 * 30 ** 2)) * 0.92
            g_curve = np.exp(-((x - 142) ** 2) / (2 * 34 ** 2)) * 0.86
            b_curve = np.exp(-((x - 110) ** 2) / (2 * 32 ** 2)) * 0.80

        alpha = theme.HISTOGRAM_COLORS["alpha"]
        for curve, color in (
            (r_curve, theme.HISTOGRAM_COLORS["r"]),
            (g_curve, theme.HISTOGRAM_COLORS["g"]),
            (b_curve, theme.HISTOGRAM_COLORS["b"]),
        ):
            ax.plot(x, curve, color=color, alpha=alpha, linewidth=1.4)
            ax.fill_between(x, curve, color=color, alpha=0.10)

        ax.set_ylim(0, 1.05)
        self._mpl_canvas.draw_idle()

    def update_histogram(self, img_bgr: np.ndarray) -> None:
        """Memperbarui kurva histogram dari array citra BGR aktual tanpa membuat canvas baru."""
        data = compute_histogram_data(img_bgr)
        self._last_hist_data = data

        ax = self._ax
        ax.clear()
        self._configure_axes()

        x = np.arange(256, dtype=np.float32)
        if data["mode"] == "gray":
            curve = data["gray"]
            color = theme.HISTOGRAM_COLORS["gray"]
            ax.plot(x, curve, color=color, alpha=0.85, linewidth=1.4)
            ax.fill_between(x, curve, color=color, alpha=0.12)
        else:
            alpha = theme.HISTOGRAM_COLORS["alpha"]
            for key, color in (
                ("r", theme.HISTOGRAM_COLORS["r"]),
                ("g", theme.HISTOGRAM_COLORS["g"]),
                ("b", theme.HISTOGRAM_COLORS["b"]),
            ):
                curve = data[key]
                ax.plot(x, curve, color=color, alpha=alpha, linewidth=1.35)
                ax.fill_between(x, curve, color=color, alpha=0.08)

        ax.set_ylim(0, 1.05)
        self._mpl_canvas.draw_idle()
