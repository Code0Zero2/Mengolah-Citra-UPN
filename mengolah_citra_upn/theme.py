"""Konstanta warna, tipografi, dan spasi untuk aplikasi Mengolah Citra UPN."""

from tkinter import font as tkfont

# Identitas dan ukuran jendela
APP_TITLE = "Mengolah Citra UPN"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 780
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 680

# Palet warna mode terang (sesuai bagian 6.2 PRD)
COLORS = {
    "bg_window": "#F7F7F5",
    "bg_panel": "#FFFFFF",
    "bg_image_area": "#EFEFEC",
    "bg_toolbar_hover": "#ECECE8",
    "border": "#E3E3E0",
    "text_primary": "#1C1C1A",
    "text_secondary": "#6B6B66",
    "accent": "#2F5D50",
    "accent_hover": "#244A40",
    "accent_text": "#FFFFFF",
    "disabled_bg": "#D9D9D5",
    "disabled_text": "#8C8C86",
    "error_text": "#A64444",
}

# Warna kurva histogram (sesuai bagian 6.2 PRD)
HISTOGRAM_COLORS = {
    "r": "#C0504D",
    "g": "#5B9A6B",
    "b": "#4F7CAC",
    "gray": "#3A3A38",
    "alpha": 0.60,
    "grid": "#E3E3E0",
}

# Spasi konsisten kelipatan 8 px dan geometri elemen
SPACING = {
    "xs": 8,
    "sm": 16,
    "md": 24,
    "lg": 32,
}

CORNER_RADIUS = 6
BORDER_WIDTH = 1

# Ukuran dan bobot tipografi (maksimal dua bobot: regular dan medium/normal-bold Tkinter)
FONT_SIZES = {
    "section": 12,
    "label": 11,
    "body": 12,
    "status": 12,
}

FONT_WEIGHTS = {
    "regular": "normal",
    "medium": "bold",
}

_FALLBACK_FONTS = ("Inter", "Segoe UI", "Arial")


def resolve_font_family() -> str:
    """Memilih font Inter jika tersedia, jika tidak gunakan Segoe UI atau font sistem."""
    try:
        available = set(tkfont.families())
        for family in _FALLBACK_FONTS:
            if family in available:
                return family
    except RuntimeError:
        pass
    return "Segoe UI"
