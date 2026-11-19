"""Modul fungsi pengolahan citra murni (input/output array BGR NumPy)."""

import cv2
import numpy as np


def _validate_bgr_image(img_bgr: np.ndarray) -> np.ndarray:
    """Memastikan input adalah array NumPy BGR 3-channel bertipe uint8 yang valid."""
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("Citra input harus berupa numpy.ndarray.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.size == 0:
        raise ValueError("Citra input harus berformat BGR 3-channel dan tidak kosong.")
    if img_bgr.dtype != np.uint8:
        raise ValueError("Tipe data citra input harus uint8.")
    return img_bgr


def _validate_numeric_range(
    value: float | int,
    min_val: float,
    max_val: float,
    name: str,
) -> float:
    """Memvalidasi parameter numerik berada dalam rentang inklusif [min_val, max_val]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"Parameter '{name}' harus berupa angka.")
    numeric = float(value)
    if np.isnan(numeric) or np.isinf(numeric):
        raise ValueError(f"Parameter '{name}' tidak boleh NaN atau tak hingga.")
    if numeric < min_val or numeric > max_val:
        raise ValueError(
            f"Parameter '{name}' harus berada di antara {min_val} dan {max_val}."
        )
    return numeric


def _validate_odd_kernel(
    kernel_size: int,
    min_val: int = 3,
    max_val: int = 31,
    name: str = "kernel_size",
) -> int:
    """Memvalidasi ukuran kernel berupa bilangan bulat ganjil pada rentang yang diizinkan."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise TypeError(f"Parameter '{name}' harus berupa bilangan bulat.")
    k = int(kernel_size)
    if k < min_val or k > max_val or k % 2 == 0:
        raise ValueError(
            f"Parameter '{name}' harus berupa bilangan bulat ganjil antara {min_val} dan {max_val}."
        )
    return k


def grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Mengonversi citra BGR ke skala keabuan berdasarkan bobot luminans lalu dikembalikan ke BGR."""
    _validate_bgr_image(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def negative(img_bgr: np.ndarray) -> np.ndarray:
    """Menghasilkan citra negatif dengan membalik intensitas piksel (255 - piksel)."""
    _validate_bgr_image(img_bgr)
    return (255 - img_bgr).astype(np.uint8)


def brightness(img_bgr: np.ndarray, value: int = 0) -> np.ndarray:
    """Mengubah kecerahan citra dengan menambahkan konstanta dan membatasi nilai 0-255.

    Parameter:
        value: Konstanta penambahan kecerahan (-100 sampai 100).
    """
    _validate_bgr_image(img_bgr)
    offset = int(round(_validate_numeric_range(value, -100, 100, "value")))
    adjusted = img_bgr.astype(np.int16) + offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def thresholding(img_bgr: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Melakukan binerisasi citra: piksel >= ambang menjadi 255, selainnya 0.

    Parameter:
        threshold: Nilai ambang batas (0 sampai 255, default 127).
    """
    _validate_bgr_image(img_bgr)
    thresh = int(round(_validate_numeric_range(threshold, 0, 255, "threshold")))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = np.where(gray >= thresh, 255, 0).astype(np.uint8)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def saturation_hue(
    img_bgr: np.ndarray,
    hue: int = 0,
    saturation: float = 100.0,
) -> np.ndarray:
    """Mengubah komponen Hue dan Saturation citra melalui ruang warna HSV.

    Parameter:
        hue: Pergeseran Hue (-90 sampai 90).
        saturation: Persentase saturasi (0 sampai 200 persen).
    """
    _validate_bgr_image(img_bgr)
    hue_shift = int(round(_validate_numeric_range(hue, -90, 90, "hue")))
    sat_percent = _validate_numeric_range(saturation, 0.0, 200.0, "saturation")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Geser nilai Hue secara melingkar pada rentang OpenCV [0, 179]
    h_new = ((h.astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    s_scale = sat_percent / 100.0
    s_new = np.clip(s.astype(np.float32) * s_scale, 0, 255).astype(np.uint8)

    merged = cv2.merge((h_new, s_new, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def equalization(img_bgr: np.ndarray) -> np.ndarray:
    """Melakukan equalisasi histogram pada channel luminans (YCrCb) agar warna tetap terjaga."""
    _validate_bgr_image(img_bgr)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def blur(
    img_bgr: np.ndarray,
    method: str = "Gaussian",
    kernel_size: int = 5,
) -> np.ndarray:
    """Menghaluskan citra menggunakan metode Gaussian atau Average Blur.

    Parameter:
        method: Metode blur ('Gaussian' atau 'Average').
        kernel_size: Ukuran kernel ganjil (3 sampai 31).
    """
    _validate_bgr_image(img_bgr)
    if not isinstance(method, str):
        raise TypeError("Parameter 'method' harus berupa string.")
    k = _validate_odd_kernel(kernel_size, 3, 31, "kernel_size")

    norm_method = method.strip().lower()
    if norm_method == "gaussian":
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    if norm_method == "average":
        return cv2.blur(img_bgr, (k, k))
    raise ValueError("Metode blur harus 'Gaussian' atau 'Average'.")


def sharpening(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Menajamkan detail citra menggunakan teknik unsharp mask.

    Parameter:
        strength: Kekuatan penajaman (0.5 sampai 3.0).
    """
    _validate_bgr_image(img_bgr)
    alpha = _validate_numeric_range(strength, 0.5, 3.0, "strength")
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 1.0)
    # Tambah komponen tepi (asli - blur) sebesar alpha secara stabil
    return cv2.addWeighted(img_bgr, 1.0 + alpha, blurred, -alpha, 0)


def edge_detection(
    img_bgr: np.ndarray,
    method: str = "Canny",
    threshold: int = 100,
    threshold2: int | None = None,
) -> np.ndarray:
    """Mendeteksi tepi citra dengan metode Sobel atau Canny.

    Parameter:
        method: Metode deteksi tepi ('Sobel' atau 'Canny').
        threshold: Nilai ambang Canny (0 sampai 255).
        threshold2: Ambang atas opsional untuk Canny (0 sampai 255).
    """
    _validate_bgr_image(img_bgr)
    if not isinstance(method, str):
        raise TypeError("Parameter 'method' harus berupa string.")
    norm_method = method.strip().lower()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if norm_method == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edges = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if norm_method == "canny":
        t1 = int(round(_validate_numeric_range(threshold, 0, 255, "threshold")))
        if threshold2 is None:
            low = t1 // 2
            high = t1
        else:
            t2 = int(round(_validate_numeric_range(threshold2, 0, 255, "threshold2")))
            low, high = min(t1, t2), max(t1, t2)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    raise ValueError("Metode deteksi tepi harus 'Sobel' atau 'Canny'.")


def mirroring(img_bgr: np.ndarray, direction: str = "Horizontal") -> np.ndarray:
    """Membalik orientasi citra secara horizontal atau vertikal.

    Parameter:
        direction: Arah pencerminan ('Horizontal' atau 'Vertical').
    """
    _validate_bgr_image(img_bgr)
    if not isinstance(direction, str):
        raise TypeError("Parameter 'direction' harus berupa string.")
    norm_dir = direction.strip().lower()
    if norm_dir == "horizontal":
        return cv2.flip(img_bgr, 1)
    if norm_dir in ("vertical", "vertikal"):
        return cv2.flip(img_bgr, 0)
    raise ValueError("Arah mirroring harus 'Horizontal' atau 'Vertical'.")


def rotate(img_bgr: np.ndarray, angle: float = 0) -> np.ndarray:
    """Memutar citra terhadap titik tengah tanpa memotong tepi citra.

    Parameter:
        angle: Sudut rotasi dalam derajat (-180 sampai 180).
    """
    _validate_bgr_image(img_bgr)
    deg = _validate_numeric_range(angle, -180.0, 180.0, "angle")
    if deg == 0.0:
        return img_bgr.copy()
    if deg == 90.0:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == -90.0:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if abs(deg) == 180.0:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)

    height, width = img_bgr.shape[:2]
    cx, cy = width / 2.0, height / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)

    # Hitung ukuran kanvas baru agar seluruh piksel hasil rotasi tetap termuat utuh
    cos_val = abs(float(matrix[0, 0]))
    sin_val = abs(float(matrix[0, 1]))
    new_width = max(int(np.ceil(height * sin_val + width * cos_val - 1e-9)), 1)
    new_height = max(int(np.ceil(height * cos_val + width * sin_val - 1e-9)), 1)

    matrix[0, 2] += (new_width / 2.0) - cx
    matrix[1, 2] += (new_height / 2.0) - cy

    return cv2.warpAffine(
        img_bgr,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def sketch(img_bgr: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """Mengubah citra menjadi efek sketsa pensil (grayscale, invert, Gaussian blur, color dodge).

    Parameter:
        kernel_size: Ukuran kernel Gaussian blur ganjil (3 sampai 31).
    """
    _validate_bgr_image(img_bgr)
    k = _validate_odd_kernel(kernel_size, 3, 31, "kernel_size")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (k, k), 0)
    # Color dodge: bagi intensitas abu-abu dengan (255 - blurred) agar area datar menjadi putih
    divisor = np.maximum(255 - blurred, 1)
    dodged = np.clip((gray.astype(np.float32) * 255.0) / divisor.astype(np.float32), 0, 255).astype(np.uint8)
    return cv2.cvtColor(dodged, cv2.COLOR_GRAY2BGR)


def vignette(
    img_bgr: np.ndarray,
    strength: float = 0.6,
    radius: float = 0.8,
) -> np.ndarray:
    """Memberikan efek penggelapan tepi menggunakan mask radial Gaussian.

    Parameter:
        strength: Kekuatan efek vignette (0.0 sampai 1.0).
        radius: Radius terang relatif terhadap dimensi citra (0.2 sampai 1.5).
    """
    _validate_bgr_image(img_bgr)
    s = _validate_numeric_range(strength, 0.0, 1.0, "strength")
    r = _validate_numeric_range(radius, 0.2, 1.5, "radius")

    height, width = img_bgr.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    max_dist = max(float(np.hypot(cx, cy)), 1.0)

    y_coords, x_coords = np.ogrid[:height, :width]
    dist_sq = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (max_dist ** 2)
    gaussian_mask = np.exp(-dist_sq / (2.0 * (r ** 2))).astype(np.float32)
    mask = (1.0 - s) + s * gaussian_mask

    shaded = img_bgr.astype(np.float32) * mask[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def add_gaussian_noise(
    img_bgr: np.ndarray,
    sigma: float = 25.0,
    seed: int | None = None,
) -> np.ndarray:
    """Menambahkan noise berdistribusi normal (mean 0) ke citra.

    Parameter:
        sigma: Standar deviasi noise (5 sampai 80).
        seed: Seed opsional untuk hasil acak yang deterministik.
    """
    _validate_bgr_image(img_bgr)
    std = _validate_numeric_range(sigma, 5.0, 80.0, "sigma")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, img_bgr.shape).astype(np.float32)
    noisy = img_bgr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    img_bgr: np.ndarray,
    amount: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """Mengganti sebagian piksel secara acak menjadi 0 (pepper) atau 255 (salt).

    Parameter:
        amount: Persentase piksel yang terkena noise (1 sampai 20 persen).
        seed: Seed opsional untuk hasil acak yang deterministik.
    """
    _validate_bgr_image(img_bgr)
    percent = _validate_numeric_range(amount, 1.0, 20.0, "amount")

    rng = np.random.default_rng(seed)
    prob = percent / 100.0
    height, width = img_bgr.shape[:2]
    rand_map = rng.random((height, width))

    output = img_bgr.copy()
    salt_mask = rand_map < (prob / 2.0)
    pepper_mask = (rand_map >= (prob / 2.0)) & (rand_map < prob)
    output[salt_mask] = 255
    output[pepper_mask] = 0
    return output


def denoising(
    img_bgr: np.ndarray,
    method: str = "Median",
    kernel_size: int = 5,
    sigma: float = 75.0,
) -> np.ndarray:
    """Mengurangi noise citra dengan pilihan metode Median, Gaussian, atau Bilateral.

    Parameter:
        method: Metode denoising ('Median', 'Gaussian', atau 'Bilateral').
        kernel_size: Ukuran kernel ganjil (3 sampai 31).
        sigma: Nilai sigma untuk filter Bilateral (5 sampai 150).
    """
    _validate_bgr_image(img_bgr)
    if not isinstance(method, str):
        raise TypeError("Parameter 'method' harus berupa string.")
    k = _validate_odd_kernel(kernel_size, 3, 31, "kernel_size")
    norm_method = method.strip().lower()

    if norm_method == "median":
        return cv2.medianBlur(img_bgr, k)
    if norm_method == "gaussian":
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    if norm_method == "bilateral":
        sigma_val = _validate_numeric_range(sigma, 5.0, 150.0, "sigma")
        return cv2.bilateralFilter(img_bgr, d=k, sigmaColor=sigma_val, sigmaSpace=sigma_val)

    raise ValueError("Metode denoising harus 'Median', 'Gaussian', atau 'Bilateral'.")
