# Mengolah Citra UPN

Aplikasi desktop pengolahan citra digital berbasis Python untuk membuka gambar dari berkas atau webcam, menerapkan filter secara tunggal maupun berantai, membandingkan tampilan *Before/After* beserta histogramnya, serta memproses umpan kamera secara *real-time*.

---

## Teknologi yang Digunakan

- Python 3.10+
- CustomTkinter
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib (`TkAgg`)
- Pillow (`ImageTk`)

---

## Ringkasan Fitur Utama

- **Open File & Save File:** Membuka citra `.jpg`, `.jpeg`, `.png`, `.bmp` serta menyimpan hasil pada resolusi penuh ke `.png` atau `.jpg`/`.jpeg`.
- **Open Camera & Live Processing:** Mengambil satu frame melalui jendela `Camera Capture` atau menjalankan pemrosesan kamera langsung pada workspace utama lengkap dengan tombol `Capture` hasil filter dan `Stop Live`.
- **Before / After & Dynamic Histogram:** Menampilkan perbandingan citra asli (`Before`) dan hasil (`After`) secara proporsional beserta `Original Histogram` dan `Processed Histogram` (RGB maupun Grayscale).
- **Mode Pengolahan:** Mendukung `Single Filter` (dari citra asli) dan `Filter Chaining` (berurutan, termasuk saat `Live Processing` melalui `Add Filter` dan `Clear Chain`), serta `Reset` untuk kembali ke citra asli.

---

## Ringkasan 16 Filter

- **Basic:** `Grayscale`, `Negative`, `Thresholding`
- **Color:** `Brightness`, `Saturation/Hue`, `Equalization`
- **Spatial:** `Blur` (`Gaussian` / `Average`), `Sharpening`, `Edge Detection` (`Canny` / `Sobel`), `Rotate` (slider `-180°` s.d. `180°` dan preset `±90°`, `±180°`)
- **Effects:** `Mirroring` (`Horizontal` / `Vertical`), `Sketch`, `Vignette`
- **Noise:** `Add Gaussian Noise`, `Add Salt & Pepper`, `Denoising` (`Median` / `Gaussian` / `Bilateral`)

---

## Instalasi dan Menjalankan Aplikasi (Windows PowerShell)

```powershell
cd mengolah_citra_upn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Untuk menjalankan unit test filter:

```powershell
python -m unittest discover -p "test_*.py" -v
```

---

## Dokumentasi Lengkap

Panduan lengkap fitur, rincian parameter setiap filter, alur `Live Processing`, dan struktur aplikasi tersedia pada [dokumentasi aplikasi](mengolah_citra_upn/README.md).
