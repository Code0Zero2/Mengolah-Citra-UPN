# Mengolah Citra UPN

Aplikasi desktop pengolahan citra digital berbasis Python yang memungkinkan pengguna membuka gambar dari berkas atau webcam, menerapkan berbagai filter spasial maupun warna secara tunggal atau berantai, membandingkan hasil secara visual dan melalui histogram, serta memproses frame kamera secara langsung (*real-time*).

**Tujuan:** memenuhi proyek tugas Pengolahan Citra dengan antarmuka desktop yang minimalis, responsif, dan siap digunakan untuk demonstrasi maupun evaluasi.

---

## Teknologi yang Digunakan

- **Bahasa:** Python 3.10+ (diuji pada Python 3.14.3)
- **GUI:** CustomTkinter
- **Pengolahan Citra:** OpenCV (`opencv-python`) dan NumPy
- **Histogram:** Matplotlib (backend `TkAgg` tertanam di GUI)
- **Tampilan Pratinjau:** Pillow (`ImageTk`)

---

## Fitur Aplikasi

- **Open File:** Memuat berkas citra berformat `.jpg`, `.jpeg`, `.png`, dan `.bmp` ke workspace dengan deteksi mode warna (`RGB` atau `Grayscale`).
- **Open Camera:** Membuka jendela khusus `Camera Capture` untuk mengambil satu frame dari webcam (indeks `0`) ke workspace utama.
- **Live Processing:** Menjalankan kamera langsung ke workspace utama; panel `Sebelum` menampilkan frame asli dan panel `Sesudah` menampilkan hasil pemrosesan secara *real-time*.
- **Capture (Hasil Live Processing):** Mengambil output frame yang telah diproses pada `Live Processing` untuk dijadikan citra baru pada workspace utama.
- **Panel Before dan After:** Menampilkan perbandingan citra asli (`Sebelum`) dan citra hasil (`Sesudah`) dengan penskalaan proporsional mengikuti ukuran jendela.
- **Histogram Before dan After:** Menampilkan kurva distribusi intensitas RGB (3 kurva) atau Grayscale (1 kurva) secara dinamis untuk citra statis maupun saat `Live Processing`.
- **16 Filter Pengolahan Citra:** Dikelompokkan dalam 5 kategori (`Dasar`, `Warna`, `Spasial`, `Efek`, `Noise`) lengkap dengan kontrol parameter dinamis dan 4 tombol preset untuk `Rotate`.
- **Mode Pengolahan:** Mendukung mode `Single Filter` dan `Filter Chaining`.
- **Save File:** Menyimpan citra hasil (`current_image`) pada resolusi penuh ke format `.png` atau `.jpg`/`.jpeg`.
- **Reset:** Mengembalikan citra hasil ke citra asli (`original_image`) tanpa mengubah mode pemrosesan atau filter yang sedang dipilih.

---

## Daftar Filter

| No | Filter | Kategori | Deskripsi Singkat | Parameter |
| --- | --- | --- | --- | --- |
| 1 | `Grayscale` | Dasar | Konversi BGR ke skala keabuan berdasarkan bobot luminans | Tidak ada parameter |
| 2 | `Negative` | Dasar | Membalik nilai intensitas piksel (`255 - nilai piksel`) | Tidak ada parameter |
| 3 | `Thresholding` | Dasar | Binerisasi citra (`piksel >= ambang` menjadi `255`, lainnya `0`) | `Threshold`: `0`–`255` (default `127`) |
| 4 | `Brightness` | Warna | Mengubah kecerahan dengan penambahan konstanta dan clipping `0–255` | `Brightness`: `-100` s.d. `100` (default `0`) |
| 5 | `Saturation/Hue` | Warna | Mengubah komponen Hue dan Saturation melalui ruang warna HSV | `Hue`: `-90` s.d. `90` (default `0`), `Saturation (%)`: `0`–`200` (default `100`) |
| 6 | `Equalization` | Warna | Equalisasi histogram pada channel luminans (`YCrCb`) | Tidak ada parameter |
| 7 | `Blur` | Spasial | Penghalusan citra menggunakan metode Gaussian atau Average | `Method`: `Gaussian`/`Average`, `Kernel Size`: ganjil `3`–`31` (default `5`) |
| 8 | `Sharpening` | Spasial | Penajaman detail citra dengan teknik *unsharp mask* | `Strength`: `0.5`–`3.0` (step `0.1`, default `1.0`) |
| 9 | `Edge Detection` | Spasial | Deteksi tepi menggunakan operator Sobel atau algoritma Canny | `Method`: `Canny`/`Sobel`, `Threshold`: `0`–`255` (default `100`, aktif pada `Canny`) |
| 10 | `Rotate` | Spasial | Memutar citra terhadap titik tengah tanpa memotong tepi citra | `Angle`: `-180` s.d. `180` (default `0`), serta preset `Rotate 90°`, `Rotate 180°`, `Rotate -90°`, `Rotate -180°` |
| 11 | `Mirroring` | Efek | Pencerminan orientasi citra secara horizontal atau vertikal | `Direction`: `Horizontal`/`Vertical` |
| 12 | `Sketch` | Efek | Efek sketsa pensil (grayscale, invert, Gaussian blur, *color dodge*) | `Blur Kernel`: ganjil `3`–`31` (default `21`) |
| 13 | `Vignette` | Efek | Penggelapan bagian tepi citra memakai mask radial Gaussian | `Strength`: `0.00`–`1.00` (default `0.60`), `Radius`: `0.20`–`1.50` (default `0.80`) |
| 14 | `Add Gaussian Noise` | Noise | Menambahkan noise berdistribusi normal (mean `0`) | `Sigma`: `5`–`80` (default `25`) |
| 15 | `Add Salt & Pepper` | Noise | Mengganti sebagian piksel acak menjadi `0` atau `255` | `Amount (%)`: `1`–`20` (default `5`) |
| 16 | `Denoising` | Noise | Pengurangan noise dengan tiga metode pilihan | `Method`: `Median`/`Gaussian`/`Bilateral`, `Kernel Size`: ganjil `3`–`31` (default `5`), `Sigma`: `5`–`150` (default `75`, aktif pada `Bilateral`) |

---

## Mode Pengolahan

### 1. Single Filter (Default)
- **Pada citra biasa:** Setiap kali tombol `Process` ditekan, filter yang sedang dipilih diterapkan dari salinan baru `original_image`. Memilih filter lain lalu menekan `Process` kembali memulai dari `original_image`, bukan dari hasil sebelumnya.
- **Pada Live Processing:** Filter yang sedang dipilih dan nilai parameternya langsung diterapkan pada setiap frame kamera mentah terbaru tanpa perlu menekan `Process`.

### 2. Filter Chaining
- **Pada citra biasa:** Operasi `Process` pertama dalam sebuah sesi chaining dimulai dari `original_image`. Operasi `Process` berikutnya menggunakan `current_image` (hasil proses sebelumnya). Setiap tombol filter yang berhasil diproses tetap disorot dengan warna aksen. Menekan `Reset`, mengganti mode pemrosesan, atau memuat gambar baru akan memulai sesi chaining baru.
- **Pada Live Processing:** Saat `Filter Chaining` aktif bersama `Live Processing`, tombol `Add Filter` dan `Clear Chain` muncul di area kontrol bawah. Setiap klik `Add Filter` menambahkan filter terpilih beserta snapshot parameternya ke `live_chain` sesuai urutan (tanpa duplikasi). Setiap frame kamera asli diproses berurutan melewati seluruh rantai di `live_chain` (terpisah dari riwayat chaining gambar biasa). Klik `Clear Chain` mengosongkan rantai dan mengembalikan panel `Sesudah` ke frame mentah.

---

## Live Processing dan Capture

1. Klik **Live Processing** pada toolbar atas untuk menyalakan webcam langsung ke workspace utama (tombol berubah menjadi **Stop Live**).
2. Panel **Sebelum** selalu menampilkan frame kamera mentah, sedangkan panel **Sesudah** menampilkan hasil pemrosesan.
3. **Histogram Asli** membaca frame kamera mentah dan **Histogram Hasil** membaca frame hasil pemrosesan secara dinamis (diperbarui berkala setiap `0.15` detik dan langsung diperbarui ketika filter, parameter, atau mode berubah).
4. Selama `Live Processing` aktif, pengguna bebas mengganti filter, menggeser parameter, atau menyusun `live_chain` tanpa menekan `Process`.
5. Klik **Capture** pada toolbar saat `Live Processing` berjalan untuk mengambil output filter yang sedang tampil pada panel `Sesudah` (bukan frame kamera mentah). Hasil `Capture` menghentikan kamera, lalu masuk ke workspace sebagai `original_image` dan `current_image` baru beresolusi penuh sehingga siap diolah lanjut atau disimpan lewat `Save File`.
6. Klik **Stop Live** untuk menghentikan kamera kapan saja dan mengembalikan workspace ke gambar yang sebelumnya dibuka.

---

## Instalasi dan Menjalankan Aplikasi

Pastikan Anda berada pada direktori root proyek terlebih dahulu, lalu masuk ke folder aplikasi `mengolah_citra_upn/`:

```powershell
cd mengolah_citra_upn
```

1. **Buat virtual environment** (lewati jika `.venv` sudah tersedia):
   ```powershell
   python -m venv .venv
   ```

2. **Aktifkan virtual environment di Windows PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instal dependensi dari `requirements.txt`:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```powershell
   python main.py
   ```

5. **Jalankan unit test filter:**
   ```powershell
   python -m unittest discover -p "test_*.py" -v
   ```

---

## Struktur Proyek

```text
Mengolah Citra UPN/
├── PRD.md                    # Dokumen spesifikasi kebutuhan produk 
├── main.py               # Entry point aplikasi
├── app.py                # Jendela utama, tata letak GUI, manajemen state, dan alur kerja
├── filters.py            # Fungsi pengolahan citra murni (OpenCV & NumPy, BGR uint8)
├── histogram.py          # Perhitungan dan panel histogram Matplotlib (TkAgg)
├── camera.py             # Akses webcam (CameraSession), worker live, dan jendela Capture
├── theme.py              # Konstanta warna, tipografi, ukuran jendela, dan spasi
├── test_filters.py       # Unit test mandiri untuk seluruh fungsi pada filters.py
├── requirements.txt      # Daftar dependensi Python
└── README.md             # Dokumentasi proyek
```

---

## Batasan dan Catatan

- Fitur `Open Camera` dan `Live Processing` mengakses webcam default dengan indeks `0` (`cv2.VideoCapture(0)`). Jika kamera tidak terpasang atau sedang dipakai aplikasi lain, aplikasi menampilkan pesan error pada status bar tanpa mengalami crash.
- Filter komputasi berat seperti `Denoising` dengan metode `Bilateral` berkernel besar memiliki waktu pemrosesan per frame yang lebih tinggi dibandingkan filter ringan; worker `LiveFrameProcessor` selalu memproses frame terbaru saat idle agar antarmuka tetap responsif.
- Pemrosesan dan penyimpanan citra (`Save File`) selalu menggunakan resolusi penuh dari `current_image`, sedangkan pratinjau tampilan dibatasi hingga dimensi maksimum `2000 px` agar UI tetap ringan untuk citra beresolusi besar.

---

## Menjalankan Aplikasi

1. Pastikan komputer sudah terinstall python. Apabila belum bisa didownload pada [link berikut](https://www.python.org/downloads/). 
2. Apabila sudah terinstall, buka terminal dan ketikkan: `python main.py`
