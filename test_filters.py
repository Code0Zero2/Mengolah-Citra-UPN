"""Pengujian unit mandiri untuk seluruh fungsi filter pada filters.py."""

import unittest
import cv2
import numpy as np

import filters


def _make_synthetic_bgr(height: int = 64, width: int = 96) -> np.ndarray:
    """Membuat citra uji BGR sintetis dengan gradien warna dan bentuk geometris."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]

    b = np.tile(x, (height, 1))
    g = np.tile(y, (1, width))
    r = ((b.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
    img = np.stack([b, g, r], axis=-1)

    cv2.rectangle(img, (16, 12), (48, 44), (20, 210, 240), -1)
    cv2.circle(img, (70, 34), 14, (230, 50, 40), 2)
    return img


class TestFilters(unittest.TestCase):
    """Menguji seluruh algoritma pengolahan citra di filters.py."""

    def setUp(self) -> None:
        self.img = _make_synthetic_bgr()
        self.snapshot = self.img.copy()

    def _assert_valid_bgr(self, out: np.ndarray) -> None:
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.array_equal(self.img, self.snapshot))

    def test_all_filters_default_and_immutability(self) -> None:
        calls = [
            lambda im: filters.grayscale(im),
            lambda im: filters.negative(im),
            lambda im: filters.brightness(im),
            lambda im: filters.thresholding(im),
            lambda im: filters.saturation_hue(im),
            lambda im: filters.equalization(im),
            lambda im: filters.blur(im),
            lambda im: filters.sharpening(im),
            lambda im: filters.edge_detection(im),
            lambda im: filters.mirroring(im),
            lambda im: filters.rotate(im),
            lambda im: filters.sketch(im),
            lambda im: filters.vignette(im),
            lambda im: filters.add_gaussian_noise(im, seed=1),
            lambda im: filters.add_salt_pepper_noise(im, seed=1),
            lambda im: filters.denoising(im),
        ]
        for fn in calls:
            out = fn(self.img)
            self._assert_valid_bgr(out)

    def test_grayscale_and_negative(self) -> None:
        gray_bgr = filters.grayscale(self.img)
        self._assert_valid_bgr(gray_bgr)
        self.assertTrue(np.array_equal(gray_bgr[:, :, 0], gray_bgr[:, :, 1]))
        self.assertTrue(np.array_equal(gray_bgr[:, :, 1], gray_bgr[:, :, 2]))

        neg = filters.negative(self.img)
        self._assert_valid_bgr(neg)
        self.assertTrue(np.array_equal(neg, 255 - self.img))

    def test_brightness_boundaries(self) -> None:
        zero = filters.brightness(self.img, value=0)
        dark = filters.brightness(self.img, value=-100)
        bright = filters.brightness(self.img, value=100)
        self._assert_valid_bgr(zero)
        self._assert_valid_bgr(dark)
        self._assert_valid_bgr(bright)
        self.assertTrue(np.array_equal(zero, self.img))
        self.assertLessEqual(float(dark.mean()), float(self.img.mean()))
        self.assertGreaterEqual(float(bright.mean()), float(self.img.mean()))

        with self.assertRaises(ValueError):
            filters.brightness(self.img, value=-101)
        with self.assertRaises(ValueError):
            filters.brightness(self.img, value=101)

    def test_thresholding_boundaries(self) -> None:
        t0 = filters.thresholding(self.img, threshold=0)
        t127 = filters.thresholding(self.img, threshold=127)
        t255 = filters.thresholding(self.img, threshold=255)
        for out in (t0, t127, t255):
            self._assert_valid_bgr(out)
            self.assertTrue(set(np.unique(out)).issubset({0, 255}))

        self.assertTrue(np.all(t0 == 255))
        with self.assertRaises(ValueError):
            filters.thresholding(self.img, threshold=-1)
        with self.assertRaises(ValueError):
            filters.thresholding(self.img, threshold=256)

    def test_saturation_hue_boundaries(self) -> None:
        desat = filters.saturation_hue(self.img, hue=0, saturation=0)
        self._assert_valid_bgr(desat)
        self.assertTrue(np.array_equal(desat[:, :, 0], desat[:, :, 1]))
        self.assertTrue(np.array_equal(desat[:, :, 1], desat[:, :, 2]))

        h_min = filters.saturation_hue(self.img, hue=-90, saturation=200)
        h_max = filters.saturation_hue(self.img, hue=90, saturation=150)
        self._assert_valid_bgr(h_min)
        self._assert_valid_bgr(h_max)

        with self.assertRaises(ValueError):
            filters.saturation_hue(self.img, hue=-91)
        with self.assertRaises(ValueError):
            filters.saturation_hue(self.img, hue=91)
        with self.assertRaises(ValueError):
            filters.saturation_hue(self.img, saturation=-1)
        with self.assertRaises(ValueError):
            filters.saturation_hue(self.img, saturation=201)

    def test_equalization_preserves_shape_and_changes_luminance(self) -> None:
        low_contrast = np.clip(self.img // 3 + 80, 0, 255).astype(np.uint8)
        eq = filters.equalization(low_contrast)
        self.assertEqual(eq.shape, low_contrast.shape)
        self.assertEqual(eq.dtype, np.uint8)
        self.assertGreater(int(eq.max()) - int(eq.min()), int(low_contrast.max()) - int(low_contrast.min()))

    def test_blur_methods_and_kernel_validation(self) -> None:
        for method in ("Gaussian", "Average"):
            for k in (3, 31):
                out = filters.blur(self.img, method=method, kernel_size=k)
                self._assert_valid_bgr(out)

        for bad_k in (1, 2, 4, 16, 32, 33):
            with self.assertRaises(ValueError):
                filters.blur(self.img, method="Gaussian", kernel_size=bad_k)
        with self.assertRaises(ValueError):
            filters.blur(self.img, method="Invalid", kernel_size=5)

    def test_sharpening_boundaries(self) -> None:
        low = filters.sharpening(self.img, strength=0.5)
        high = filters.sharpening(self.img, strength=3.0)
        self._assert_valid_bgr(low)
        self._assert_valid_bgr(high)

        with self.assertRaises(ValueError):
            filters.sharpening(self.img, strength=0.4)
        with self.assertRaises(ValueError):
            filters.sharpening(self.img, strength=3.1)

    def test_edge_detection_sobel_and_canny(self) -> None:
        sobel = filters.edge_detection(self.img, method="Sobel")
        canny_def = filters.edge_detection(self.img, method="Canny", threshold=100)
        canny_pair = filters.edge_detection(self.img, method="Canny", threshold=0, threshold2=255)
        for out in (sobel, canny_def, canny_pair):
            self._assert_valid_bgr(out)
            self.assertTrue(np.array_equal(out[:, :, 0], out[:, :, 1]))
            self.assertTrue(np.array_equal(out[:, :, 1], out[:, :, 2]))

        self.assertGreater(int(sobel.max()), 0)
        self.assertGreater(int(canny_def.max()), 0)

        with self.assertRaises(ValueError):
            filters.edge_detection(self.img, method="Laplacian")
        with self.assertRaises(ValueError):
            filters.edge_detection(self.img, method="Canny", threshold=256)

    def test_mirroring_horizontal_and_vertical(self) -> None:
        horiz = filters.mirroring(self.img, direction="Horizontal")
        vert = filters.mirroring(self.img, direction="Vertical")
        self._assert_valid_bgr(horiz)
        self._assert_valid_bgr(vert)
        self.assertTrue(np.array_equal(horiz[:, 0], self.img[:, -1]))
        self.assertTrue(np.array_equal(vert[0, :], self.img[-1, :]))

        with self.assertRaises(ValueError):
            filters.mirroring(self.img, direction="Diagonal")

    def test_rotate_angles_expansion_and_validation(self) -> None:
        r_zero = filters.rotate(self.img, angle=0)
        self._assert_valid_bgr(r_zero)
        self.assertTrue(np.array_equal(r_zero, self.img))
        self.assertFalse(np.shares_memory(r_zero, self.img))

        # Rotasi 90 dan -90 derajat menukar lebar dan tinggi tanpa memotong citra
        r_90 = filters.rotate(self.img, angle=90)
        r_neg90 = filters.rotate(self.img, angle=-90)
        self.assertEqual(r_90.shape, (self.img.shape[1], self.img.shape[0], 3))
        self.assertEqual(r_neg90.shape, (self.img.shape[1], self.img.shape[0], 3))
        self.assertEqual(r_90.dtype, np.uint8)
        self.assertTrue(np.array_equal(self.img, self.snapshot))

        # Rotasi 45 dan -45 derajat memperluas kanvas dengan latar hitam pada sudut kosong
        solid = np.full((40, 80, 3), (120, 160, 200), dtype=np.uint8)
        r_45 = filters.rotate(solid, angle=45)
        r_neg45 = filters.rotate(solid, angle=-45)
        self.assertGreater(r_45.shape[0], solid.shape[0])
        self.assertGreater(r_45.shape[1], solid.shape[1])
        self.assertEqual(r_45.shape, r_neg45.shape)
        self.assertEqual(tuple(r_45[0, 0]), (0, 0, 0))

        # Batas -180 dan +180 derajat
        r_180 = filters.rotate(self.img, angle=180)
        r_neg180 = filters.rotate(self.img, angle=-180)
        self._assert_valid_bgr(r_180)
        self._assert_valid_bgr(r_neg180)

        with self.assertRaises(ValueError):
            filters.rotate(self.img, angle=181)
        with self.assertRaises(ValueError):
            filters.rotate(self.img, angle=-181)
        with self.assertRaises(TypeError):
            filters.rotate(self.img, angle=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filters.rotate(np.zeros((10, 10), dtype=np.uint8), angle=15)

    def test_sketch_and_vignette(self) -> None:
        s_min = filters.sketch(self.img, kernel_size=3)
        s_max = filters.sketch(self.img, kernel_size=31)
        self._assert_valid_bgr(s_min)
        self._assert_valid_bgr(s_max)
        self.assertTrue(np.array_equal(s_min[:, :, 0], s_min[:, :, 1]))

        v_zero = filters.vignette(self.img, strength=0.0, radius=0.8)
        v_strong = filters.vignette(self.img, strength=1.0, radius=0.2)
        self._assert_valid_bgr(v_zero)
        self._assert_valid_bgr(v_strong)
        self.assertTrue(np.array_equal(v_zero, self.img))
        self.assertLess(float(v_strong[-1, -1].mean()), float(self.img[-1, -1].mean()))
        self.assertLess(float(v_strong.mean()), float(self.img.mean()))

        with self.assertRaises(ValueError):
            filters.vignette(self.img, strength=1.2)
        with self.assertRaises(ValueError):
            filters.vignette(self.img, radius=0.1)

    def test_noise_boundaries_and_determinism(self) -> None:
        g1 = filters.add_gaussian_noise(self.img, sigma=5, seed=42)
        g2 = filters.add_gaussian_noise(self.img, sigma=5, seed=42)
        g_high = filters.add_gaussian_noise(self.img, sigma=80, seed=7)
        self._assert_valid_bgr(g1)
        self._assert_valid_bgr(g_high)
        self.assertTrue(np.array_equal(g1, g2))

        sp1 = filters.add_salt_pepper_noise(self.img, amount=1, seed=42)
        sp2 = filters.add_salt_pepper_noise(self.img, amount=1, seed=42)
        sp_high = filters.add_salt_pepper_noise(self.img, amount=20, seed=7)
        self._assert_valid_bgr(sp1)
        self._assert_valid_bgr(sp_high)
        self.assertTrue(np.array_equal(sp1, sp2))

        with self.assertRaises(ValueError):
            filters.add_gaussian_noise(self.img, sigma=4)
        with self.assertRaises(ValueError):
            filters.add_gaussian_noise(self.img, sigma=81)
        with self.assertRaises(ValueError):
            filters.add_salt_pepper_noise(self.img, amount=0)
        with self.assertRaises(ValueError):
            filters.add_salt_pepper_noise(self.img, amount=21)

    def test_denoising_all_three_methods(self) -> None:
        noisy_g = filters.add_gaussian_noise(self.img, sigma=30, seed=99)
        noisy_sp = filters.add_salt_pepper_noise(self.img, amount=10, seed=99)

        d_med = filters.denoising(noisy_sp, method="Median", kernel_size=3)
        d_gau = filters.denoising(noisy_g, method="Gaussian", kernel_size=5)
        d_bil = filters.denoising(noisy_g, method="Bilateral", kernel_size=7, sigma=75)

        for out in (d_med, d_gau, d_bil):
            self._assert_valid_bgr(out)

        # Pastikan denoising menurunkan error terhadap citra bersih
        mse_sp_before = float(np.mean((noisy_sp.astype(np.float32) - self.img.astype(np.float32)) ** 2))
        mse_sp_after = float(np.mean((d_med.astype(np.float32) - self.img.astype(np.float32)) ** 2))
        self.assertLess(mse_sp_after, mse_sp_before)

        mse_g_before = float(np.mean((noisy_g.astype(np.float32) - self.img.astype(np.float32)) ** 2))
        mse_g_after = float(np.mean((d_gau.astype(np.float32) - self.img.astype(np.float32)) ** 2))
        self.assertLess(mse_g_after, mse_g_before)

        with self.assertRaises(ValueError):
            filters.denoising(self.img, method="Unknown")
        with self.assertRaises(ValueError):
            filters.denoising(self.img, method="Median", kernel_size=4)
        with self.assertRaises(ValueError):
            filters.denoising(self.img, method="Bilateral", kernel_size=5, sigma=2)

    def test_invalid_image_and_parameter_types(self) -> None:
        with self.assertRaises(TypeError):
            filters.grayscale(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            filters.negative([[1, 2, 3]])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filters.grayscale(np.zeros((0, 0, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            filters.grayscale(np.zeros((20, 20), dtype=np.uint8))
        with self.assertRaises(ValueError):
            filters.grayscale(np.zeros((20, 20, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            filters.grayscale(np.zeros((20, 20, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            filters.brightness(self.img, value=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            filters.blur(self.img, kernel_size=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filters.sharpening(self.img, strength=float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
