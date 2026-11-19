"""Entry point aplikasi Mengolah Citra UPN."""

from app import MengolahCitraApp


def main() -> None:
    """Menjalankan jendela utama aplikasi."""
    app = MengolahCitraApp()
    app.mainloop()


if __name__ == "__main__":
    main()
