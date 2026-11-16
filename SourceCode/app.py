"""Small desktop application for adaptive alpha-trimmed image restoration."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from adaptive_alpha_trimmed import adaptive_alpha_trimmed, add_noise, psnr


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Adaptive Alpha-Trimmed Mean Filter")
        self.geometry("1100x700")
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.preview = ttk.Label(self)
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=5)
        ttk.Button(controls, text="Buka", command=self.open_image).pack(side="left")
        ttk.Button(controls, text="Tambahkan Noise", command=self.noise).pack(side="left", padx=5)
        ttk.Button(controls, text="Restorasi", command=self.restore).pack(side="left")
        ttk.Button(controls, text="Simpan", command=self.save_image).pack(side="left", padx=5)
        self.model = tk.StringVar(value="salt_pepper")
        ttk.Combobox(
            controls, textvariable=self.model, state="readonly",
            values=("gaussian", "salt_pepper", "speckle", "poisson", "mixed"), width=12
        ).pack(side="left", padx=8)
        ttk.Label(controls, text="Window").pack(side="left")
        self.window = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=3, to=15, increment=2, textvariable=self.window, width=4).pack(side="left")
        self.status = tk.StringVar(value="Buka gambar untuk mulai.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=10, pady=5)

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(filetypes=(("Images", "*.png *.jpg *.jpeg *.bmp *.tif"),))
        if not filename:
            return
        self.original = np.array(Image.open(filename).convert("RGB"))
        self.current = self.original.copy()
        self.reference = self.original.copy()
        self.show_image()
        self.status.set(f"Memuat {Path(filename).name}")

    def noise(self) -> None:
        if self.original is None:
            messagebox.showwarning("Belum ada gambar", "Buka gambar terlebih dahulu.")
            return
        self.current = add_noise(self.original, self.model.get(), amount=0.10, sigma=0.08, seed=None)
        self.show_image()
        self.status.set(f"Noise {self.model.get()} ditambahkan")

    def restore(self) -> None:
        if self.current is None:
            messagebox.showwarning("Belum ada gambar", "Buka gambar terlebih dahulu.")
            return
        try:
            self.current = adaptive_alpha_trimmed(self.current, int(self.window.get()))
        except ValueError as error:
            messagebox.showerror("Parameter tidak valid", str(error))
            return
        self.show_image()
        score = psnr(self.reference, self.current) if self.reference is not None else 0.0
        self.status.set(f"Restorasi selesai | PSNR terhadap gambar awal: {score:.2f} dB")

    def save_image(self) -> None:
        if self.current is None:
            messagebox.showwarning("Belum ada gambar", "Buka gambar terlebih dahulu.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG", "*.png"),))
        if filename:
            Image.fromarray(self.current).save(filename)
            self.status.set(f"Tersimpan: {Path(filename).name}")

    def show_image(self) -> None:
        if self.current is None:
            return
        image = Image.fromarray(self.current)
        image.thumbnail((1050, 580))
        self._photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self._photo)


if __name__ == "__main__":
    App().mainloop()
