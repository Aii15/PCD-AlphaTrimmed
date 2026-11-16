# PCD-AlphaTrimmed


# Adaptive Alpha-Trimmed Mean Filter

Program ini mengimplementasikan filter adaptive alpha-trimmed mean dari
Öten dan de Figueiredo, *Adaptive Alpha-Trimmed Mean Filters Under Deviations
From Assumed Noise Model*, IEEE Transactions on Image Processing, 2004.

## Menjalankan

Pastikan Python 3.10+ serta `numpy` dan `Pillow` tersedia:

```powershell
python -m pip install numpy Pillow
python app.py
```

Alur aplikasi: **Buka** gambar RGB, pilih model noise, **Tambahkan Noise**,
lalu **Restorasi**. Ukuran window harus ganjil. Filter memproses setiap kanal
warna, memulai dari trim maksimum dan mengurangi trim sampai bagian tengah
window memenuhi uji simetri lokal. Parameter eksperimen paper adalah window
5×5, threshold 0.5, dan `max_alpha=0.45`.

API inti juga dapat dipakai dari Python:

```python
from PIL import Image
import numpy as np
from adaptive_alpha_trimmed import adaptive_alpha_trimmed

image = np.array(Image.open("input.png").convert("RGB"))
restored = adaptive_alpha_trimmed(image, window_size=5)
Image.fromarray(restored).save("restored.png")
```
