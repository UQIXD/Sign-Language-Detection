# Manual Book — Sign Language Detection (Isyarat Kita)

Panduan cara menjalankan project deteksi bahasa isyarat BISINDO real-time pakai YOLOv8 + webcam. Ada 2 cara: pakai versi yang sudah di-deploy (tanpa install apapun), atau jalankan sepenuhnya di komputer sendiri (lokal).

---

## Cara 1 — Pakai Versi Deploy (paling cepat)

Backend deteksi jalan di Google Colab, frontend sudah online di Netlify.

1. Buka Google Colab ini: https://colab.research.google.com/drive/1iEm8iKSnOP_KApThe5DWhSeIV-BUePyu?usp=sharing
2. Jalankan cell akan menyalakan backend + tunnel (ngrok) supaya bisa diakses dari luar Colab.
3. Setelah backend aktif, buka: https://uqi-sld.netlify.app/
4. Izinkan akses kamera saat browser minta izin.
5. Arahkan tangan ke kamera — hasil deteksi (label + bounding box) akan tampil langsung.

---

## Cara 2 — Jalankan Lokal (di komputer sendiri)

### Prasyarat

- Python sudah terinstall (disarankan Python 3.9–3.11, kompatibel dengan Ultralytics/PyTorch).
- Git sudah terinstall.
- VS Code + ekstensi **Live Server**.
- Webcam.

### Langkah-langkah

1. **Clone repo:**

   ```
   git clone https://github.com/UQIXD/Sign-Language-Detection.git
   cd Sign-Language-Detection
   ```

2. **Buat & aktifkan virtual environment:**

   ```
   python -m venv venv
   ```

   Windows:

   ```
   venv\Scripts\activate
   ```

   Linux/Mac:

   ```
   source venv/bin/activate
   ```

3. **Install dependency:**

   ```
   pip install -r requirements.txt
   ```

4. **Jalankan backend (server deteksi):**

   ```
   py app.py
   ```

   (atau `python app.py` kalau `py` tidak dikenali). Server jalan di `http://127.0.0.1:5000`. Biarkan terminal ini tetap terbuka selama dipakai.

5. **Buka frontend:** buka file `local.html` di VS Code, klik kanan → **Open with Live Server**. File ini sudah diset untuk connect ke backend lokal (`http://127.0.0.1:5000/detect`).

6. **Izinkan akses kamera** di browser, lalu mulai deteksi.

**Catatan model:** file model `best.pt` sudah ada di root folder repo dan otomatis dipakai `app.py` — tidak perlu download terpisah.

---

## Struktur File Penting

| File                        | Fungsi                                                                          |
| --------------------------- | ------------------------------------------------------------------------------- |
| `app.py`                    | Backend Flask, load model `best.pt`, endpoint `POST /detect`                    |
| `best.pt`                   | Model YOLOv8 (dipakai backend Flask/`app.py`)                                   |
| `best.onnx`                 | Model versi ONNX (dipakai `onnx.html`, jalan langsung di browser tanpa backend) |
| `local.html`                | Frontend untuk dipasangkan dengan backend lokal (`app.py`)                      |
| `index.html` / `colab.html` | Frontend untuk dipasangkan dengan backend Colab (ngrok)                         |
| `hug.html`                  | Frontend untuk dipasangkan dengan backend Hugging Face Space                    |
| `requirements.txt`          | Daftar dependency Python                                                        |
| `css/style.css`, `img/`     | Aset tampilan (hasil build Tailwind, ikon huruf/kosakata)                       |

---

## Troubleshooting

- **Port 5000 sudah dipakai:** matikan proses lain yang pakai port itu, atau jalankan `app.py` dengan env var `PORT` lain (`set PORT=5001` di Windows sebelum run).
- **Kamera tidak muncul / izin ditolak:** cek permission kamera di browser (klik ikon gembok di address bar), pastikan tidak dipakai aplikasi lain.
- **Frontend gagal connect ke backend (CORS/network error):** pastikan `app.py` masih jalan di terminal, dan pastikan buka `local.html` (bukan `index.html`/`colab.html`) untuk mode lokal.
- **Model gagal load:** pastikan file `best.pt` ada di root folder yang sama dengan `app.py`.
- **`pip install` error/ketinggalan paket:** pastikan venv sudah aktif (`venv\Scripts\activate` / `source venv/bin/activate`) sebelum `pip install -r requirements.txt`.

---

## Referensi Tambahan

Untuk info dataset, proses training model, dan link Colab training, lihat [README.md](README.md).
