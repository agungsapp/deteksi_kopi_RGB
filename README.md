# Proyek Deteksi Kualitas Biji Kopi Berdasarkan RGB

## Persyaratan Sistem

-   Python 3.7 atau lebih baru
-   pip (biasanya udah terinstal bersama Python)

## instalasi

1. Clone repositori ini:

```
git clone https://github.com/agungsapp/deteksi_kopi_RGB.git
```

2. Buat virtual environment:

```
python -m venv venv
```

3. Aktifkan virtual environment:

-   Untuk Windows:
    ```
    venv\Scripts\activate
    ```
-   Untuk macOS dan Linux:
    ```
    source venv/bin/activate
    ```

Setelah diaktifkan, Anda akan melihat `(venv)` di awal prompt command line Anda.

4. Instal dependensi:

```
pip install -r requirements.txt
```

## Menjalankan Program

1. Pastikan virtual environment sudah aktif (lihat langkah 3 di atas jika belum).

2. Jalankan aplikasi utama:

```
python main.py
```

## Struktur Proyek

-   `main.py`: File utama untuk menjalankan aplikasi GUI
-   `train.py`: Script untuk melatih model
-   `evaluate.py`: Script untuk mengevaluasi model
-   `coffee_model.h5`: Model yang sudah dilatih
-   `dataset/`: Folder berisi dataset (pastikan sudah terisi)
-   `requirements.txt`: Daftar dependensi Python

## Troubleshooting

-   Jika muncul error saat instalasi dependensi, coba update pip:

```
pip install --upgrade pip
```

Kemudian ulangi langkah instalasi dependensi.

-   Jika ada masalah dengan TensorFlow, pastikan versi Python Anda kompatibel dengan versi TensorFlow yang digunakan.

-   Untuk pengguna Windows, jika ada masalah dengan eksekusi script, coba jalankan PowerShell sebagai Administrator dan eksekusi:

```
Set-ExecutionPolicy RemoteSigned
```
