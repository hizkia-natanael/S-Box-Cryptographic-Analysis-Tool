# S-Box Analyzer

## Deskripsi
**S-Box Analyzer** adalah aplikasi berbasis GUI yang dirancang untuk menganalisis properti kriptografi dari Substitution Box (S-Box) menggunakan berbagai parameter seperti:

- **NL** (Nonlinearity)
- **SAC** (Strict Avalanche Criterion)
- **BIC-NL** (Bit Independence Criterion - Nonlinearity)
- **BIC-SAC** (Bit Independence Criterion - SAC)
- **LAP** (Linear Approximation Probability)
- **DAP** (Differential Approximation Probability)

Aplikasi ini dikembangkan menggunakan Python dan Streamlit.

## Fitur
- Memuat S-Box dari file Excel (.xlsx atau .xls).
- Menampilkan tabel S-Box yang diunggah.
- Melakukan analisis S-Box berdasarkan properti kriptografi.
- Menyediakan hasil analisis dalam format yang dapat diunduh (file Excel).

## Instalasi
1. Pastikan Python 3.8 atau versi yang lebih baru telah terinstal.
2. Clone repositori ini:
   ```bash
   git clone https://github.com/username/sbox-analyzer.git
   cd sbox-analyzer
   ```
3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Penggunaan
1. Jalankan aplikasi dengan perintah berikut:
   ```bash
   streamlit run app.py
   ```
2. Buka browser Anda dan akses aplikasi di `http://localhost:8501`.
3. Unggah file Excel berisi S-Box pada aplikasi.
4. Pilih parameter analisis yang diinginkan.
5. Klik tombol **Analyze S-Box** untuk memulai analisis.
6. Unduh hasil analisis dalam format Excel.

## Struktur File
- `app.py`: File utama aplikasi Streamlit.
- `requirements.txt`: Daftar dependensi Python yang diperlukan.

## Contoh Input
Unggah file Excel dengan format seperti berikut:

|   | 0  | 1  | 2  | 3  |
|---|----|----|----|----|
| 0 | 12 | 5  | 9  | 15 |
| 1 | 2  | 14 | 11 | 6  |

**Catatan**: Pastikan S-Box yang diunggah berbentuk matriks persegi.

## Dibuat Oleh
Kelompok 3 Rombel 2:
- Hizkia Natanael Richardo (4611422053)
- Nabil Mutawakkil Qisthi (4611422054)
- Fathimah Az-Zahra Sanjani (4611422057)
- Melinda Wijaya (4611422060)

## Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).
