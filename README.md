# Deteksi Akun Palsu Instagram

Aplikasi web [Streamlit](https://streamlit.io) yang memperkirakan apakah sebuah akun Instagram palsu atau asli. Pengguna mengisi data profil secara manual, lalu model machine learning menghitung probabilitas akun palsu.

Tidak ada koneksi ke Instagram. Semua data diisi sendiri dan tidak dikirim ke mana pun.

## Cara menjalankan

Kebutuhan: Python 3.10 atau lebih baru (diuji di 3.12).

```bash
git clone <url-repo-ini>
cd <nama-folder-repo>

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Buka http://localhost:8501 jika browser tidak terbuka otomatis. Tekan `Ctrl+C` di terminal untuk berhenti.

> `model_terbaik.pkl` dilatih dengan scikit-learn 1.6.1. Versi library lain bisa gagal memuat model atau menghasilkan peringatan, jadi pakai versi di `requirements.txt`.

## Cara kerja

### 1. Input
Form di `app.py` meminta:

| Input | Dipakai untuk |
|---|---|
| Username, nama lengkap, bio | Fitur teks (lihat di bawah) |
| Foto profil, URL eksternal, akun privat | Fitur 0/1 |
| Jumlah post, follower, following | Fitur angka |

Tombol **Contoh akun asli**, **Contoh akun palsu**, dan **Kosongkan** mengisi atau mereset form lewat `st.session_state`.

### 2. Fitur
Dari input, aplikasi menghitung 11 fitur dengan nama dan urutan yang sama seperti data latih:

| Fitur | Arti |
|---|---|
| `profile pic` | 1 jika punya foto profil |
| `nums/length username` | rasio digit terhadap panjang username |
| `fullname words` | jumlah kata di nama lengkap |
| `nums/length fullname` | rasio digit terhadap panjang nama lengkap |
| `name==username` | 1 jika nama lengkap sama dengan username (tanpa membedakan huruf besar/kecil) |
| `description length` | jumlah karakter bio |
| `external URL` | 1 jika ada URL di profil |
| `private` | 1 jika akun privat |
| `#posts`, `#followers`, `#follows` | jumlah post, follower, following |

### 3. Prediksi
`model_terbaik.pkl` adalah scikit-learn `Pipeline` berisi preprocessing dan `RandomForestClassifier` (300 pohon). Model dimuat sekali dengan `st.cache_resource`. Setiap input berubah, `predict_proba` menghasilkan probabilitas kelas "palsu". Nilai 50% atau lebih ditampilkan sebagai **kemungkinan akun palsu**, selain itu **kemungkinan akun asli**.

### 4. Tampilan
- Kartu hasil di kolom kanan: persentase, vonis, dan bar probabilitas dengan penanda 50%.
- Tabel fitur turunan, termasuk rasio follower per following (hanya info, tidak dipakai model).
- Tema ada di `.streamlit/config.toml`, CSS kartu hasil ada di `app.py`.

## Performa model

Pada data uji (987 akun), model Random Forest mencapai akurasi sekitar 98,3%, F1 98,3%, dan ROC-AUC 0,999. Hasil ini berlaku untuk data latih yang dipakai dan belum tentu berlaku untuk akun Instagram terkini. Anggap hasil sebagai perkiraan, bukan keputusan final.

## Struktur

```
app.py               aplikasi Streamlit
model_terbaik.pkl    model terlatih (Pipeline + Random Forest)
data_latih.csv       data latih (fitur + label `fake`)
requirements.txt     dependensi
.streamlit/          konfigurasi tema
```

## Keamanan

`model_terbaik.pkl` dimuat dengan `joblib` (berbasis pickle). Jangan mengganti file ini dengan file dari sumber yang tidak dipercaya, karena pickle dapat mengeksekusi kode saat dimuat.
