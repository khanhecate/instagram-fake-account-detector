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

## Data dan pelatihan

Seluruh proses ada di [`notebook/fake_account_detection.ipynb`](notebook/fake_account_detection.ipynb) (dijalankan di Google Colab).

- **Data**: dataset Kaggle [`rajumavinmar/fake-instagram-profile-dataset`](https://www.kaggle.com/datasets/rajumavinmar/fake-instagram-profile-dataset), 5.000 akun dengan 11 fitur dan label `fake`. Kelas seimbang (2.500 asli, 2.500 palsu). Setelah 66 baris duplikat dibuang, tersisa 4.934 akun.
- **Pembagian data**: 80% latih (3.947) dan 20% uji (987), stratified, `random_state=42`. `data_latih.csv` di repo ini adalah bagian latih saja, bukan seluruh dataset.
- **Preprocessing**: `log1p` pada `#posts`, `#followers`, `#follows` karena distribusinya sangat miring, lalu `StandardScaler` pada semua fitur.
- **Tuning**: `GridSearchCV` dengan 5-fold stratified CV dan metrik F1, hanya pada data latih. Data uji dipakai sekali di akhir.

## Perbandingan model

Tiga model dibandingkan dengan pipeline preprocessing yang sama. Tabel diurutkan berdasarkan CV F1 (data latih), bukan hasil data uji.

| Model | CV F1 (latih) | Akurasi | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **0,9860** | 0,9828 | 0,9897 | 0,9756 | 0,9826 | **0,9991** |
| SVM (RBF) | 0,9787 | **0,9838** | **0,9917** | 0,9756 | **0,9836** | 0,9948 |
| Decision Tree | 0,9698 | 0,9777 | 0,9796 | 0,9756 | 0,9776 | 0,9777 |

Hyperparameter terbaik: Random Forest `n_estimators=300, max_depth=None`; SVM `C=10, gamma='scale'`; Decision Tree `max_depth=None, min_samples_leaf=1`.

Laporan klasifikasi Random Forest pada data uji:

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| Asli | 0,98 | 0,99 | 0,98 | 495 |
| Palsu | 0,99 | 0,98 | 0,98 | 492 |

## Kenapa Random Forest

1. **Dipilih dari data latih, bukan data uji.** Random Forest punya CV F1 tertinggi (0,9860 dibanding 0,9787 untuk SVM dan 0,9698 untuk Decision Tree). Dengan begitu data uji tidak ikut menentukan pilihan.
2. **Selisih di data uji dengan SVM sangat kecil.** SVM unggul 0,001 pada akurasi dan F1 data uji. Dengan 987 sampel, itu setara sekitar satu akun, jadi bukan alasan yang cukup untuk memilih SVM. Recall keduanya sama (0,9756).
3. **ROC-AUC tertinggi (0,9991).** Random Forest paling baik memisahkan akun asli dan palsu di semua ambang, jadi probabilitas yang ditampilkan di aplikasi lebih bisa dipakai. SVM bawaan tidak menghasilkan probabilitas tanpa kalibrasi tambahan.
4. **Bisa dijelaskan.** Random Forest memberi `feature_importances_`. Fitur terpenting: `profile pic` (0,237), `#followers` (0,199), `nums/length username` (0,176), `#posts` (0,170), `description length` (0,106). Fitur `private`, `external URL`, dan `name==username` hampir tidak berpengaruh (di bawah 0,01).
5. **Lebih stabil daripada satu pohon.** Decision Tree paling lemah di semua metrik dan paling mudah overfit.

## Batas penggunaan

Akurasi 98% hanya berlaku untuk data seperti dataset latih. Jangan dibaca sebagai akurasi di Instagram nyata.

- **Dataset kecil dan satu sumber.** Hanya 4.934 akun dari satu dataset Kaggle. Pola akun palsu terbaru (misalnya akun palsu yang diberi foto dan bio dari AI) bisa saja tidak ada di sana.
- **Pola yang dipelajari sangat mudah dipisahkan.** Di data latih, median follower akun asli 653 dan akun palsu 43. Median post 77 dibanding 0, dan median panjang bio 27 dibanding 0. Skor ROC-AUC 0,999 menunjukkan data ini hampir terpisah sempurna, jadi akun nyata yang ambigu kurang terwakili.
- **Bias terhadap akun asli yang baru atau sepi.** Pengujian manual pada model yang dipakai aplikasi ini:

  | Profil | Probabilitas palsu |
  |---|---|
  | Pengguna baru, ada foto, 3 post, 40 follower, bio kosong | 52,7% (dianggap palsu) |
  | Pengguna baru, tanpa foto, 0 post, 20 follower | 98,3% |
  | Akun usaha kecil, ada foto, 50 post, 300 follower, ada bio | 4,3% |

  Orang asli yang jarang aktif bisa salah ditandai palsu. Di data latih, 132 dari 1.981 akun asli punya kurang dari 100 follower, jadi model hanya punya sedikit contoh untuk kasus seperti ini.
- **Hanya fitur profil.** Model tidak melihat isi postingan, riwayat waktu, pola komentar, atau jaringan follower. Akun palsu yang dibuat rapi bisa lolos.
- **Data diisi manual.** Hasil bergantung pada ketepatan input pengguna.
- **Ambang 50% adalah default.** Untuk kasus yang lebih mementingkan "jangan salah menuduh akun asli", naikkan ambang. Untuk kasus yang lebih mementingkan "jangan sampai ada akun palsu lolos", turunkan.
- **Bukan untuk keputusan yang berdampak pada orang.** Jangan dipakai sendirian untuk memblokir akun, menuduh seseorang, atau menolak layanan. Pakai sebagai penyaring awal, lalu verifikasi manual.
- **Terikat versi library.** Model dilatih dengan scikit-learn 1.6.1.

## Struktur

```
app.py               aplikasi Streamlit
model_terbaik.pkl    model terlatih (Pipeline + Random Forest)
data_latih.csv       bagian latih dari dataset (fitur + label `fake`)
notebook/            notebook pelatihan dan perbandingan model
requirements.txt     dependensi
.streamlit/          konfigurasi tema
```

## Keamanan

`model_terbaik.pkl` dimuat dengan `joblib` (berbasis pickle). Jangan mengganti file ini dengan file dari sumber yang tidak dipercaya, karena pickle dapat mengeksekusi kode saat dimuat.
