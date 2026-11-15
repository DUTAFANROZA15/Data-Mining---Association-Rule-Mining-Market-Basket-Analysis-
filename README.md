# Praktikum Data Mining — Association Rule Mining (Market Basket Analysis)

Praktikum Data Mining, Program Studi Informatika, Universitas Negeri Padang.
Topik: menemukan pola "item apa yang cenderung dibeli bersamaan" dari data transaksi memakai algoritma **Apriori** dan **FP-Growth** dengan pustaka `mlxtend`.

## Kelompok 9

| No | Nama | NIM |
|---|---|---|
| 1 | Duta Fanroza | 24343005 |
| 2 | Fa'iz Thahir Nasution | 24343035 |
| 3 | Rachel Mutiara Hesa | 23343110 |

---

## Daftar Isi

1. [Tujuan Praktikum](#tujuan-praktikum)
2. [Struktur Repositori](#struktur-repositori)
3. [Dataset](#dataset)
4. [Dasar Teori Singkat](#dasar-teori-singkat)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Notebook 1: Langkah Praktikum](#notebook-1-langkah-praktikum-bagian-41-sampai-48)
7. [Notebook 2: Tugas Praktikum](#notebook-2-tugas-praktikum-tugas-1-sampai-4)
8. [Kesimpulan](#kesimpulan)
9. [Daftar Pustaka](#daftar-pustaka)

---

## Tujuan Praktikum

1. Memahami data transaksi (transaction database) dan mengubahnya ke format one-hot dengan `TransactionEncoder`.
2. Menemukan frequent itemset dengan Apriori dan membentuk aturan asosiasi memakai `mlxtend`.
3. Menafsirkan ukuran Support, Confidence, dan Lift, serta Leverage dan Conviction.
4. Menganalisis pengaruh `min_support` dan `min_confidence` terhadap jumlah itemset dan aturan.
5. Membandingkan Apriori dan FP-Growth dari sisi hasil dan waktu eksekusi.
6. Menerjemahkan aturan asosiasi menjadi rekomendasi bisnis, termasuk pada dataset mandiri.

## Struktur Repositori

```
.
├── Praktikum_ARM_1_Langkah.ipynb               # Langkah-langkah praktikum (Bagian 4.1 - 4.8)
├── Praktikum_ARM_2_Tugas.ipynb                 # Pengerjaan Tugas 1 - 4
├── transactions_arm_praktikum.csv              # Dataset 1: daftar item per transaksi
├── transactions_arm_praktikum_onehot.csv       # Dataset 2: matriks one-hot 60 x 12
├── TRANSAKSI_TOKO_BANGUNAN_GABUNGAN.csv        # Dataset mandiri (Tugas 4): 50 transaksi
├── Laporan_Praktikum_ARM.docx                  # Laporan praktikum (opsional di repo)
└── README.md
```

Folder `figs/` dan berkas `rules.csv` dibuat otomatis saat notebook dijalankan.

> Semua file CSV harus berada di folder yang sama dengan notebook.

## Dataset

### Dataset 1 dan 2: transaksi toko kelontong (dari modul)

Dataset sintetis dari modul praktikum: **60 transaksi** dengan **12 jenis item** (Roti, Mentega, Susu, Telur, Kopi, Gula, Teh, Keju, Mie Instan, Kecap, Sabun, Shampo). Setiap transaksi berisi 1 sampai 5 item (rata-rata sekitar 2,7 item).

| Berkas | Format | Kegunaan |
|---|---|---|
| `transactions_arm_praktikum.csv` | kolom `TID`, `Items` (daftar item per transaksi) | dataset utama, di-encode dengan `TransactionEncoder` |
| `transactions_arm_praktikum_onehot.csv` | matriks one-hot 60 x 12 (True/False) | verifikasi hasil encoding dan hasil Apriori |

### Dataset mandiri (Tugas 4): transaksi toko bangunan

`TRANSAKSI_TOKO_BANGUNAN_GABUNGAN.csv` disusun dari dua sumber karena modul mensyaratkan minimal 40 transaksi dan 8 jenis item:

- **30 transaksi asli** dari Kaggle: [Data Transaksi Toko Bangunan](https://www.kaggle.com/datasets/revgaming/data-transaksi-toko-bangunan) (T001 sampai T030, tidak diubah).
- **20 transaksi tambahan** (T031 sampai T050) yang **disusun sendiri** dengan produk, kategori, satuan, dan harga satuan yang sama seperti data asli. Polanya: 8 transaksi pondasi/dinding, 6 pengecatan, 4 instalasi pipa, dan 2 campuran acak.
- Kolom `Sumber` menandai asal tiap baris: `Kaggle (asli)` atau `Tambahan (disusun)`.

Hasilnya 143 baris (format long) yang dikelompokkan per `ID Transaksi` menjadi **50 transaksi** dengan **9 produk**: Semen, Pasir, Batu Bata, Besi Beton, Pipa PVC, Sambungan PVC, Cat Tembok, Cat Kayu, dan Kuas.

> **Catatan:** karena 20 dari 50 transaksi disusun sendiri, hasil Tugas 4 bersifat ilustrasi penerapan metode, bukan temuan perilaku pembeli yang sebenarnya.

## Dasar Teori Singkat

- **Market basket analysis**: mencari pola barang yang dibeli bersamaan dalam data transaksi, untuk cross-selling, penataan rak, bundling promo, dan sistem rekomendasi.
- **Apriori**: mencari frequent itemset bertahap (level-wise) dengan prinsip *downward closure*: jika suatu itemset tidak frequent, semua superset-nya pasti tidak frequent.
- **FP-Growth**: menambang frequent itemset dari struktur FP-Tree tanpa membangkitkan kandidat. Hasilnya sama dengan Apriori, hanya cara kerjanya berbeda.
- Ukuran evaluasi:
  - **Support(X)** = proporsi transaksi yang memuat X.
  - **Confidence(X → Y)** = support(X ∪ Y) / support(X), perkiraan P(Y | X).
  - **Lift(X → Y)** = confidence(X → Y) / support(Y). Lift > 1 berarti korelasi positif, = 1 independen, < 1 korelasi negatif.
  - **Leverage** = support(X ∪ Y) − support(X) · support(Y).
  - **Conviction** = (1 − support(Y)) / (1 − confidence(X → Y)).

## Cara Menjalankan

**Kebutuhan:** Python 3.9+, Jupyter Notebook/JupyterLab, dan pustaka berikut.

```bash
pip install pandas numpy matplotlib mlxtend jupyter
```

**Langkah:**

1. Unduh/clone repositori ini.
2. Pastikan semua file CSV berada di folder yang sama dengan kedua notebook.
3. Jalankan Jupyter:
   ```bash
   jupyter notebook
   ```
4. Buka dan jalankan sel dari atas ke bawah (*Run All*):
   - `Praktikum_ARM_1_Langkah.ipynb` terlebih dahulu.
   - `Praktikum_ARM_2_Tugas.ipynb` (notebook ini bisa dijalankan sendiri, sel persiapannya mengulang hasil Bagian 4 yang dibutuhkan).
5. Isi sel `NAMA`, `NIM`, dan `KELAS` di bagian awal notebook.

> Jika `mlxtend` belum terpasang, pasang lewat terminal/Anaconda Prompt (`pip install mlxtend`). Jangan memakai `%pip` di dalam notebook agar tidak bentrok dengan matplotlib yang sudah termuat.

---

## Notebook 1: Langkah Praktikum (Bagian 4.1 sampai 4.8)

File: `Praktikum_ARM_1_Langkah.ipynb`

| Bagian | Isi | Hasil utama |
|---|---|---|
| 0. Pemeriksaan berkas | Mengecek kedua CSV tersedia | - |
| 4.1 Persiapan data | Memuat Dataset 1, menghitung frekuensi item, diagram batang | 60 transaksi; Susu item terpopuler (support 0,317) |
| 4.2 Encoding one-hot | `TransactionEncoder`, verifikasi silang dengan Dataset 2 | Matriks 60 x 12, identik dengan Dataset 2 |
| 4.3 Apriori | `min_support = 0.15` | **17 frequent itemset** (10 tunggal, 6 pasangan, 1 tiga-item `{Mentega, Roti, Susu}`) |
| 4.4 Aturan asosiasi | `min_confidence = 0.5`, diurutkan menurut Lift | **18 aturan**, seluruhnya Lift > 2 |
| 4.5 Visualisasi | Scatter plot Support-Confidence (ukuran = Lift), bar chart 8 aturan teratas | `figs/gambar3_*.png`, `figs/gambar4_*.png` |
| 4.6 Interpretasi bisnis | Tiga aturan terpilih | lihat tabel di bawah |
| 4.7 Eksperimen `min_support` | Sweep 0.10 sampai 0.40 | 0.10: 19 itemset/20 aturan; 0.15: 17/18; 0.20: 12/4; mulai 0.35 tidak ada itemset |
| 4.8 Apriori vs FP-Growth | `min_support = 0.15`, rata-rata 30 kali pengulangan | Kedua algoritma menghasilkan 17 itemset yang sama |

Tiga aturan yang diinterpretasikan pada 4.6:

| Aturan | Confidence | Lift | Rekomendasi |
|---|---|---|---|
| Mentega, Susu → Roti | 100% | 3,333 | Bundling "Paket Sarapan Hemat" |
| Sabun → Shampo | 60% | 2,769 | Tata berdekatan di area toiletries |
| Kecap → Mie Instan | 75% | 2,500 | Dasar fitur "pelanggan yang membeli ini juga membeli" |

## Notebook 2: Tugas Praktikum (Tugas 1 sampai 4)

File: `Praktikum_ARM_2_Tugas.ipynb`

### Tugas 1 — Eksplorasi ambang batas

- **1.1** `min_support` 0.10 vs 0.15: menurunkan ambang menambah 2 itemset (19 vs 17) dan 2 aturan (20 vs 18). Aturan baru (Gula ↔ Kopi) hanya didukung 8 dari 60 transaksi sehingga lebih rentan terhadap kebetulan.
- **1.2** `min_confidence = 0.7`: 7 dari 18 aturan lolos, 11 gugur. Contoh pola bermakna yang ikut gugur: Sabun ↔ Shampo (Lift 2,769).
- **1.3 dan 1.4** Sweep `min_confidence`: jumlah aturan 18 (0.5), 14 (0.6), 7 (0.7), 3 (0.8), 1 (0.9 dan 1.0). Satu-satunya yang bertahan sampai confidence 1,0 adalah Mentega, Susu → Roti. Ambang sebaiknya tidak dipilih hanya dari confidence, tetapi juga Lift dan jumlah transaksi pendukung.

### Tugas 2 — Analisis mendalam aturan

Dua aturan yang belum dibahas dihitung Support, Confidence, Lift, Leverage, dan Conviction:

| Aturan | Support | Confidence | Lift | Leverage | Conviction |
|---|---|---|---|---|---|
| Mie Instan → Telur | 0,167 | 0,556 | 2,222 | 0,092 | 1,688 |
| Susu → Roti | 0,217 | 0,684 | 2,281 | 0,122 | 2,217 |

Lift kedua aturan hampir sama, tetapi Leverage dan Conviction menunjukkan Susu → Roti sedikit lebih kuat (sekitar 7 transaksi lebih banyak dari yang diharapkan jika independen, dibanding sekitar 5,5 untuk Mie Instan → Telur). Nilai dicek terhadap keluaran `mlxtend`.

### Tugas 3 — Apriori vs FP-Growth pada `min_support = 0.05`

- Itemset naik dari 17 menjadi **42** (12 tunggal, 21 pasangan, 8 tiga-item, 1 empat-item), dan kedua algoritma tetap menghasilkan itemset yang **sama persis**.
- Pada dataset sekecil ini FP-Growth hanya sedikit lebih cepat (rasio sekitar 1,5x, selisih sekitar 0,5 sampai 0,6 ms) dan masih dalam rentang variasi pengukuran, jadi keunggulannya belum terlihat jelas.
- Uji tambahan dengan transaksi direplikasi (hingga 30.000 baris): pada implementasi `mlxtend` Apriori justru lebih cepat (sekitar 5 ms vs sekitar 59 ms untuk FP-Growth pada 30.000 transaksi), karena Apriori `mlxtend` memakai operasi vektor NumPy sedangkan FP-Growth membangun pohon dengan kode Python. Klaim bahwa FP-Growth jauh lebih cepat tidak dapat direproduksi di sini; hasil praktis bergantung pada pustaka dan karakteristik data.
- Angka waktu berubah sedikit di setiap eksekusi.

### Tugas 4 — Studi kasus mandiri: toko bangunan

Dataset: `TRANSAKSI_TOKO_BANGUNAN_GABUNGAN.csv` (50 transaksi, 9 produk; lihat bagian [Dataset](#dataset)). Data format long dikelompokkan per `ID Transaksi` menjadi keranjang (2 sampai 4 produk, rata-rata 2,86).

- **Parameter:** `min_support = 0.10` (minimal 5 transaksi) dan `min_confidence = 0.6`.
- **Alasan:** `0.05` menghasilkan 37 itemset dan 21 aturan dengan banyak pola yang hanya muncul 3 kali; `0.15` sampai `0.20` menyisakan 10 sampai 11 aturan dan membuang pola bermakna seperti Sambungan PVC → Besi Beton; `0.25` hanya menyisakan 2 aturan. Pada confidence 0.5 ada 19 aturan, 5 di antaranya hanya berconfidence 0,50 sampai 0,59.
- **Hasil:** **22 frequent itemset** (9 tunggal, 11 pasangan, 2 tiga-item) dan **14 aturan** dengan Lift 1,59 sampai 2,32.

Rekomendasi bisnis:

| Paket | Aturan utama | Lift | Confidence |
|---|---|---|---|
| Bahan Bangunan Dasar | Pasir, Semen → Batu Bata | 2,315 | 83,3% |
| | Batu Bata, Semen → Pasir | 2,137 | 76,9% |
| | Batu Bata, Pasir → Semen | 2,083 | 83,3% |
| Instalasi Pipa dan Besi | Sambungan PVC → Besi Beton | 2,121 | 63,6% |
| Pengecatan | Cat Kayu, Cat Tembok → Kuas | 1,984 | 83,3% |
| | Cat Tembok → Kuas | 1,786 | 75,0% |

Dari 10 transaksi yang memuat Pasir, Semen, dan Batu Bata sekaligus, 7 berasal dari data tambahan yang disusun sendiri. Kuatnya aturan bahan bangunan dasar sebagian mencerminkan pola yang dibangun, sehingga rekomendasi ini perlu divalidasi dengan data penjualan nyata.

## Kesimpulan

- Association rule mining dengan Apriori mampu menemukan pola belanja yang bermakna: pada dataset toko kelontong diperoleh 17 itemset dan 18 aturan (`min_support = 0.15`, `min_confidence = 0.5`) dengan Lift maksimum 3,333.
- Dataset 1 (daftar item) dan Dataset 2 (one-hot) memberikan hasil yang sama, sehingga encoding terverifikasi.
- `min_support` dan `min_confidence` menentukan keseimbangan antara banyaknya pola (risiko noise) dan keandalan pola (risiko kehilangan pola bermakna).
- Support, Confidence, Lift, Leverage, dan Conviction sebaiknya dibaca bersama, bukan salah satu saja.
- Apriori dan FP-Growth menghasilkan itemset yang sama; perbedaan waktunya bergantung pada implementasi pustaka dan ukuran data.
- Aturan asosiasi menunjukkan keterkaitan, bukan sebab-akibat; rekomendasi bisnis sebaiknya divalidasi lebih lanjut (misalnya lewat uji A/B).

## Daftar Pustaka

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *Proceedings of the 20th International Conference on Very Large Data Bases (VLDB)*, 487–499.
- Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques* (3rd ed.). Morgan Kaufmann.
- Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *ACM SIGMOD Record*, 29(2), 1–12.
- mlxtend Developers. (2024). *Frequent pattern mining — mlxtend documentation*. https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/
- Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. *Journal of Open Source Software*, 3(24), 638.
- revgaming. (n.d.). *Data transaksi toko bangunan* [Data set]. Kaggle. https://www.kaggle.com/datasets/revgaming/data-transaksi-toko-bangunan
- Tan, P.-N., Steinbach, M., & Kumar, V. (2019). *Introduction to data mining* (2nd ed.). Pearson.
