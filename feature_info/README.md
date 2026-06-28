# Feature Info

Folder ini berisi metadata dan referensi hasil ekstraksi fitur dari dataset aksara Nusantara (Bali, Jawa, Sunda) yang digunakan dalam proyek **Nusantara Script Recognition ML**.

---

## Daftar File

| File | Deskripsi |
|------|-----------|
| `feature_block_schema.csv` | Skema blok fitur — rentang kolom dan jumlah fitur per metode ekstraksi |
| `feature_column_detail.csv` | Pemetaan detail setiap kolom fitur ke blok dan indeks lokalnya |
| `feature_extraction_test_preview.csv` | Preview 300 sampel test set beserta seluruh nilai fitur yang telah diekstrak |

---

## Skema Fitur (`feature_block_schema.csv`)

Total fitur yang diekstrak per gambar: **6.382 fitur**, dibagi menjadi 5 blok:

| Blok | Kolom Mulai | Kolom Akhir | Jumlah Fitur | Keterangan |
|------|-------------|-------------|--------------|------------|
| `hog` | 0 | 5807 | 5.808 | Histogram of Oriented Gradients |
| `lbp` | 5808 | 5833 | 26 | Local Binary Pattern |
| `gabor` | 5834 | 5961 | 128 | Filter Gabor (tekstur frekuensi/orientasi) |
| `zoning_projection` | 5962 | 6361 | 400 | Proyeksi zona horizontal & vertikal |
| `shape_skeleton` | 6362 | 6381 | 20 | Fitur bentuk dan kerangka (skeleton) |

---

## Detail Kolom Fitur (`feature_column_detail.csv`)

File ini berisi **6.382 baris** (1 header + 6.382 fitur), dengan kolom:

| Kolom | Deskripsi |
|-------|-----------|
| `feature_col` | Nama kolom fitur (format: `feat_XXXX`) |
| `feature_block` | Blok asal fitur (`hog`, `lbp`, `gabor`, `zoning_projection`, `shape_skeleton`) |
| `local_index` | Indeks fitur di dalam bloknya masing-masing (dimulai dari 0) |

Contoh:
```
feat_0000 → hog, local_index=0
feat_5808 → lbp, local_index=0
feat_5834 → gabor, local_index=0
feat_5962 → zoning_projection, local_index=0
feat_6362 → shape_skeleton, local_index=0
```

---

## Preview Dataset (`feature_extraction_test_preview.csv`)

File ini berisi **300 sampel** dari split `test`, dengan struktur kolom:

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `split` | string | Selalu `test` |
| `filepath` | string | Path gambar di Kaggle (`/kaggle/input/...`) |
| `label_id` | int | ID kelas numerik |
| `label` | string | Label gabungan aksara dan karakter (contoh: `Bali_pa`) |
| `script` | string | Aksara: `Bali`, `Jawa`, atau `Sunda` |
| `character` | string | Nama karakter dalam aksara tersebut |
| `feat_0000` … `feat_6381` | float | Nilai 6.382 fitur hasil ekstraksi |

### Distribusi Kelas dalam Preview

Total **63 label unik** dari 3 aksara:

**Aksara Bali (18 karakter)**
`ba`, `ca`, `da`, `ga`, `ha`, `ja`, `ka`, `la`, `ma`, `na`, `nga`, `nya`, `pa`, `ra`, `sa`, `ta`, `wa`, `ya`

**Aksara Jawa (13 karakter dalam preview)**
`ba`, `ca`, `da`, `dha`, `ga`, `ha`, `ja`, `ka`, `la`, `ma`, `na`, `nga`, `nya`

**Aksara Sunda (32 karakter)**
`a`, `ba`, `ca`, `da`, `e`, `é`, `eu`, `fa`, `ga`, `ha`, `i`, `ja`, `ka`, `kha`, `la`, `ma`, `na`, `nga`, `nya`, `o`, `pa`, `qa`, `ra`, `sa`, `sya`, `ta`, `u`, `va`, `wa`, `xa`, `ya`, `za`

### Sumber Data
Dataset gambar berasal dari Kaggle:
`/kaggle/input/datasets/fadhlannurrachman/script-aksara/`

---

## Cara Membaca dengan Pandas

```python
import pandas as pd

# Load skema blok fitur
schema = pd.read_csv("feature_block_schema.csv")

# Load pemetaan kolom fitur
col_detail = pd.read_csv("feature_column_detail.csv")

# Load preview dataset (hati-hati: 6382 kolom fitur)
preview = pd.read_csv("feature_extraction_test_preview.csv")

# Ambil hanya kolom fitur dari satu blok, misal Gabor
gabor_cols = col_detail[col_detail["feature_block"] == "gabor"]["feature_col"].tolist()
gabor_features = preview[gabor_cols]
```
