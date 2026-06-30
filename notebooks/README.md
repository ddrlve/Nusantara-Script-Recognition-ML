# Notebooks — Comparative Study of Classical ML untuk Aksara Nusantara

**COMP6577001 — Machine Learning, Bina Nusantara University**

---

## Daftar Notebook

| Notebook | Deskripsi | Status |
|---|---|---|
| `experiment-notebook.ipynb` | Notebook eksperimen aktual — dieksekusi di Kaggle T4x2, menghasilkan model final yang di-deploy | **Hasil Resmi / Final** |
| `training-aksara-ml.ipynb` | Refaktor bersih dari notebook di atas — kode lebih rapi, penjelasan lengkap, satu konfigurasi linear | Versi dokumentasi |

---

## Kenapa Ada Dua Notebook?

`experiment-notebook.ipynb` adalah notebook yang dijalankan secara aktual di Kaggle dan menghasilkan model yang di-deploy ke Hugging Face Spaces (`ddrlvee/nusantara_script`). Notebook tersebut fungsional tapi sulit dibaca: menggunakan sistem 4-preset `if/elif`, kode repetitif, dan penjelasan yang minim.

`training-aksara-ml.ipynb` adalah refaktor bersih dari notebook di atas dengan tujuan keterbacaan dan dokumentasi. Pipeline-nya identik (data, split, fitur, seed, hyperparameter), hanya **struktur dan presentasi kode yang diperbaiki** — bukan algoritmanya.

---

## Kenapa Angka Metrik Berbeda?

Kedua notebook menjalankan **pipeline yang persis sama**. Perbedaan angka bukan disebabkan oleh perubahan kode, melainkan oleh **variasi antar-run GPU** yang tidak dapat dihindari:

### 1. GPU Non-Determinism

XGBoost (`device=cuda`) dan CatBoost (`task_type=GPU`) menggunakan operasi histogram paralel di GPU yang bersifat **tidak bit-reproducible** antar-run, walau `random_state=42` sudah diset. Ini perilaku umum pada boosting berbasis GPU.

### 2. Efek Cascading ke Ensemble

Karena skor tiap model bergeser ±0.01, grid-search pemilihan bobot ensemble menemukan kombinasi optimal yang berbeda:

| | `experiment-notebook.ipynb` | `training-aksara-ml.ipynb` |
|---|---|---|
| Best ensemble | lgbm + cat + two_stage_lgbm | xgb + lgbm + cat + two_stage_lgbm |
| Bobot | [0.125, 0.625, 0.25] | [0.182, 0.091, 0.455, 0.273] |

### 3. PCA Dimension Sedikit Berbeda

PCA dengan `n_components=0.98` menemukan jumlah komponen yang memenuhi 98% variance. Karena floating-point berbeda, threshold jatuh di titik berbeda: 2479 vs 2477. Perbedaan ini tidak signifikan secara praktis.

---

## Perbandingan Hasil Lengkap

| Metrik | `experiment-notebook.ipynb` | `training-aksara-ml.ipynb` |
|---|---|---|
| **Test Accuracy** | **0.9140** | 0.9220 |
| **Macro F1** | **0.9094** | 0.9087 |
| Val Macro F1 | 0.9096 | 0.9006 |
| Macro Precision | — | 0.9088 |
| Macro Recall | — | 0.9106 |
| Weighted F1 | — | 0.9215 |
| F1 Bali | 0.9313 | 0.9620 |
| F1 Jawa | 0.8736 | 0.8357 |
| F1 Sunda | 0.9127 | 0.9191 |
| PCA dimensi | 6382D → **2479D** | 6382D → 2477D |
| Train samples | 15,774 | 15,774 |
| Val samples | 2,401 | 2,401 |
| Test samples | 4,001 | 4,001 |
| Overfit verdict | RINGAN / SEDANG | RINGAN |

> **Catatan F1 per-script:** Perbedaan mencolok pada Jawa (0.8736 vs 0.8357) sebagian besar disebabkan oleh kelas `Jawa_pa` yang hanya memiliki **2 sampel test**. Satu prediksi benar/salah pada kelas ini menggeser rata-rata F1 Jawa sebesar ~0.05. Ini artefak ukuran sampel, bukan perbedaan kemampuan model.

---

## Perbandingan Struktur Kode

| Aspek | `experiment-notebook.ipynb` | `training-aksara-ml.ipynb` |
|---|---|---|
| Jumlah sel | 55 | 90 |
| Konfigurasi | 4-preset `if/elif` | 1 blok `CONFIG` linear |
| Penjelasan | Minim | `#### Interpretasi` setelah setiap sel kode |
| Visualisasi EDA | Terbatas | Distribusi kelas, galeri glyph, statistik ukuran gambar |
| Loss curve | Tidak ada | Ada (train vs val per iterasi) |
| Guardrail assembly | Tidak lengkap | Lengkap (`GuardedAksaraClassifier`) |
| Export CSV fitur | Ada | Ada (`notebooks/example_csv/`) |
| Style | Eksperimental | Beginner-friendly, linear |

---

## Hasil Model Final (Resmi)

Diambil dari `experiment-notebook.ipynb` — model yang di-deploy ke produksi.

```
Ensemble   : Weighted SoftVote
Komposisi  : LightGBM (bobot 0.125) + CatBoost (0.625) + Two-Stage LightGBM (0.25)
Accuracy   : 91.40%
Macro F1   : 90.94%
Val Macro F1: 90.96%

Per-script F1:
  Bali   : 93.13%
  Jawa   : 87.36%
  Sunda  : 91.27%

Overfit: Train Acc=1.00 | Val Acc=0.9225 | Test Acc=0.9140
         Gap Train-Test: 0.086 -> Verdict: RINGAN / SEDANG
```

Model artifacts disimpan di `models/`:
- `scaler.pkl` — StandardScaler
- `pca_model.pkl` — PCA (6382D → 2479D, 98% variance)
- `guarded_model.pkl` — GuardedAksaraClassifier (model final)
- `class_names.npy` — 64 label kelas
- `inference_config.json` — konfigurasi guardrail dan ensemble weights
