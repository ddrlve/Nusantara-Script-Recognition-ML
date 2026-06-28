# Model Artifacts

Download semua file dan letakkan di folder `models/`:  
**Link**: https://drive.google.com/drive/folders/1zx3ggOan2yVCtfAO-IDO60Ec_hcjhfuG?usp=sharing

---

| File | Ukuran | Kegunaan |
|------|--------|----------|
| `scaler.pkl` | ~150 KB | Normalisasi vektor fitur sebelum PCA |
| `pca_model.pkl` | ~60 MB | Reduksi dimensi 6.382 → 2.479 (98% variance) |
| `guarded_model.pkl` | ~193 MB | Model klasifikasi utama (ensemble CatBoost + LightGBM) |
| `class_names.npy` | ~1 KB | 64 label kelas dalam format `Script_Karakter` |
| `inference_config.json` | ~1 KB | Konfigurasi threshold guardrail & metrik evaluasi |
