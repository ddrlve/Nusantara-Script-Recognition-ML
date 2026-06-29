# Model Artifacts

This folder contains the files required by the Streamlit inference pipeline.

## Download Link

Download the complete model artifacts from Google Drive:

[https://drive.google.com/drive/folders/1zx3ggOan2yVCtfAO-IDO60Ec_hcjhfuG?usp=sharing](https://drive.google.com/drive/folders/1zx3ggOan2yVCtfAO-IDO60Ec_hcjhfuG?usp=sharing)

Place the downloaded files inside this `models/` directory.

## Required Files

| File | Approximate Size | Purpose |
| --- | ---: | --- |
| `scaler.pkl` | 150 KB | StandardScaler used to normalize handcrafted feature vectors before PCA |
| `pca_model.pkl` | 60 MB | PCA transformer that reduces 6,382 raw features to 2,479 features while retaining 98% variance |
| `guarded_model.pkl` | 193 MB | Main guarded ensemble classifier used for prediction |
| `class_names.npy` | 1 KB | Class label list for the 64 script-character classes |
| `inference_config.json` | 1 KB | Inference metadata, model metrics, and guardrail threshold configuration |

## Expected Structure

```text
models/
|-- scaler.pkl
|-- pca_model.pkl
|-- guarded_model.pkl
|-- class_names.npy
`-- inference_config.json
```

## Notes

- `pca_model.pkl` and `guarded_model.pkl` are large artifacts and may not be included directly in the repository.
- The Streamlit app expects these files to be available before inference can run successfully.
- If the app fails during startup or prediction, check that every required artifact exists in this folder.
