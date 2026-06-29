# Nusantara Script Recognition

Nusantara Script Recognition is a Streamlit application for classifying handwritten characters from Indonesian local scripts: Balinese, Javanese, and Sundanese. The project uses a Classical Machine Learning pipeline with handcrafted image features, dimensionality reduction, ensemble classifiers, and inference guardrails.

This project was developed for the COMP6577001 - Machine Learning course at Bina Nusantara University.

## Project Links

| Resource     | Link                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| Live Demo    | [Hugging Face Spaces](https://huggingface.co/spaces/ddrlvee/nusantara_script)                        |
| Presentation | [Canva PPT](https://canva.link/dg58uk6dioz5zlg)                                                      |
| Demo Video   | [YouTube](https://youtu.be/9QKtasruUsY?si=7nYltDBAybGO4MwC)                                          |
| Backup Files | [Google Drive](https://drive.google.com/drive/folders/1fhfxlPtg8OB9WwjVLdLvW_sTNRzHNXo_?usp=sharing) |

## Team Members

| Name                    | Student ID |
| ----------------------- | ---------- |
| Dian Rakhmawati Lestari | 2802539085 |
| Fadhlan Nur Rachman     | 2802491690 |
| Bintang Nur Fadhlillah  | 2802536083 |

## What the Application Does

The application accepts an image containing one handwritten script character and predicts:

- The script family: Balinese, Javanese, or Sundanese
- The character class
- The confidence score
- The top prediction candidates
- Whether the image should be rejected by guardrails because it is ambiguous, low confidence, or contains more than one detected object

The deployed app is available at [https://huggingface.co/spaces/ddrlvee/nusantara_script](https://huggingface.co/spaces/ddrlvee/nusantara_script).

## Model Summary

This project intentionally uses Classical Machine Learning instead of Deep Learning. The final model is a weighted soft-voting ensemble built from gradient boosting models.

| Component             | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| Input scripts         | Balinese, Javanese, Sundanese                                             |
| Number of classes     | 64                                                                        |
| Input image size      | 96 x 96 pixels                                                            |
| Raw feature dimension | 6,382                                                                     |
| PCA feature dimension | 2,479                                                                     |
| PCA variance retained | 98%                                                                       |
| Main model            | Weighted SoftVote ensemble                                                |
| Ensemble members      | LightGBM, CatBoost, Two-Stage LightGBM                                    |
| Guardrails            | Confidence threshold, prediction margin threshold, multi-object rejection |

## Pipeline

| Stage                 | Details                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Dataset preparation   | Class distribution analysis, class balancing, and augmentation up to 250 samples per class                   |
| Preprocessing         | Grayscale conversion, CLAHE, denoising, adaptive Otsu binarization, morphology, crop, padding, and centering |
| Feature extraction    | HOG, LBP, Gabor filters, zoning/projection features, and shape/skeleton features                             |
| Scaling and reduction | StandardScaler followed by PCA with 98% retained variance                                                    |
| Training              | XGBoost, LightGBM, CatBoost, and Two-Stage LightGBM with Optuna tuning                                       |
| Ensemble              | Weighted soft voting using CatBoost, Two-Stage LightGBM, and LightGBM                                        |
| Inference safety      | Rejects predictions with low confidence, small probability margin, or multiple large detected components     |

## Evaluation Results

| Metric       |  Value |
| ------------ | -----: |
| Accuracy     | 91.40% |
| Macro F1     | 90.94% |
| Balinese F1  | 93.13% |
| Sundanese F1 | 91.27% |
| Javanese F1  | 87.36% |

The train-validation gap is approximately 0.0002, indicating that the final model does not show meaningful overfitting under the selected evaluation setup.

## Repository Structure

```text
Nusantara-Script-Recognition-ML/
|-- app.py                         # Streamlit application and inference pipeline
|-- requirements.txt               # Python dependencies
|-- Dockerfile                     # Container configuration for deployment
|-- README.md                      # Main project documentation
|-- dataset/
|   `-- README.md                  # Dataset download and placement instructions
|-- docs/
|   |-- LINK.md                    # Project submission links
|   `-- Submit Final Project...docx
|-- feature_info/
|   |-- README.md                  # Feature metadata documentation
|   |-- feature_block_schema.csv
|   |-- feature_column_detail.csv
|   `-- feature_extraction_test_preview.csv
|-- models/
|   |-- README.md                  # Model artifact instructions
|   |-- class_names.npy
|   |-- scaler.pkl
|   `-- inference_config.json
|-- notebooks/
    |-- training-aksara-ml.ipynb   # Main training notebook
    |-- reference_notebook/
```

Large model artifacts such as `guarded_model.pkl` and `pca_model.pkl` may not be stored directly in this repository. See [models/README.md](models/README.md) for download instructions.

## Local Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Nusantara-Script-Recognition-ML
```

### 2. Create and Activate a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Model Artifacts

Place the required model files inside the `models/` directory:

```text
models/
|-- scaler.pkl
|-- pca_model.pkl
|-- guarded_model.pkl
|-- class_names.npy
|-- inference_config.json
```

If any large model files are missing, download them using the link in [models/README.md](models/README.md).

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Dataset

The dataset can be downloaded from [Mendeley Data](https://data.mendeley.com/datasets/vfj32bpjsf/1).

After downloading and extracting the dataset, place the image folders as follows:

```text
dataset/
|-- Bali/
|-- Jawa/
|-- Sunda/
```

For more details, see [dataset/README.md](dataset/README.md).

## Usage Guidelines

For best prediction quality:

- Upload an image containing exactly one handwritten character.
- Use a clean and bright background.
- Make sure the character is written with dark, clear strokes.
- Capture the image from a straight overhead angle.
- Avoid blurry images, multiple characters, full words, textured paper, or heavy shadows.

## Important Notes

- This project is focused on Classical Machine Learning and handcrafted feature engineering.
- The deployed demo runs on Hugging Face Spaces using Streamlit.
- Some files are stored externally because they are too large for normal repository storage.
- Backup materials are available in the project backup link listed above.

## Deployment Configuration

---

title: Nusantara Script
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
tags:

- streamlit
- machine-learning
- computer-vision
  short_description: Indonesian local handwritten script character recognition
  license: apache-2.0

---
