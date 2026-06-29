# Feature Information

This folder contains metadata and reference outputs for the handcrafted feature extraction process used in the Nusantara Script Recognition project.

The documented features were extracted from Balinese, Javanese, and Sundanese script images and are used by the Classical Machine Learning pipeline before scaling, PCA, and classification.

## Files

| File | Description |
| --- | --- |
| `feature_block_schema.csv` | Defines each feature block, its column range, and the number of features produced by each extraction method |
| `feature_column_detail.csv` | Maps every feature column to its source feature block and local block index |
| `feature_extraction_test_preview.csv` | Preview of 300 test-set samples with extracted feature values |

## Feature Block Schema

Each image is converted into 6,382 raw handcrafted features divided into five blocks.

| Feature Block | Start Column | End Column | Feature Count | Description |
| --- | ---: | ---: | ---: | --- |
| `hog` | 0 | 5807 | 5,808 | Histogram of Oriented Gradients features |
| `lbp` | 5808 | 5833 | 26 | Local Binary Pattern texture features |
| `gabor` | 5834 | 5961 | 128 | Gabor filter texture features across frequencies and orientations |
| `zoning_projection` | 5962 | 6361 | 400 | Horizontal and vertical zone projection features |
| `shape_skeleton` | 6362 | 6381 | 20 | Shape and skeleton-based morphology features |

## Column Detail File

`feature_column_detail.csv` contains one row for each of the 6,382 extracted features.

| Column | Description |
| --- | --- |
| `feature_col` | Feature column name using the format `feat_XXXX` |
| `feature_block` | Source block name, such as `hog`, `lbp`, `gabor`, `zoning_projection`, or `shape_skeleton` |
| `local_index` | Feature index within its own block, starting from 0 |

Example mapping:

```text
feat_0000 -> hog, local_index=0
feat_5808 -> lbp, local_index=0
feat_5834 -> gabor, local_index=0
feat_5962 -> zoning_projection, local_index=0
feat_6362 -> shape_skeleton, local_index=0
```

## Test Preview File

`feature_extraction_test_preview.csv` contains 300 samples from the test split.

| Column | Type | Description |
| --- | --- | --- |
| `split` | string | Dataset split name, always `test` in this preview |
| `filepath` | string | Original image path from the training environment |
| `label_id` | integer | Numeric class ID |
| `label` | string | Combined script and character label, for example `Bali_pa` |
| `script` | string | Script family: `Bali`, `Jawa`, or `Sunda` |
| `character` | string | Character name inside the script family |
| `feat_0000` to `feat_6381` | float | Extracted handcrafted feature values |

## Classes in the Preview

The preview contains 63 unique labels from three script families.

Balinese characters:

```text
ba, ca, da, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, wa, ya
```

Javanese characters:

```text
ba, ca, da, dha, ga, ha, ja, ka, la, ma, na, nga, nya
```

Sundanese characters:

```text
a, ba, ca, da, e, eu, fa, ga, ha, i, ja, ka, kha, la, ma, na, nga, nya,
o, pa, qa, ra, sa, sya, ta, u, va, wa, xa, ya, za
```

## Reading the Files with Pandas

```python
import pandas as pd

schema = pd.read_csv("feature_block_schema.csv")
column_detail = pd.read_csv("feature_column_detail.csv")
preview = pd.read_csv("feature_extraction_test_preview.csv")

gabor_columns = column_detail[
    column_detail["feature_block"] == "gabor"
]["feature_col"].tolist()

gabor_features = preview[gabor_columns]
```

## Notes

- The raw feature vector has 6,382 dimensions before scaling and PCA.
- The production inference pipeline reduces the feature vector to 2,479 PCA features.
- These files are documentation and inspection aids; the Streamlit app performs feature extraction directly from uploaded images.
