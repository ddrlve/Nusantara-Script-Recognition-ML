# Dataset

This folder is used to store the image dataset for Nusantara Script Recognition.

## Download Source

Download the dataset from Mendeley Data:

[https://data.mendeley.com/datasets/vfj32bpjsf/1](https://data.mendeley.com/datasets/vfj32bpjsf/1)

## Expected Folder Structure

After downloading and extracting the dataset, place the script image folders in this directory:

```text
dataset/
├── Bali/
├── Jawa/
└── Sunda/
```

Each folder should contain the handwritten character images for the corresponding Indonesian local script.

## Notes

- The dataset images are required for training and experimentation.
- The deployed Streamlit demo uses prepared model artifacts, so the dataset is not required only for running inference.
- Keep large raw dataset files out of Git unless the repository is configured with a suitable large-file storage solution.
