# SignBridge Phase 1 MVP

Real-time Azerbaijani Sign Language word recognition from webcam using the real dataset at `dataset/AzSLD_Words_100`.

## Dataset

The dataset is discovered dynamically. Labels are inferred from first-level folders and samples are all supported media files found below each label folder.

Current inspection found:

- 100 labels
- 7,248 samples
- File format: `.mp4`
- Organization: class-per-folder

Regenerate the report:

```powershell
python src/inspect_dataset.py --dataset-dir dataset/AzSLD_Words_100
```

Outputs:

- `reports/dataset_report.md`
- `reports/dataset_report.json`
- `reports/dataset_manifest.csv`

## Setup

Use Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MediaPipe 0.10.35 on Python 3.12 uses the Tasks API. Download the hand landmark model into `models/hand_landmarker.task`:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

## Feature Extraction

Extract MediaPipe Hands landmarks from the real videos and save NumPy features:

```powershell
python src/extract_features.py --dataset-dir dataset/AzSLD_Words_100 --sequence-length 30
```

Outputs:

- `data/features/X.npy`
- `data/features/y.npy`
- `data/features/splits.npz`
- `data/features/labels.json`
- `data/features/feature_manifest.csv`
- `data/features/feature_config.json`

Each sample is represented as a fixed sequence of 30 frames. Each frame has 126 values: 2 hands x 21 landmarks x xyz coordinates.

## Training

Train the LSTM classifier and evaluate on the held-out test split:

```powershell
python src/train.py --epochs 60 --batch-size 32
```

Outputs:

- `models/sign_model.h5`
- `models/labels.json`
- `models/metrics.json`

The metrics file includes accuracy, macro precision, macro recall, macro F1, and the confusion matrix.

## Webcam Inference

Run real-time recognition:

```powershell
python src/infer.py --camera-index 0 --confidence-threshold 0.5
```

The webcam window shows the predicted Azerbaijani sign word and confidence score. Press `q` to quit.

## Notes

- No fake or generated training data is used.
- Dataset paths are configurable through CLI arguments.
- The code handles Azerbaijani labels with UTF-8 JSON and CSV outputs.
- Install dependencies before feature extraction, training, or webcam inference.
- The dataset contains videos recorded at 35 FPS with both frontal and side-view angles for enhanced robustness.

## Dataset, Funding & Citation

This software utilizes the **Azerbaijani Sign Language Dataset (AzSLD)**, developed by researchers at **ADA University**.

### Funding Acknowledgement

The data collection was funded by the project *"Strengthening Data Analytics Research and Training Capacity through Establishment of dual Master of Science in Computer Science and Master of Science in Data Analytics (MSCS/DA) degree program at ADA University"*, supported by **BP** and the **Ministry of Education of the Republic of Azerbaijan**.

### License

The dataset is officially licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

### Academic Citation

If you use this dataset or software in your work, please cite the following official paper:

> Alishzade, N., Hasanov, J. (2025). AzSLD: Azerbaijani sign language dataset for fingerspelling, word, and sentence translation with baseline software, *Data in Brief*, Volume 58, 2025, 111230, ISSN 2352-3409, https://doi.org/10.1016/j.dib.2024.111230.

Official Repository: [ADA-SITE-JML/azsl_dataloader](https://github.com/ADA-SITE-JML/azsl_dataloader)
