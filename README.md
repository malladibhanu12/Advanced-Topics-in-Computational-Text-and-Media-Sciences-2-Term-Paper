# Evaluating the Robustness of Explainable AI in CNNs and Vision Transformers under Small, Diagnosis-Preserving Image Perturbations

This project compares ResNet18 and ViT-Tiny on HAM10000, with a primary focus on the stability of their explanation heatmaps under small changes to the input image. ResNet18 is explained using Grad-CAM, while ViT-Tiny is explained using attention rollout.

The main research question is: **How much do explanation heatmaps change under controlled perturbations, particularly when the model's predicted diagnosis remains unchanged?**

## Experimental workflow

1. Load HAM10000 metadata and check that the image files are available.
2. Split the dataset by lesion ID into training, validation, and test sets to prevent lesion overlap.
3. Examine class distributions and calculate class weights from the training split.
4. Train both models using weighted cross-entropy and deterministic preprocessing.
5. Evaluate classification performance on the original test images.
6. Apply controlled perturbations only during test-time evaluation.
7. Compare prediction stability and spatially aligned explanation heatmaps.
8. Summarize results by perturbation and diagnostic class, including prediction-stable cases.

## Dataset

HAM10000 contains 10,015 dermoscopic images across seven diagnostic classes:

| Code | Diagnostic class |
| --- | --- |
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

Download the dataset from [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). The dataset is not included in this repository.

The split reserves 15% of lesions for testing and then 15% of the remaining lesions for validation. These correspond to approximately 72.25% training, 12.75% validation, and 15% testing at the lesion level. Image proportions can differ because a lesion may have multiple images. Assertions check that lesion IDs do not overlap between splits.

## Setup

Install Python and create an environment for the project, then install the dependencies:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `1763945_Term_Paper.ipynb` and run the cells in order. Update `META_PATH`, `IMG_DIR1`, and `IMG_DIR2` to match the locations of the downloaded metadata and extracted image folders.

The expected dataset files are:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

Training uses CUDA when available and otherwise runs on the CPU. For GPU use, install a PyTorch build compatible with your CUDA environment. Internet access is needed for the initial download of pretrained model weights.

## Models and training

| Setting | Value |
| --- | --- |
| Models | ResNet18 and ViT-Tiny (`vit_tiny_patch16_224`) |
| Initialization | Pretrained ImageNet weights |
| Input resolution | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 5 per model |
| Batch size | 32 |
| Loss | Class-weighted cross-entropy |
| Checkpoint selection | Highest validation macro F1 |
| Split seed | 42 |

For class `c`, the loss weight is calculated as `N / (K × n_c)`, where `N` is the number of training images, `K` is the number of classes, and `n_c` is the number of training images in that class. Both models use the same weight calculation. Weighted sampling is not combined with weighted loss.

Images are resized, converted to tensors, and normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Rotations, flips, brightness changes, and blur are not used as training augmentation in this experiment.

## Evaluation

### Baseline classification

The untouched test set is evaluated using accuracy, balanced accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion matrices, and one-vs-rest ROC-AUC.

These metrics establish whether the models classify lesions correctly. They are supporting measurements, not substitutes for explanation robustness evaluation.

### Controlled perturbations

The test-time transformations are:

- Rotation by −5° and +5°.
- Horizontal flip.
- Brightness factors of 0.9 and 1.1.
- Gaussian blur with kernel size 5 and sigma 1.0.

Prediction robustness measures agreement between original and perturbed predictions, per-class and macro agreement, and probability changes.

### Explanation robustness

Grad-CAM highlights regions contributing to a selected ResNet18 class score. During paired comparisons, the original predicted class is retained as the target. Attention rollout combines attention across ViT blocks and is class-agnostic.

Heatmaps from flipped and rotated images are mapped back to the original coordinates before comparison. Brightness and blur do not require spatial realignment. Rotation realignment is approximate because interpolation and cropping can lose information.

Explanation stability is evaluated using:

- Heatmap correlation: similarity in the spatial variation of heatmap values.
- Cosine similarity: similarity between flattened heatmaps.
- Top-20% salient-region intersection over union (IoU): overlap between the most salient regions.

Heatmap visualizations support the quantitative comparisons. Prediction-stable cases are analyzed separately to identify explanation changes that occur without a change in the predicted class.

## Scope of the current results

The saved notebook contains baseline and prediction-robustness results for the full test set. Its saved quantitative XAI evaluation covers a pilot subset of 20 images, all from the `bkl` class. Those outputs do not support full-test or seven-class conclusions about explanation stability.

For a full XAI evaluation, set `XAI_MAX_SAMPLES = None` and rerun the explanation-robustness section. This stage can be computationally expensive because explanations are generated for original and perturbed images.

The split seed controls data partitioning; it does not by itself make all training operations deterministic. Package versions in `requirements.txt` are currently unpinned.

## Limitations and responsible use

This study uses one dataset, two models, and different explanation methods. Therefore, measured differences cannot be attributed solely to architecture. Grad-CAM has limited spatial resolution, and attention rollout is not a class-specific attribution method.

Stable heatmaps do not automatically demonstrate faithful explanations or clinically correct localization. The experiment does not include dermatologist validation or lesion-mask localization assessment.

These models are research prototypes and must not be used as diagnostic devices or as a replacement for clinical assessment.
