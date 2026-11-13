# Conjunctivitis Detection and Subtype Classification with a Truncated VGG-16

**Author:** Esha Sabir, University of Sahiwal · Research code, not a medical device

## Abstract

This repository contains two Google Colab notebooks that fine-tune a truncated, ImageNet-pretrained VGG-16 on external eye photographs. `binary.ipynb` separates conjunctivitis from healthy eyes (573 images, 2 classes). `Multi_classification.ipynb` separates five classes: allergic, bacterial and viral conjunctivitis, uveitis and healthy eyes (769 images). On a single random held-out split, the saved notebook outputs show 58/58 correct test predictions for the binary model and 76/77 (accuracy 0.987) for the five-class model. The test sets are small, the split is not stratified or grouped by patient or source, and no external validation was done, so these figures are early, single-run results and not evidence of clinical performance.

## Problem and clinical motivation

Conjunctivitis ("pink eye") is common, contagious and usually diagnosed by visual examination. Bacterial, viral and allergic forms are treated differently but can look alike, and laboratory tests are rarely used in primary care. Pakistan had a large viral conjunctivitis outbreak in 2023 that closed schools in Punjab. A screening aid that works from an ordinary eye photograph could help triage, but only if it is validated on clinical data. This project is a first step: it tests whether a pretrained CNN can separate these classes on publicly collected images.

## Dataset

The image folders are **not included** in this repository. The notebooks read them from Google Drive (`/content/drive/My Drive/binaryDataset` and `/content/drive/My Drive/multidataset`).

| Task | Classes (folder names) | Images | Source of the counts |
|---|---|---|---|
| Binary | `conjunctivitis`, `healthy eye` | 573 total | `binary.ipynb`, cell 13 output |
| Five-class | `allergic conjunctivitis` 130, `Uveitis` 150, `bacterial conjuntivities` 150, `healthy eye` 209, `viral conjuntivities` 130 | 769 total | `Multi_classification.ipynb`, cell 6 output |

Per-class counts for the binary set were not printed in the saved notebook. In its test split, 41 images are conjunctivitis and 17 are healthy.

**Sources and licence.** The images were collected from public online datasets. The accompanying draft manuscript cites two of them: S. Ali, *Conjunctivitis dataset*, and Kondwani, *Eye disease dataset* (both accessed January 2025). The images are not redistributed here. Anyone reproducing this work must obtain them from the original sources under those sources' terms.

Image sizes vary (for example 194×259 and 165×306 pixels); some healthy-eye images are greyscale and are converted to RGB.

## Method

**Preprocessing.** Each image is resized to 224×224×3. Pixel values are passed to the network as raw 0–255 values: there is no rescaling and no data augmentation in the training code. Labels are integer-encoded with scikit-learn `LabelEncoder` (alphabetical order) and one-hot encoded.

**Split.** `train_test_split` with `random_state=42`, first 70/30 and then the 30% split 2:1, giving approximately 70% training, 20% validation and 10% test. The split is random and not stratified.

| Task | Train | Validation | Test |
|---|---|---|---|
| Binary | 401 | 114 | 58 |
| Five-class | 538 | 154 | 77 |

**Architecture.** VGG-16 with ImageNet weights (`include_top=False`), cut after `block4_conv3` (Keras layer index 13, output 28×28×512). All layers remain trainable.

| | Binary | Five-class |
|---|---|---|
| Backbone | VGG-16 up to `block4_conv3` | VGG-16 up to `block4_conv3` |
| Head | GlobalAveragePooling2D → Dense 512 (ReLU) → Dropout 0.7 → Dense 256 (ReLU) → Dropout 0.5 → Dense 2 (softmax) | GlobalAveragePooling2D → Dense 256 (ReLU) → Dropout 0.5 → Dense 5 (softmax) |
| Parameters (all trainable) | 8,029,762 | 7,767,877 |

## Training setup

- Optimiser: Adam, learning rate 1e-4; loss: categorical cross-entropy
- Learning-rate schedule: constant for epochs 0–9, then `1e-4 · exp(-0.1 · (epoch − 10))`
- Early stopping on validation loss, patience 3, best weights restored
- Up to 20 epochs. The binary run stopped after 15 epochs; the five-class run completed all 20.
- Class weights: inverse class frequency (binary: `n / (2·count)`; five-class: scikit-learn `compute_class_weight('balanced')`)
- Hardware and software: Google Colab GPU. The five-class notebook prints TensorFlow 2.18.0, Keras 3.8.0, NumPy 2.0.2 and OpenCV 4.11.0. The binary notebook's log format shows it ran under an older Keras 2 / TensorFlow 2.x runtime; its exact versions were not recorded.

## Results

All numbers below are copied from the saved outputs of the notebooks in this repository (one training run each, evaluated on the held-out test split). Precision, recall and F1 are weighted averages.

| Task | Test images | Accuracy | Precision | Recall | F1 | MCC | Cohen's κ |
|---|---|---|---|---|---|---|---|
| Binary | 58 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Five-class | 77 | 0.987 | 0.988 | 0.987 | 0.987 | 0.984 | 0.983 |

**Binary confusion matrix** (rows = true, columns = predicted; 0 = conjunctivitis, 1 = healthy eye):

```
[[41  0]
 [ 0 17]]
```

**Five-class confusion matrix** (0 = Uveitis, 1 = allergic, 2 = bacterial, 3 = healthy, 4 = viral; the order comes from `LabelEncoder`, which sorts the capitalised `Uveitis` first):

```
[[21  0  0  0  0]
 [ 0 12  0  0  0]
 [ 0  0 14  0  1]
 [ 0  0  0 19  0]
 [ 0  0  0  0 10]]
```

The single error is one bacterial conjunctivitis image predicted as viral. Training and validation curves and the confusion-matrix heat maps are rendered inside each notebook.

With 58 and 77 test images, one more or one fewer error changes accuracy by 1.3–1.7 percentage points. These results should be read as "the model fits this collection well", not as an estimate of real-world accuracy.

## How to reproduce

The notebooks were written for Google Colab with Google Drive mounted.

1. Open a notebook in Colab with the badge at its top, or upload it, and select a GPU runtime.
2. Download the source datasets and arrange them in Google Drive as:

   ```
   My Drive/
   ├── binaryDataset/
   │   ├── conjunctivitis/
   │   └── healthy eye/
   └── multidataset/
       ├── allergic conjunctivitis/
       ├── bacterial conjuntivities/
       ├── healthy eye/
       ├── Uveitis/
       └── viral conjuntivities/
   ```

   The folder names must match exactly, including the existing spellings, because they become the class labels.
3. Create `My Drive/EyeSpy_Model/` (the notebooks save the trained models there).
4. Run the cells from top to bottom. Skip cell 2 of `Multi_classification.ipynb` (it installs from a `requirement.txt` on the author's Drive) and install from this repository instead:

   ```bash
   pip install -r requirements.txt
   ```

To run locally instead of in Colab, install the requirements, remove the `google.colab` cells and set `dataset_path` to your local folder. Exact numbers will vary between runs because GPU training is not fully deterministic and no global seed is set.

## Repository structure

```
.
├── binary.ipynb                # conjunctivitis vs healthy: data loading, training, evaluation
├── Multi_classification.ipynb  # five-class model, evaluation and a Gradio demo
├── requirements.txt            # Python dependencies
├── CITATION.cff                # how to cite this repository
└── LICENSE                     # MIT (code only; the datasets keep their own licences)
```

## Limitations

- **Small, single-split evaluation.** One random split, 58 and 77 test images, no cross-validation and no confidence intervals.
- **Possible leakage and shortcut learning.** Images come from several public collections. The split is not grouped by patient or by source, and near-duplicate images were not removed, so the model may partly learn which collection an image came from (lighting, framing, resolution, greyscale healthy images) rather than the disease. Perfect test scores on a small set are a reason for caution.
- **No external or clinical validation.** Labels come from the public datasets and were not confirmed by an ophthalmologist for this study.
- **Preprocessing mismatch in the demo.** Training uses raw 0–255 pixels, but the Gradio function in `Multi_classification.ipynb` divides by 255, and its `class_names` list is not in the `LabelEncoder` order shown above. The demo's predictions are therefore not reliable as written. The notebook's last cells are an unfinished investigation of this issue.
- **Notebook hygiene.** Some cells in `binary.ipynb` carry outputs from an earlier version (cells 5–7), and cell 11 refers to a variable that is not defined in that notebook. Cell 10 overwrites the images in the Drive folder with resized copies.

## Ethical note

This is research and educational code. It is **not** a medical device and must not be used to diagnose, triage or treat anyone. Eye photographs are health data: anyone extending this work with new images needs consent and the relevant ethics approval.

## Related

- Thesis repository with the methodology figure and the earlier run's figures: <https://github.com/EshaRana17/Categorising-conjunctivitis-FYP-Thesis>

## Citation

If you use this code, please cite the repository (see `CITATION.cff`):

```bibtex
@software{sabir_conjunctivitis_2025,
  author = {Sabir, Esha},
  title  = {Conjunctivitis Detection and Subtype Classification with a Truncated VGG-16},
  year   = {2025},
  url    = {https://github.com/EshaRana17/Conjunctivitis}
}
```

## Licence

Code: MIT, see [LICENSE](LICENSE). Datasets are not included and remain under their original licences.

## Contact

Esha Sabir · GitHub [@EshaRana17](https://github.com/EshaRana17) · [LinkedIn](https://www.linkedin.com/in/esha-sabir-3b269a220/)
