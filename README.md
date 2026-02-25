<p align="center">

<img width="1022" height="510" alt="MATCH-A: An Artificial Benchmark Dataset for Robust Image Matching in the GenAI Era" src="https://github.com/user-attachments/assets/93054020-4068-4c61-a74a-380220537a66" />

[![Static Badge](https://img.shields.io/badge/Dataset-MATCH--A%20Dataset-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/rfsit/MATCH-A)
[![Static Badge](https://img.shields.io/badge/GitHub-Code_Repository-blue?style=flat&logo=github)](https://github.com/raphaelfrick/MATCHA)

</p>


**MATCH-A: An Artificial Benchmark Dataset for Robust Image Matching in the GenAI Era** is a large, fully synthetic benchmark for training and evaluating robust image matching systems that link edited images to authentic sources.

This repository contains a modular framework for creating image-matching baselines with swappable encoders, projection heads, and loss functions.

## News

Latest updates to the repository:

- **2026-02-25**: Updated the framework to v1.1 to be more modular. Fixed training bugs. Added scripts for richer evaluation statistics and visualization of results.
- **2026-02-22**: Initial release.



## Why MATCH-A?

The rapid spread of powerful editing and generative tools has made it easy to produce convincing fabrications. Integrity and fact-checking workflows rely on image matching, yet both everyday edits and AI-driven transformations can heavily degrade recall.

**The Problem:**
- Social media platforms process billions of images daily
- Most images are edited with filters, crops, AI transformations
- Adversaries layer edits to evade duplicate detection
- Existing benchmarks don't cover modern GenAI manipulations

**The Solution:**
MATCH-A provides a privacy-preserving benchmark with 217,473 authentic gallery images and 22,482 query images covering 34+ manipulation types (inpainting, outpainting, style transfer, and more).

## Overview

The MATCH-A Framework is a training and evaluation system for image retrieval and matching tasks built around the [MATCH-A Dataset](https://huggingface.co/datasets/rfsit/MATCH-A).

Current Framework capabilities:
- Model training (Triplet ResNet, Contrastive ViT, Contrastive CLIP)
- Evaluation on the MATCH-A benchmark
- Prediction and retrieval against a gallery


## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Convert the Hugging Face [dataset](https://huggingface.co/datasets/rfsit/MATCH-A) into the local folder structure. Note: the dataset is gated, and access is granted individually:

```bash
python convert_matcha_hf.py --output_dir local_matcha_dataset
```

This creates `local_matcha_dataset/` with `reference_db/`, `queries/`, and split CSVs (including `data_splits.csv` and `gallery_*.csv`).

Train (example with `contrastive_vit.yaml`):

```bash
python train.py --config configs/contrastive_vit.yaml
```

Evaluate:

```bash
python eval.py --config configs/contrastive_vit.yaml --checkpoint outputs/contrastive_vit/best_model.pt
```

Predict:

```bash
python predict.py --checkpoint outputs/contrastive_vit/best_model.pt --gallery_csv local_matcha_dataset/gallery_test.csv --query_dir local_matcha_dataset/queries/test
```

Single-image prediction + visualization:

```bash
python predict.py --checkpoint outputs/contrastive_vit/best_model.pt --gallery_csv local_matcha_dataset/gallery_test.csv --query_image path/to/query.jpg --top_k 5 --visualize
```


## Framework

The MATCH-A framework is config-driven: choose an encoder, projector, and loss, then use the same pipeline for training, evaluation, and prediction. The `matcher` model composes the encoder + projector + loss, and `preprocess_in_dataset` controls whether resize/normalize happen in the dataset for better GPU utilization (default: true).

Core building blocks:
- Encoders: ResNet-50, DINOv2, CLIP
- Projectors: MLP or MLP+BN
- Losses: Triplet, InfoNCE

Framework Components:

| Script | Purpose |
|--------|---------|
| `train.py` | Train models on the MATCH-A dataset |
| `eval.py` | Evaluate trained models on the test split |
| `predict.py` | Run inference on query images |
| `convert_matcha_hf.py` | Download and convert the dataset from Hugging Face |

Example config:

```yaml
model: matcher
encoder: resnet50
encoder_type: torchvision
pretrained: true
projector: mlp
embedding_dim: 256
loss: infonce
temperature: 0.1
threshold: 0.5
batch_size: 32
num_workers: 4
image_size: 224
preprocess_in_dataset: true
```

## CLI Parameters (Most Important)

Train (`train.py`):
- `--config` path to config YAML (default: `configs/contrastive_clip.yaml`)
- `--csv` path to data_splits.csv (default: `local_matcha_dataset/data_splits.csv`)
- `--epochs`, `--batch_size`, `--lr` overrides
- `--device` `cuda` or `cpu`
- `--resume` checkpoint path
- `--output_dir` output folder (default: `outputs/<config_stem>`)

Eval (`eval.py`):
- `--config` path to config YAML (default: `configs/contrastive_clip.yaml`)
- `--csv` path to data_splits.csv (default: `local_matcha_dataset/data_splits.csv`)
- `--gallery_csv` path to gallery CSV (default: `local_matcha_dataset/gallery_test.csv`)
- `--checkpoint` checkpoint path
- `--batch_size` override
- `--device` `cuda` or `cpu`
- `--output_dir` output folder (default: `outputs/<config_stem>`)

Predict (`predict.py`):
- `--checkpoint` checkpoint path (required)
- `--query_image` or `--query_dir` or `--query_csv` (required: pick one)
- `--gallery_csv` gallery CSV path (required)
- `--output` output file path (default: `predictions.json`)
- `--top_k` number of matches (default: 5)
- `--device` `cuda` or `cpu`
- `--visualize` save query + top-k grid images


## Dataset

MATCH-A is a fully synthetic benchmark for robust image matching under heavy edits and GenAI transformations. Each split pairs a large gallery with queries; queries are labeled as connected (match exists in the gallery) or orphan (no match).

| Split | Gallery Images | Queries | Connected | Orphan | Avg Manipulations |
|-------|----------------|---------|-----------|--------|-------------------|
| train | 140,896 | 17,987 | 16,906 | 1,081 | 2.84 |
| val | 148,876 | 1,995 | 1,884 | 111 | 2.78 |
| test | 148,890 | 2,500 | 2,181 | 319 | 2.71 |
| **total** | **217,473** | **22,482** | **20,971** | **1,511** | **2.82** |

Dataset usage:

```python
from datasets import load_dataset

queries = load_dataset("rfsit/MATCH-A", name="queries", split="train")
trusted = load_dataset("rfsit/MATCH-A", name="trusted_db")
```

## Baseline Results

Baselines below are reported on the test split. Higher is better for Hit@k, MRR, and AUROC; lower is better for FPRorph(tau).

| Metric | ResNet-50 | DINOv2 | CLIP |
|--------|-----------|--------|------|
| Hit@1 | 0.6983 | **0.9221** | 0.8482 |
| Hit@5 | 0.7685 | **0.9477** | 0.9175 |
| Hit@10 | 0.7946 | **0.9578** | 0.9358 |
| MRR | 0.7321 | **0.9338** | 0.8793 |
| FPRorph(tau) | 1.0000 | **0.9815** | 0.9940 |
| AUROC | 0.5331 | **0.5494** | 0.5396 |


### Example Predictions

DINOv2 (Contrastive Loss):

<img width="4480" height="640" alt="I_42c65f4e8f072bda83fddaa4a2ca578c_top5" src="https://github.com/user-attachments/assets/71c1db92-a4ec-4c4f-91ab-b32562f14857" />
<img width="4480" height="640" alt="I_11d0a1b5161f32ca1b98c32decebc7a4_top5" src="https://github.com/user-attachments/assets/a0ce54b1-a702-422c-a2a7-6745f4eff515" />
<img width="4480" height="640" alt="I_7f2ccaa0d614c9d66dff11dc37101b4e_top5" src="https://github.com/user-attachments/assets/7423be82-f525-4c50-a97b-9b56a48bef2f" />

ResNet-50 (Triplet Loss):

<img width="4480" height="640" alt="I_42c65f4e8f072bda83fddaa4a2ca578c_top5" src="https://github.com/user-attachments/assets/6e3b64ca-4f96-46cf-bc50-51510c2ce515" />
<img width="4480" height="640" alt="I_11d0a1b5161f32ca1b98c32decebc7a4_top5" src="https://github.com/user-attachments/assets/a80fb90a-b6bc-4bc0-906c-97e710fffdce" />
<img width="4480" height="640" alt="I_7f2ccaa0d614c9d66dff11dc37101b4e_top5" src="https://github.com/user-attachments/assets/4e51944d-be01-424d-83f0-604cc22bdab4" />

ResNet-50 (Contrastive Loss):

<img width="4480" height="640" alt="I_42c65f4e8f072bda83fddaa4a2ca578c_top5" src="https://github.com/user-attachments/assets/ea7699c4-8774-4596-81c6-2acaefa71d83" />
<img width="4480" height="640" alt="I_11d0a1b5161f32ca1b98c32decebc7a4_top5" src="https://github.com/user-attachments/assets/74014b52-5160-4873-acd1-72912a359123" />
<img width="4480" height="640" alt="I_7f2ccaa0d614c9d66dff11dc37101b4e_top5" src="https://github.com/user-attachments/assets/589a8859-673c-4c31-8983-5c69bf5880b5" />


## License & Contact

All materials (including the dataset, models, code, and documentation) are provided "AS IS" and "AS AVAILABLE," without warranties of any kind, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, or completeness. Use is at your own risk. The authors and maintainers shall not be liable for any claim, damages, or other liability, whether in contract, tort, or otherwise, arising from, out of, or in connection with the materials or their use; no support, guarantees, or updates are promised.

- **License**: Creative Commons Attribution-NonCommercial-ShareAlike; Images are governed by [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
- **Authors**: Raphael Antonius Frick, Martin Steinebach (Fraunhofer SIT | ATHENE Center)
- **Repository**: https://github.com/raphaelfrick/MATCHA
- **Issues**: https://github.com/raphaelfrick/MATCHA/issues


## Contributing

We'd love to see your results or models. Join the discussion in the community tab of Hugging Face or contribute via an issue or pull request on GitHub.
