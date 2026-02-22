
<p align="center">

<img width="1022" height="510" alt="MATCH-A: An Artificial Benchmark Dataset for Robust Image Matching in the GenAI Era" src="https://github.com/user-attachments/assets/93054020-4068-4c61-a74a-380220537a66" />

[![Static Badge](https://img.shields.io/badge/Dataset-MATCH--A%20Dataset-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/rfsit/MATCH-A)
[![Static Badge](https://img.shields.io/badge/GitHub-Code_Repository-blue?style=flat&logo=github)](https://github.com/raphaelfrick/MATCHA)


</p>




**MATCH-A: An Artificial Benchmark Dataset for Robust Image Matching in the GenAI Era** is a large, fully synthetic dataset for training and evaluating robust image matching systems that link edited images to authentic sources.
  

## Why MATCH-A?

The rapid spread of powerful editing and generative tools has made it easy to produce convincing fabrications. Integrity and fact-checking workflows rely on image matching, yet both everyday edits and AI-driven transformations can heavily degrade recall.

**The Problem:**
- Social media platforms process billions of images daily
- Most images are edited with filters, crops, AI transformations
- Adversaries layer edits to evade duplicate detection
- Existing benchmarks don't cover modern GenAI manipulations

**The Solution:**
MATCH-A provides a privacy-preserving benchmark with 217,473 authentic gallery images and 22,482 query images, covering 34+ manipulation types including inpainting, outpainting, style transfer, and more.

## Overview

The MATCH-A Framework is a training and evaluation system for image retrieval and matching tasks. It leverages the **MATCH-A Dataset**, a large synthetic benchmark containing 217,473 authentic gallery images and 22,482 query images covering 34+ manipulation types.

### Framework Capabilities

- **Model Training**: Train various architectures (Triplet Network, Contrastive ViT, Contrastive CLIP)
- **Evaluation**: Evaluate model performance on the MATCH-A benchmark
- **Prediction**: Perform inference on query images against a gallery

## Quick Start

### 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Before training or evaluation, download and convert the MATCH-A dataset from HuggingFace:

```bash
python convert_matcha_hf.py --output_dir ../local_matcha_dataset
```

This creates the following structure:
```
local_matcha_dataset/
├── reference_db/              # Gallery/reference images (PNG)
├── queries/                   # Query images (JPG)
│   ├── train/
│   ├── val/
│   └── test/
├── data_splits.csv            # Combined metadata
├── gallery_train.csv
├── gallery_test.csv
├── gallery_val.csv
├── queries_train.csv
├── queries_val.csv
└── queries_test.csv
```

### 3. Train a Model

```bash
python train.py --config configs/contrastive_clip.yaml --csv ../local_matcha_dataset/data_splits.csv
```

### 4. Evaluate the Model

```bash
python eval.py --config configs/contrastive_clip.yaml --csv ../local_matcha_dataset/data_splits.csv --checkpoint outputs/best_model.pt
```

### 5. Run Predictions

```bash
python predict.py --checkpoint outputs/best_model.pt --query_image path/to/query.jpg --gallery_csv ../local_matcha_dataset/gallery_test.csv
```


## Framework Components

| Script | Purpose |
|--------|---------|
| [`train.py`](train.py) | Train models on the MATCH-A dataset |
| [`eval.py`](eval.py) | Evaluate trained models on test set |
| [`predict.py`](predict.py) | Run inference on query images |
| [`convert_matcha_hf.py`](convert_matcha_hf.py) | Download and convert dataset from HuggingFace |

---

## Training: `train.py`

### Currently Supported Models

| Model | Config File |
|-------|-------------|
| Triplet Network | `configs/triplet_net.yaml` |
| Contrastive ViT | `configs/contrastive_vit.yaml` |
| Contrastive CLIP | `configs/contrastive_clip.yaml` |

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | `str` | `configs/contrastive_clip.yaml` | Path to YAML configuration file |
| `--csv` | `str` | `local_matcha_dataset/data_splits.csv` | Path to CSV with split columns |
| `--model` | `str` | `None` | Model name (overrides config) |
| `--epochs` | `int` | `None` | Number of epochs (overrides config) |
| `--batch_size` | `int` | `None` | Batch size (overrides config) |
| `--lr` | `float` | `None` | Learning rate (overrides config) |
| `--device` | `str` | `None` | Device to use (cuda or cpu) |
| `--seed` | `int` | `42` | Random seed |
| `--resume` | `str` | `None` | Path to checkpoint to resume from |
| `--output_dir` | `str` | `outputs` | Output directory for checkpoints and logs |

### Examples

```bash
# Train a ResNet-50 model using triplet loss
python train.py --config configs/triplet_net.yaml --csv ../local_matcha_dataset/data_splits.csv

# Train a ViT with contrastive loss
python train.py --config configs/contrastive_vit.yaml --csv ../local_matcha_dataset/data_splits.csv

# Resume from checkpoint
python train.py --config configs/contrastive_clip.yaml --resume outputs/checkpoint_epoch_10.pt

# Custom batch size and learning rate
python train.py --config configs/triplet_net.yaml --batch_size 64 --lr 0.0001
```

### Output

- **Checkpoints**: `outputs/best_model.pt`, `outputs/checkpoint_epoch_N.pt`
- **Logs**: `outputs/train.log`

---

## Evaluation: `eval.py`

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | `str` | `configs/contrastive_clip.yaml` | Path to YAML configuration file |
| `--csv` | `str` | `local_matcha_dataset/data_splits.csv` | Path to CSV with split columns |
| `--gallery_csv` | `str` | `local_matcha_dataset/gallery_test.csv` | Path to gallery CSV file |
| `--model` | `str` | `None` | Model name (overrides config) |
| `--checkpoint` | `str` | `None` | Path to checkpoint to evaluate |
| `--batch_size` | `int` | `None` | Batch size (overrides config) |
| `--device` | `str` | `None` | Device to use (cuda or cpu) |
| `--seed` | `int` | `42` | Random seed |
| `--output_dir` | `str` | `outputs` | Output directory for logs |

### Examples

```bash
# Evaluate a trained model
python eval.py --config configs/contrastive_clip.yaml --checkpoint outputs/best_model.pt --csv ../local_matcha_dataset/data_splits.csv

# With custom batch size
python eval.py --config configs/contrastive_clip.yaml --checkpoint outputs/best_model.pt --batch_size 64
```

### Output

- **Logs**: `outputs/eval.log`
- **Metrics**: Hit@1, MRR, FPR_orph, TNR_orph, AUROC, AUPRC


## Prediction: `predict.py`

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | `str` | `required` | Path to model checkpoint |
| `--config` | `str` | `None` | Path to config file (optional if in checkpoint) |
| `--model` | `str` | `None` | Model name (optional if in checkpoint) |
| `--query_image` | `str` | `None` | Path to single query image |
| `--query_dir` | `str` | `None` | Directory containing query images |
| `--query_csv` | `str` | `None` | CSV file with query image paths |
| `--gallery_csv` | `str` | `required` | Path to CSV file with gallery images |
| `--gallery_split` | `str` | `test` | Split to use for gallery |
| `--output` | `str` | `predictions.json` | Path to output file |
| `--output_format` | `str` | `json` | Output format (json or csv) |
| `--top_k` | `int` | `5` | Number of top matches to retrieve |
| `--batch_size` | `int` | `32` | Batch size for processing |
| `--device` | `str` | `None` | Device to use (cuda or cpu) |

### Examples

```bash
# Predict on a single image
python predict.py --checkpoint outputs/best_model.pt --query_image path/to/query.jpg --gallery_csv ../local_matcha_dataset/gallery_test.csv

# Predict on a directory of images
python predict.py --checkpoint outputs/best_model.pt --query_dir path/to/queries --gallery_csv ../local_matcha_dataset/gallery_test.csv

# Predict with top-10 matches and CSV output
python predict.py --checkpoint outputs/best_model.pt --query_image path/to/query.jpg --gallery_csv ../local_matcha_dataset/gallery_test.csv --top_k 10 --output_format csv --output predictions.csv
```

### Output

- **Results**: `predictions.json` (or specified output path)
- Contains query paths and top-k matches with scores


## Dataset Conversion: `convert_matcha_hf.py`

Downloads the MATCH-A dataset from HuggingFace and converts it to the framework format.

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | `str` | `./local_matcha_dataset` | Output directory for the converted dataset |
| `--num_workers` | `int` | `8` | Number of worker threads |
| `--download_only` | `bool` | `False` | Download dataset without processing |
| `--hf_dataset` | `str` | `rfsit/MATCH-A` | HuggingFace dataset name |

### Example

```bash
python convert_matcha_hf.py --output_dir ../local_matcha_dataset --num_workers 8
```



## Configuration Files

| Config File | Model | Key Settings |
|-------------|-------|--------------|
| `triplet_net.yaml` | Triplet Network | `model: triplet_net` |
| `contrastive_vit.yaml` | Contrastive ViT | `model: contrastive_vit` |
| `contrastive_clip.yaml` | Contrastive CLIP | `model: contrastive_clip` |



## License & Contact

All materials (including the dataset, models, code, and documentation) are provided "AS IS" and "AS AVAILABLE," without warranties of any kind, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, or completeness. Use is at your own risk. The authors and maintainers shall not be liable for any claim, damages, or other liability, whether in contract, tort, or otherwise, arising from, out of, or in connection with the materials or their use; no support, guarantees, or updates are promised.

- **License**: Creative Commons Attribution Non Commercial Share Alike; Images are governed by [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
- **Authors**: Raphael Antonius Frick, Martin Steinebach (Fraunhofer SIT | ATHENE Center)
- **Repository**: https://github.com/raphaelfrick/MATCHA
- **Issues**: https://github.com/raphaelfrick/MATCHA/issues


## Contributing

We’d love to see your results or models. Join the discussion in the community tab of HuggingFace or contribute via an issue or pull request on GitHub.
