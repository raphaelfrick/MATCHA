
<p align="center">
  
![MATCH-A: An Artificial Benchmark Dataset for Robust Image Matching in the GenAI Era](https://github.com/user-attachments/assets/7c726f36-f51d-4874-b1dd-642a1adaa613)


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

## Dataset Overview

MATCH-A is a large-scale synthetic benchmark designed to advance research in image matching and retrieval under challenging real-world conditions. The dataset addresses the growing need for robust systems that can reliably link edited or manipulated images back to their authentic sources, even when multiple transformations have been applied. With over 217,000 gallery images and 22,000 query images spanning 34 manipulation types, MATCH-A provides a comprehensive testbed for evaluating image matching algorithms in the GenAI era.

The dataset consists of two complementary components: a gallery of authentic images and a set of query images that have been subjected to various manipulations. Each query is either connected to its source image (connected query) or has no match in the gallery (orphan query), enabling evaluation of both retrieval accuracy and appropriate abstention.

<div align="center">

| Split | Gallery Images | Queries | Connected | Orphan | Avg Manipulations |
|-------|----------------|---------|-----------|--------|-------------------|
| train | 140,896 | 17,987 | 16,906 | 1,081 | 2.84 |
|  val | 148,876 | 1,995 | 1,884 | 111 | 2.78 |
|  test | 148,890 | 2,500 | 2,181 | 319 | 2.71 |
|  **total** | **217,473** | **22,482** | **20,971** | **1,511** | **2.82** |

</div>

The train and validation splits focus on common manipulation types, while the test set introduces several unseen effects to assess generalization capabilities. This design ensures that models trained on MATCH-A can be evaluated on their ability to handle both familiar and novel transformation types.


### Key Features

- **Scale**: 200,000+ realistic synthetic images
- **Diversity**: 10 scene categories (portraits, landscapes, products, food, architecture, animals, sports, travel, textures, candid photos)
- **Manipulations**: 34 parameterized transformation types
- **Privacy**: Fully synthetic - no real people or locations
- **Standardization**: Predefined train/val/test splits

## Dataset Structure

The dataset is organized into two complementary parts stored as Hugging Face dataset configurations.

### Queries Dataset (`queries/`)

Contains manipulated query images paired with metadata. Each query is linked to its authentic source (connected) or has no match (orphan).

```
queries/
├── train/
├── val/
└── test/
```

**Columns:**
- `query`: The manipulated query image (PIL Image)
- `query_id`: Query image ID (extracted from filename, e.g., "I_115733e71330")
- `positive_id`: Positive image ID (extracted from filename, e.g., "I_115733e71330")
- `has_positive`: 1 if connected to a source, 0 if orphan
- `query_transforms`: Pipe-separated list of applied transformations (e.g., "DehazeEffect|ColorPopEffect")
- `quality`: Quality score (0-100), -1 if missing
- `split`: Dataset split ('train', 'val', or 'test')

### Trusted DB Dataset (`trusted_db/`)

Contains authentic images that serve as the trusted corpus for matching. Images are deduplicated (each unique image appears once) with split membership indicated by boolean flags.

**Columns:**
- `image`: The authentic image (PIL Image)
- `image_id`: Unique image identifier (extracted from filename, e.g., "I_115733e71330")
- `prompt`: The prompt used to generate this image (from prompts_lookup.csv)
- `train`: Boolean - True if image belongs to train split
- `val`: Boolean - True if image belongs to val split
- `test`: Boolean - True if image belongs to test split

## Usage

The dataset is available on Hugging Face and can be loaded using the `datasets` library.

```python
from datasets import load_dataset

# Load queries dataset
queries_ds = load_dataset("rfsit/MATCH-A", name="queries", split="train")

# Load trusted_db dataset (deduplicated - all unique images in one split)
trusted_db_ds = load_dataset("rfsit/MATCH-A", name="trusted_db")

# Filter trusted_db by split using boolean flags
train_trusted_db = trusted_db_ds.filter(lambda x: x["train"])
val_trusted_db = trusted_db_ds.filter(lambda x: x["val"])
test_trusted_db = trusted_db_ds.filter(lambda x: x["test"])

# Access a sample
sample = queries_ds[0]
print(sample["query"])  # Query image (PIL Image)
print(sample["query_id"])  # Query image ID (e.g., "I_115733e71330")
print(sample["positive_id"])  # Positive image ID (e.g., "I_115733e71330")
print(sample["has_positive"])  # 1 if connected, 0 if orphan
print(sample["query_transforms"])  # List of transformations applied
print(sample["quality"])  # Quality score

# Access trusted_db sample with prompt
trusted_db_sample = trusted_db_ds[0]
print(trusted_db_sample["image"])  # Trusted DB image (PIL Image)
print(trusted_db_sample["image_id"])  # Image ID (e.g., "I_115733e71330")
print(trusted_db_sample["prompt"])  # Prompt used to generate this image
print(trusted_db_sample["train"])  # True if in train split
```

## Data Generation Pipeline

The dataset is generated through a four-stage pipeline designed to create realistic, diverse, and challenging image pairs.

1. **Prompt Engineering**: Qwen3-30B-A3B generates photorealistic prompts across 10 scene categories including portraits, landscapes, products, food, architecture, animals, sports, travel, textures, and candid photography.

2. **Image Synthesis**: FLUX.1-Krea-dev creates authentic gallery images from the prompts. Each image uses random seeds, varying guidance scales (3.5-6.0), and curated aspect ratios.

3. **Augmentation**: Query images are created by applying 1-5 random transforms from the 34 parameterized filters. Additionally, 200 human-edited queries improve realism and test generalization to real-world edits.

4. **Filtering**: Samples where manipulations degraded image quality beyond recognition are removed to preserve dataset quality.

**Models Used:**
- Prompts: Qwen3-30B-A3B-Instruct-2507
- Text-to-image: FLUX.1-Krea-dev (+ quantized variant for efficiency)
- Segmentation: SAM3 for mask-based edits
- Inpainting: FLUX.1-Fill-dev for object removal/insertion
- Style transfer: FLUX.1-Kontext for image-to-image transformations
- Text generation: Llama 3.2 1B for meme/text overlays

## Filter Categories

MATCH-A covers 34 parameterized transformation types, organized into 11 categories. Each manipulation can be applied at multiple strength levels to create varying degrees of difficulty.

<div align="center">

| Category | Examples |
|----------|----------|
| **Color/Tone** | brightness, contrast, saturation, vibrance, temperature, highlights/shadows, fade, split-toning |
| **Detail/Texture** | sharpen, clarity, film grain |
| **Atmosphere** | dehaze, vignette |
| **Blur/Motion** | Gaussian, tilt-shift, motion/zoom blur |
| **Distortion** | lens distortion/fisheye, chromatic aberration |
| **Palette** | black & white, duotone, posterize, color pop |
| **Stylize** | halftone, mosaic, pixelate |
| **Glow/Lighting** | bloom, lens flare/light leaks |
| **Framing** | frames/borders, double exposure |
| **Overlays** | text overlays/object overlays |
| **GenAI Edits** | random object inpaint, outpaint/refill, sky replacement, style transfer |

</div>

> **Note**: Some effects (Gaussian blur, text/object overlay, style transfer) appear only in the test set to evaluate generalization to unseen manipulation types.

## Evaluation Protocol

MATCH-A evaluates two complementary abilities: retrieving matches when they exist, and abstaining when they don't.

### Metrics

**Retrieval Quality (connected queries):**

- **Hit@k**: Measures the fraction of connected queries whose top-k results contain the true match. Higher is better.

<div display: block; text-align: center;>

 

$$ Hit@k = \frac{1}{N_{connected}} \sum_{i=1}^{N_{connected}} \mathbb{1}[rank(q_i) \leq k] $$

 

</div>

- **MRR (Mean Reciprocal Rank)**: Rewards placing the correct image as high as possible, aligning with fast verification.

<div display: block; text-align: center;>

$$ MRR = \frac{1}{N_{connected}} \sum_{i=1}^{N_{connected}} \frac{1}{rank(q_i)} $$

</div>

**Abstention Quality (orphan queries):**

- **FPRorph(τ)**: Fraction of orphans for which any candidate is returned. Lower is safer.

<div display: block; text-align: center;>

$$ FPR_{orph}(\tau) = \frac{1}{N_{orphan}} \sum_{i=1}^{N_{orphan}} \mathbb{1}[\max_j s(q_i, g_j) > \tau] $$

</div>

- **TNRorph(τ)**: Fraction of orphans correctly yielding no candidate.

<div display: block; text-align: center;>

$$ TNR_{orph}(\tau) = \frac{1}{N_{orphan}} \sum_{i=1}^{N_{orphan}} \mathbb{1}[\max_j s(q_i, g_j) \leq \tau] $$

</div>

- **AUROC/AUPRC**: Measures how well the maximum similarity score separates connected from orphan queries.

<div display: block; text-align: center;>

$$ AUROC = \int_0^1 TPR(FPR^{-1}(x)) dx $$

$$ AUPRC = \int_0^1 Precision(Recall^{-1}(x)) dx $$

</div>

### Baseline Results

We evaluated three representative vision backbones with frozen encoders and lightweight projection heads:

**Training Losses:**
- ResNet-50: Triplet Loss
- CLIP: InfoNCE Loss
- DINOv2: NT-Xent Loss

<div align="center">

| Metric | ResNet-50 | DINOv2 | CLIP |
|--------|-----------|--------|------|
| Hit@1 | 0.6983 | **0.9221** | 0.8482 |
| Hit@5 | 0.7685 | **0.9477** | 0.9175 |
| Hit@10 | 0.7946 | **0.9578** | 0.9358 |
| MRR | 0.7321 | **0.9338** | 0.8793 |
| FPRorph(τ) | 1.0000 | **0.9815** | 0.9940 |
| AUROC | 0.5331 | **0.5494** | 0.5396 |

</div>

**Key Findings:**
- DINOv2 achieves the strongest retrieval performance (Hit@1 = 0.9221)
- Abstention remains challenging across all models (high FPR_orph)
- Image-to-image synthesis is the hardest manipulation type
- Subtle filters (highlight/shadow, split-toning, brightness) have minimal impact

## Use Cases

MATCH-A supports a range of trust-oriented tasks in media forensics and information retrieval:

- **Trust-oriented image retrieval**: Link edited posts to verified originals and abstain when no credible source exists, enabling safer fact-checking workflows.

- **Deduplication**: Merge near-duplicates across large corpora to identify re-uploads and derivative content.

- **Provenance recovery**: Reconstruct edit histories by identifying which transformations were applied to an image.

- **Manipulation attribution**: Classify the types of edits present in a query (e.g., detecting AI-generated content).

## Ethical Considerations

MATCH-A is designed for beneficial research in media integrity and trust-oriented retrieval. Users should be aware of the following:

- **Synthetic data**: All images are generated and contain no real people, events, or locations. Any resemblance is purely coincidental.

- **Unintended artifacts**: Generative diffusion models may occasionally include unintended elements such as brands, logos, or other real-world identifiers.

- **Intended use**: The dataset is intended for trust-oriented retrieval, content moderation, and academic research. It should not be used for biometric identification or surveillance.

- **Limitations**: The dataset does not cover all possible manipulation types and may not generalize to future editing techniques.

## License & Contact

All materials (including the dataset, models, code, and documentation) are provided "AS IS" and "AS AVAILABLE," without warranties of any kind, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, or completeness. Use is at your own risk. The authors and maintainers shall not be liable for any claim, damages, or other liability, whether in contract, tort, or otherwise, arising from, out of, or in connection with the materials or their use; no support, guarantees, or updates are promised.

- **License**: Creative Commons Attribution Non Commercial Share Alike; Images are governed by [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
- **Authors**: Raphael Antonius Frick, Martin Steinebach (Fraunhofer SIT | ATHENE Center)
- **Repository**: https://github.com/raphaelfrick/MATCHA
- **Issues**: https://github.com/raphaelfrick/MATCHA/issues


## Contributing

We’d love to see your results or models. Join the discussion in the community tab of HuggingFace or contribute via an issue or pull request on GitHub.
