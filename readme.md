# On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2511.19641-b31b1b.svg)](https://arxiv.org/abs/2511.19641)


This repository provides the official implementation of the paper: **[On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction](https://arxiv.org/abs/2511.19641)**.

## Introduction
We investigate whether vision-language foundation models can enhance undersampled MRI reconstruction. Our approach leverages high-level semantic embeddings from pretrained vision-language foundation models (specifically [Janus](https://huggingface.co/deepseek-ai/Janus-Pro-1B)) to guide the reconstruction process through contrastive learning. This aligns the reconstructed image representations with a target semantic distribution, ensuring consistency with high-level perceptual cues. The proposed objective works with various deep learning-based reconstruction methods and can flexibly incorporate semantic priors from multimodal sources. We evaluated reconstruction results guided by priors derived from either image-only or image-language auxiliary information.

![Figure1.jpg](Figure1.jpg)


## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Method Overview](#method-overview)
- [Demo Data](#demo-data)
- [Related Resources](#related-resources)
- [Contact](#contact)



## Project Structure

```
Foundation-Model-MRI-Reconstruction/
├── image_language_demo.py          # Image-language embedding guided reconstruction
├── INR_demo.py                     # Implicit Neural Representation based reconstruction
├── Unet_demo.py                    # UNet-based reconstruction
├── Unrolled_demo.py                # Unrolled network reconstruction
├── model.py                        # Model definitions (Unrolled, SIREN, CG solver)
├── loss_function.py                # Contrastive and reconstruction loss functions
├── utils.py                        # Utility functions for data processing
├── feature_extraction_image.py     # Extract image embeddings from foundation model
├── feature_extraction_image_language.py  # Extract image-language embeddings
├── demo_data.mat                   # Demo k-space data
├── unet/                           # UNet architecture
│   ├── unet_model.py
│   ├── unet_parts.py
│   └── pre_trained_weights/        # Pre-trained UNet weights
├── Janus/                          # Janus foundation model
├── image_examples/                 # Example images for embedding extraction
└── prior_embeddings_image_language/  # Pre-computed embeddings
```

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- CUDA-compatible GPU

### Dependencies

```bash
# Clone the repository
git clone https://github.com/I3Tlab/Foundation-Model-MRI-Reconstruction.git
cd Foundation-Model-MRI-Reconstruction

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers scipy numpy h5py scikit-image scikit-learn
pip install sigpy tqdm tensorboard matplotlib umap-learn einops

# Install Janus (foundation model)
cd Janus
pip install -e .
cd ..

# Install minLoRA for efficient model adaptation
pip install minlora  # or from: https://github.com/changjonathanc/minLoRA
```

### Download Janus-Pro Weights

Download the pretrained Janus-Pro-1B model from [HuggingFace](https://huggingface.co/deepseek-ai/Janus-Pro-1B):

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/Janus-Pro-1B", trust_remote_code=True)
```

Or manually download and update the `foundation_model_path` in the demo scripts.

## Usage

### 1. Feature Extraction

Before reconstruction, extract prior embeddings from auxiliary images:

#### Image Embeddings
```bash
python feature_extraction_image.py
```

#### Image-Language Embeddings
```bash
python feature_extraction_image_language.py
```

This generates embeddings stored in:
- `prior_embeddings/` - Image embeddings at multiple feature levels
- `prior_embeddings_image_language/` - Image-language embeddings with UMAP visualization

### 2. MRI Reconstruction

We provide four reconstruction approaches:

#### Image-Language Guided Reconstruction (Recommended)
Combines vision and language understanding for semantic-aware reconstruction:
```bash
python image_language_demo.py
```

#### Implicit Neural Representation (INR)
Uses continuous coordinate-based neural network:
```bash
python INR_demo.py
```

#### UNet-based Reconstruction
Standard UNet with foundation model guidance:
```bash
python Unet_demo.py
```

#### Unrolled Network Reconstruction
Physics-informed unrolled architecture:
```bash
python Unrolled_demo.py
```

### 3. Results

Reconstruction results are saved in `reconstruction_results/`:
- `pred_results_*.mat` - Reconstructed images
- `model/` - Model checkpoints
- `log/` - TensorBoard logs

Monitor training with TensorBoard:
```bash
tensorboard --logdir=reconstruction_results/*/log
```

## Configuration

Key parameters in demo scripts:

```python
# Training settings
epochs = 200              # Number of training epochs
lr = 1e-3                 # Learning rate
accR = 4                  # Acceleration rate (undersampling factor)
niters = 8                # Number of unrolling iterations (for unrolled networks)

# Paths (update these for your environment)
foundation_model_path = "/path/to/Janus-Pro-1B"
feat_path = "/path/to/prior_embeddings"
```

## Method Overview

### Contrastive Learning Framework

The method uses contrastive learning to leverage foundation model embeddings:

1. **Prior Embedding Extraction**: Extract features from auxiliary high-quality (positive) and undersampled/degraded (negative) images using the foundation model's vision encoder

2. **Reconstruction Network Training**: During self-supervised training, the reconstructed image is passed through the same vision encoder

3. **Contrastive Loss**: The reconstruction embeddings are pushed toward positive embeddings and away from negative embeddings:

```python
# Multi-level contrastive loss
con_loss = 0.01 * criterion(features[0], pos_feat[:,0], neg_feat[:,0], W1, mu1) + \
           0.5  * criterion(features[12], pos_feat[:,1], neg_feat[:,1], W2, mu2) + \
           1.0  * criterion(features[23], pos_feat[:,2], neg_feat[:,2], W3, mu3)
```

### Image-Language Embedding

For enhanced semantic guidance, we combine vision and language:

```python
prompt = 'Determine whether this image is high-quality or low-quality.'
# Image + prompt → Language model → Image-language embedding
```

This allows the model to leverage the foundation model's understanding of image quality for better reconstruction guidance.


## Demo Data

The repository includes `demo_data.mat` containing multi-coil k-space data for testing. The data structure:
- Shape: `(Nsli, Nchl, Nrd, Npe)` - (slices, coils, readout, phase encoding)

## Related Resources

- [Janus-Pro Model](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
- [fastMRI Dataset](https://fastmri.org/)
- [minLoRA](https://github.com/changjonathanc/minLoRA)


## Contact

For questions or issues, please open a GitHub issue or contact the authors.

[Intelligent Imaging Innovation and Translation Lab](https://liulab.mgh.harvard.edu/) [[github]](https://github.com/I3Tlab) at the Athinoula A. Martinos Center of Massachusetts General Hospital and Harvard Medical School
* Ruimin Feng (rfeng3@mgh.harvard.edu)
* Fang Liu (fliu12@mgh.harvard.edu)

149 13th Street, Suite 2301
Charlestown, Massachusetts 02129, USA