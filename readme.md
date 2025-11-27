# On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2511.19641-b31b1b.svg)](https://arxiv.org/abs/2511.19641)


This repository provides the official implementation of the paper: **[On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction](https://arxiv.org/abs/2511.19641)**.

## Introduction
We investigate whether vision-language foundation models can enhance undersampled MRI reconstruction. Our approach leverages high-level semantic embeddings from pretrained vision-language foundation models (specifically [Janus](https://huggingface.co/deepseek-ai/Janus-Pro-1B)) to guide the reconstruction process through contrastive learning. This aligns the reconstructed image embedding with a target semantic distribution, ensuring consistency with high-level perceptual cues. The proposed objective works with various deep learning-based reconstruction methods and can flexibly incorporate semantic priors from multimodal sources. We evaluated reconstruction results guided by prior embeddings derived from either image-only or image-language auxiliary information.

![Figure1.jpg](Figure1.jpg)


## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Related Resources](#related-resources)
- [Contact](#contact)



## Project Structure

```
Foundation-Model-MRI-Reconstruction/
├── image_language_demo.py          # Image-language embedding-guided reconstruction
├── INR_demo.py                     # Implicit Neural Representation (INR)-based reconstruction guided by image-only embeddings
├── Unet_demo.py                    # UNet-based reconstruction guided by image-only embeddings
├── Unrolled_demo.py                # Unrolled network reconstruction guided by image-only embeddings
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
├── image_examples/                 # Example images for image-language embedding extraction
└── prior_embeddings_image_language/  # Pre-computed image-language embeddings
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

# Install minLoRA
Install minLoRA from the link: https://github.com/cchangcs/minLoRA

```

### Download Janus-Pro Weights

Download the pretrained Janus-Pro-1B model from [HuggingFace](https://huggingface.co/deepseek-ai/Janus-Pro-1B), and update the `foundation_model_path` in the demo scripts.

### Download FastMRI Data

The FastMRI dataset is required for extracting image embeddings. To access the dataset:

1. **Register and Request Access**: Visit the [FastMRI website](https://fastmri.org/) and complete the data access request form.

2. **Download the Dataset**: Once approved, download the knee or brain MRI data. For this project, we primarily use:
   - `knee_multicoil_train`
   - `knee_multicoil_val`

3. **Organize the Data**: Place the downloaded data in your preferred directory and update the data paths in the demo scripts accordingly.


For quick testing, we provide:

- `demo_data.mat`: A multi-coil k-space slice for demo reconstruction
- `image_examples/`: Example images used to generate image-language embeddings
- `prior_embeddings_image_language/`: Pre-generated image-language embeddings 


## Usage

> **Important**: Before running any scripts, update the following paths in the code to match your environment:
> ```python
> foundation_model_path = "/path/to/Janus-Pro-1B"
> feat_path = "/path/to/prior_embeddings"
> ```

### 1. Feature Extraction

Before reconstruction, extract prior embeddings from auxiliary images.

#### Image Embeddings

> **Note**: You must download the FastMRI dataset before running this script, as the raw data files are too large to include in this repository. Please update the data paths in the script to point to your local data directory.

```bash
python feature_extraction_image.py
```

This generates embeddings stored in `prior_embeddings/` with UMAP visualization.

#### Image-Language Embeddings

We provide example images in `image_examples/` for extracting image-language embeddings.

```bash
python feature_extraction_image_language.py
```

This generates embeddings stored in `prior_embeddings_image_language/` with UMAP visualization.

### 2. MRI Reconstruction

We provide four reconstruction approaches optimized by data consistency and contrastive loss functions:


#### U-Net-based Reconstruction
Learns a direct transformation from undersampled inputs to reconstructed images:
```bash
python Unet_demo.py
```

#### Unrolled Network Reconstruction
Unrolls a variable-splitting iterative reconstruction algorithm into a sequence of learnable stages. Each stage alternates between a U-Net and an explicit data-consistency step solved using conjugate gradient descent:
```bash
python Unrolled_demo.py
```

#### Implicit Neural Representation (INR)
Represents the MR image as a continuous function of spatial coordinates. The function is parameterized by an MLP, which takes spatial coordinates as input and predicts the corresponding image intensities:
```bash
python INR_demo.py
```

#### Image-Language Guided Reconstruction
Combines vision and language understanding for semantic-aware reconstruction:
```bash
python image_language_demo.py
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