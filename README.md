# Galaxy Photometric Redshift Prediction & Generative Modelling

A capstone project exploring deep learning approaches to photometric redshift estimation from multi-band galaxy images, ending with a Latent Diffusion Model (LDM) capable of generating physically consistent galaxy images conditioned on redshift and morphology.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Checkpoints](#checkpoints)
- [Reports](#reports)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Notebooks](#notebooks)

---

## Project Overview

This project uses the GalaxiesML dataset, 5-band (g, r, i, z, y) images of galaxies at 127×127 pixels, to show a progression of photometric redshift models, from classical baselines to physically informed deep fusion networks. A Latent Diffusion Model is then trained to generate realistic galaxy images conditioned on spectroscopic redshift and morphological parameters.

**Key components:**

- Classical baselines: Ridge Regression, KNN, Random Forest, Gradient Boosting, MLP
- CNN baselines: AlexNet, VGG16, Vision Transformer (ViT), ResNet101 with autoencoder
- Fusion models: Cross-Attention Fusion (image + morphology), Physically Informed Late Fusion (band-aware EfficientNet + Transformer)
- Generative model: Latent Diffusion Model with classifier-free guidance, DDPM/DDIM sampling, and physics consistency loss

---

## Project Structure

```
Capstone/
│
├── data/                         # Galaxy image datasets (not included — see Data section)
│   ├── 5x127x127_training_with_morphology.hdf5
│   ├── 5x127x127_validation_with_morphology.hdf5
│   └── 5x127x127_testing_with_morphology.hdf5
│
├── models/                       # Trained model checkpoints (not included — see Checkpoints section)
│   ├── alexnet_best.pth
│   ├── vgg16_full_best.pth
│   ├── vit_redshift_regressor_best.pth
│   ├── resnet_best.pth / resnet_autoencoder.pt
│   ├── fusion_crossattn_best_exp2.pt
│   ├── physfusion_best.pth
│   ├── galaxy_vae_best.pt
│   ├── galaxy_ldm_best.pt
│   ├── galaxy_physics_predictor.pt
│   └── ...                       # Training metrics CSVs, prediction arrays, sklearn pickles
│
├── notebooks/                    # Jupyter notebooks (execution order matches main.py)
│   ├── EDA.ipynb
│   ├── AlexNet.ipynb
│   ├── VGG16.ipynb
│   ├── ViT.ipynb
│   ├── ResNet101.ipynb
│   ├── Models.ipynb
│   ├── FusionModelCrossAttn.ipynb
│   ├── PhysicallyInformedLateFusion.ipynb
│   ├── Model_Comparison.ipynb
│   └── GalaxyLDM.ipynb
│
├── reports/                      # Generated figures and evaluation plots
│
├── scripts/
│   └── compute_band_stats.py     # Computes per-band mean/std from training data
│
├── src/
│   ├── data_loader.py            # HDF5 dataset classes and DataLoader factories
│   ├── metrics.py                # Redshift evaluation metrics (NMAD, outlier rate, bias)
│   ├── utils.py                  # Seeding, device detection, plotting utilities
│   └── models/
│       ├── alexnet.py            # AlexNet regressor
│       ├── vgg16.py              # VGG16 regressor
│       ├── vit.py                # Vision Transformer regressor
│       ├── resnet101.py          # ResNet101 + autoencoder
│       ├── crossattn.py          # Cross-Attention Fusion model
│       ├── phys_informed.py      # Physically Informed Late Fusion model
│       ├── simple_cnn.py         # Lightweight CNN baseline
│       └── ldm.py                # Full LDM stack: VAE, UNet, DDPM/DDIM, conditioning
│
├── main.py                       # Reproducibility runner — executes all notebooks in order
└── requirements.txt              # Python dependencies with pinned versions
```

---

## Data

The data is not included in this repository due to its large size. Download the **127×127 pixel** splits from the GalaxiesML dataset:

**Dataset page:** https://datalab.astro.ucla.edu/galaxiesml.html

Download the three HDF5 files for the 127×127 resolution (training, validation, testing) and place them in the `data/` folder at the project root.

**macOS / Linux:**
```bash
mkdir -p data
curl -o data/5x127x127_training_with_morphology.hdf5 "https://zenodo.org/records/11117528/files/5x127x127_training_with_morphology.hdf5?download=1"
curl -o data/5x127x127_validation_with_morphology.hdf5 "https://zenodo.org/records/11117528/files/5x127x127_validation_with_morphology.hdf5?download=1"
curl -o data/5x127x127_testing_with_morphology.hdf5 "https://zenodo.org/records/11117528/files/5x127x127_testing_with_morphology.hdf5?download=1"
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri "https://zenodo.org/records/11117528/files/5x127x127_training_with_morphology.hdf5?download=1" -OutFile 5x127x127_training_with_morphology.hdf5
Invoke-WebRequest -Uri "https://zenodo.org/records/11117528/files/5x127x127_validation_with_morphology.hdf5?download=1" -OutFile 5x127x127_validation_with_morphology.hdf5
Invoke-WebRequest -Uri "https://zenodo.org/records/11117528/files/5x127x127_testing_with_morphology.hdf5?download=1" -OutFile 5x127x127_testing_with_morphology.hdf5
```

---

## Checkpoints

Trained model weights are not included in this repository. Download the full `models/` folder from Google Drive and place it at the project root:

**Google Drive:** https://drive.google.com/drive/folders/1Ngl_2AizhN-wfOtPiZwnneuttb8GASQV?usp=sharing

**macOS / Linux:**
```bash
# After downloading and unzipping:
mv models/ /path/to/Capstone/models/
```

**Windows:**
```powershell
# After downloading and unzipping:
Move-Item -Path models -Destination C:\path\to\Capstone\models
```

The `models/` folder must be present for the notebooks to run, as all training cells are guarded and load from checkpoints by default.

---

## Reports

The `reports/` folder is included in the project and contains all generated evaluation figures and plots. To regenerate them locally, run the full pipeline via `main.py` (see [Running the Project](#running-the-project)).

---

## Setup

Python 3.12 and Anaconda are recommended.

**1. Download the repository:**

```bash
cd Capstone
```

**2. Create and activate a conda environment:**

```bash
conda create -n capstone python=3.12.7
conda activate capstone
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Register the kernel with Jupyter:**

```bash
python -m ipykernel install --user --name capstone --display-name "Python (Capstone)"
```

---

## Running the Project

`main.py` executes all notebooks in the correct order using `nbconvert`. Outputs are saved back into the `.ipynb` files and a timestamped log is written to `reports/`.

**Run the full pipeline:**
```bash
python main.py
```

**Run a subset of notebooks:**
```bash
python main.py --only EDA AlexNet VGG16
```

**Skip specific notebooks:**
```bash
python main.py --skip GalaxyLDM
```

---

## Notebooks

The notebooks are designed to be run sequentially. Each loads model weights from checkpoints in `models/` rather than training from scratch.

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Exploratory data analysis: band distributions, redshift histograms, morphology correlations, etc. |
| `AlexNet.ipynb` | AlexNet-based photometric redshift regressor |
| `VGG16.ipynb` | VGG16-based photometric redshift regressor |
| `ViT.ipynb` | Vision Transformer regressor fine-tuned on galaxy images |
| `ResNet101.ipynb` | ResNet101 regressor with an additional autoencoder for feature regularization |
| `Models.ipynb` | Classical ML baselines: Ridge, KNN, Random Forest, Gradient Boosting, MLP |
| `FusionModelCrossAttn.ipynb` | Cross-attention fusion of image features and morphological tabular data |
| `PhysicallyInformedLateFusion.ipynb` | Band-aware EfficientNet + band-aware Transformer with physics-informed losses |
| `Model_Comparison.ipynb` | Centralised evaluation and visualisation across all redshift models |
| `GalaxyLDM.ipynb` | Latent Diffusion Model: VAE, UNet, DDPM/DDIM, classifier-free guidance, generation and evaluation |
