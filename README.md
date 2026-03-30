# 🔭 StarCLIP: Contrastive Learning for Stellar Spectra
*adapted from Julia Kim*


This repository contains the code for **StarCLIP**, a contrastive self-supervised learning framework that embeds observed APOGEE and *ab initio* JWST/NIRSpec spectra into a unified, physically meaningful latent space.

Our approach consists of training convolutional neural networks (CNNs) to recover fundamental stellar properties from single-modal spectroscopic data. These pre-trained CNNs are then used as encoders and aligned via contrastive loss. To simulate realistic NIRSpec observations, we construct semi-empirical, stochastic NIRSpec catalogs and embed them into the shared latent space using **MockStarCLIP**, a modified CLIP-based framework.

Both models enable seamless transfer to downstream tasks, including cosine similarity search and stellar property recovery.

<p align="center">
	<img width="694" alt="StarCLIP Overview" src="https://github.com/user-attachments/assets/ba0d867e-eb4d-4f27-95ed-507f7a5d9706" />
</p>

## Results
- Developed self-supervised foundation model for NIRSpec data analysis, with a pipeline for integration with observed data.
- Trained robust CNNs in a supervised setting to predict stellar parameters with high precision and accuracy.
- Applied cross-modal contrastive training to align pre-trained encoders around shared physical properties, creating a discriminative latent space.
- Applied dimension-reduction to the latent space, visualizing its intrinsic local geometry in a lower-dimensional setting.
- Enabled accurate transfer to downstream tasks, including in-modal and cross-modal cosine similarity searches and stellar property estimation from spectra.

## File Structure
This repository is structured as follows:

- `StarCLIP/`
	- `StarNet.ipynb`, `StarCLIP.ipynb`: Notebooks for training CNNs and implementing the contrastive framework.
	- `DATA/`: Input data files and generated datasets.
	- `SCRIPTS/`: Python scripts for data processing and analysis. Includes the class `Spectra` which conducts observed reductions and organizes matching synthetic and observed pairs.
	- `MODELS/`: Pre-trained model checkpoints.
	- `FIGURES/`: Figures and plots.

## Data
- `DATA/CATALOGS/`: Catalog CSVs (APOGEE, Gaia, etc.)
- `DATA/LABELS/`: HDF5 label files (e.g., JWST_APOGEE.h5)
- `DATA/M71_SPECTRA/`: Observed Spectra files (FITS, TXT)

⚠️ Some large files (e.g., JWST_APOGEE.h5) may need to be downloaded externally. Place them in the appropriate folder as described in the notebooks.

## Installation
To reproduce results, follow these steps:

### 1. Clone the repository
```bash
git clone <repo-url>
cd StarCLIP
```
### 2. Run the notebooks
Launch the notebooks with:
```bash
jupyter notebook
```
Open `StarNet.ipynb` and `StarCLIP.ipynb` from the `StarCLIP/` folder.

---

For more details, see the individual notebook documentation and comments.

## In Progress / Next Steps

- Integrate M71 observations into the ML framework
- Create better pathways for data download
- Prepare code and documentation for broader community use (requirements.txt, improved documentation, etc.)