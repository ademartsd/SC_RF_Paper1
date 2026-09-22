# SC_RF_Paper1

Repository accompanying the manuscript:

**Crustal Structure Underneath South Carolina Determined by Receiver Function Analysis**

C. Ademar Fernández, Daniel A. Frost, Federico D. Munch, Shubham Agrawal, and H. Philip Crotwell

## Contents

**StatResults:** Station results are organized into `Brunswick` and `NonBrunswick` directories. This folder contains the station-by-station numerical and graphical results associated with the study, including:

- Moho-depth estimates
- Effective Vs estimates
- Vs-depth Kernel Density Estimation (KDE) plots
- Receiver-function fits
- Representative Vs-depth models
- Station summaries

**Code:** Source code used for the inversion and plotting. The code is organized into `Brunswick`, `NonBrunswick`, and `Sed_Reverb_Removal` directories. Instructions are provided within each directory.

**prs.yml:** Conda environment required to run the code. The environment can be installed using:

```bash
conda env create -f prs.yml
