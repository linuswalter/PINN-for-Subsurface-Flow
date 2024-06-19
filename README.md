 # FlowPINN Model

This repository contains all data models to reproduce the results of our paper ==insert paper citation here==.
This readme was tested under Manjaro Linux. It should work similarly under Windows/Mac. 🤔

## 1. Prerequisites

Open the terminal/command line and type the following commands:
```bash
conda create -n FlowPINN python=3.9 -y
conda activate FlowPINN

# Register kernel for jupyter notebook
python -m ipykernel install --user --name FlowPINN

pip install tensorflow==2.11.0
pip install -r requirements.txt
```

Do no open the Notebook in VSCode but rather in the original jupyter NB environment in order to be able to set the parameters.

## 1. Data
The `01_data` directory contains the csv-files `ogs_output_a_1e0.csv` and `ogs_output_a_5e0.csv` which contain the synthetic datasets for an equivalent fracture length of $a=1\text{m}$ and $a=5\text{m}$. 
These are outputs from the numerical code [OpenGeoSys](https://www.opengeosys.org/). To make these outputs more reproducible, we also appended the directories `model_a_1e0`  and `model_a_5e0` which contain the respective mesh files (See `mesh`-directory, the project file `LiquidFlow.prj` for running the OGS model and the OGS output files in the `pvd` and `vtu` format. The file `LiquidFlow_.pvd` can be viewed in the interactive GUI of [ParaView](https://www.paraview.org/).

## 2. Experiments

To reproduce the experimental results, just start initialize and run the Jupyter Notebook.
The model can be parameterized in the section "Experimental Parameters".

### 2.1 Experiment ANN versus PINN

For reproducing the first experiment, you need to select the following options when setting up the experiment parameters:

```
Name Ground Truth Dataset = ogs_output_a_5e0
Load weights for kd? = False
Load weights for pd_pinn? = False
Number Epochs Training kd = 500
Number Epochs pd = 1000
Perform retraining for too high loss values? = False
Number of Iterations = 1
Nr observation points = 25
Standard Deviation of the Gaussian Noise = 0
PINN with observation loss term? = True
Learning Rate values = [0.001, 1e-05]
Learning Rate method = step
Adaptive Weighting = None
Safe Plots as PNG? = True
Create GIF of PNG? = True
```

In the second part of the first experiment, one can set the experiment parameter `Standard Deviation of the Gaussian Noise` to a higher value, like $\sigma = 0.02$ or $\sigma = 0.04$.

### 2.2 Experiment ANN versus PINN

The second experiment can be reproduced by choosing the following parameter settings.

```
Name Ground Truth Dataset = ogs_output_a_1e0
Load weights for kd? = False
Load weights for pd_pinn? = False
Number Epochs Training kd = 500
Number Epochs pd = 1000
Perform retraining for too high loss values? = True
Number of Iterations = 20
Nr observation points = 0
Standard Deviation of the Gaussian Noise = 0
PINN with observation loss term? = False
Learning Rate values = [0.001, 1e-05]
Learning Rate method = step
Adaptive Weighting = None
Safe Plots as PNG? = True
Create GIF of PNG? = True
```

### Results

The results will be saved in the directory `02_model_output` -> `model_DATE_TIME` 