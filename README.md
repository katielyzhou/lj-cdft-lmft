# lj-cdft-lmft

## About the code
This repository contains the code to train a WCA system and perform neural LMFT of Lennard-Jones fluids (one-component and binary systems).

Details can be found in the paper [here][https://arxiv.org/abs/2601.21591].

Contains the following:
* ```training```: Code and data for training a neural network for a one-component WCA system.
    * ```training.py``` takes in the temperature as a direct input into the neural network.
    * ```training_parameters.py``` takes in the temperature as a parameter (may be more helpful with multiple inputs and/or parameters).
    * ```training_hyperparameter.py``` is an example script of hyperparameter optimisation (requires KerasTuner).
 
* ```models```: WCA model obtained from training. For comparison with the hard sphere reference/mean-field theory, please place the hard sphere model from Sammüller *et al.*[^1] here.
* ```data```: Some cDFT results for reference and simulation data.
* ```cdft```: Scripts to perform cDFT/neural LMFT. Equations of state for LMFT are implemented within ```LJEOS.py```. Included are:
    * Johnson *et al.*[^2] written for one-component system
    * PeTS[^3] (requires FeOs package)[^4] written for one-component and binary system

[^1]: F. Sammüller, S. Hermann, D. De Las Heras and M. Schmidt, *Proc. Natl. Acad. Sci. U.S.A.*, 2023, **120**, e2312484120.
[^2]: J. K. Johnson, J. A. Zollweg and K. E. Gubbins, *Mol. Phys.*, 1993, **78**, 591–618.
[^3]: M. Heier, S. Stephan, J. Liu, W. G. Chapman, H. Hasse and K. Langenbach, *Mol. Phys.*, 2018, **116**, 2083–2094.
[^4]: P. Rehner, G. Bauer and J. Gross, *Ind. Eng. Chem. Res.*, 2023, **62**, 5347–5357.

## Data
Training data and cDFT results for mixtures may be found on Zenodo.

## Installation
You can clone the repository with:
```sh
git clone https://github.com/CoxGroup/lj-cdft-lmft.git
```
To create a *conda* environment containing the required packages 

```sh
conda env create -f environment.yml
```
## Usage
For training and evaluating the model, *TensorFlow/Keras* is used. This can be done either on a CPU or GPU, though performance is better on the latter. Training data can be found [here][https://doi.org/10.5281/zenodo.19250208] on Zenodo and placed in ```training/data```.

To obtain equilibrium density profiles, see examples in ```cdft/notebooks_single/slit_profiles.ipynb``` for the single-component system and ```cdft/notebooks_mixture/azeotrope.ipynb``` for mixtures. For mixtures, comparisons to mean-field approaches can be found in ```cdft/notebooks_mixture/lmft_trad.ipynb```. Existing data can be found on [Zenodo][https://doi.org/10.5281/zenodo.19250208] and analysed with ```cdft/notebooks_mixture/thermodynamics.ipynb```.
