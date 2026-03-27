# lj-cdft-lmft

Code to train a WCA system and perform cDFT/LMFT of Lennard-Jones fluids (one-component and binary systems).

Contains the following:
* ```training```: Code and data for training a neural network for a one-component WCA system.
    * ```training.py``` takes in the temperature as a direct input into the neural network.
    * ```training_parameters.py``` takes in the temperature as a parameter (may be more helpful with multiple inputs and/or parameters).
    * ```training_hyperparameter.py``` is an example script of hyperparameter optimisation (requires KerasTuner).
 
* ```models```: WCA model obtained from training. For comparison with the hard sphere reference/mean-field theory, please place the hard sphere model from Sammüller *et al.*[^1] here.
* ```data```: Some cDFT results for reference and simulation data.
* ```cdft```: Scripts to perform cDFT/LMFT. Equations of state for LMFT are implemented within ```LJEOS.py```. Included are:
    * Johnson *et al.*[^2] written for one-component system
    * PeTS[^3] (requires FeOs package)[^4] written for one-component and binary system

[^1]: F. Sammüller, S. Hermann, D. De Las Heras and M. Schmidt, *Proc. Natl. Acad. Sci. U.S.A.*, 2023, **120**, e2312484120.
[^2]: J. K. Johnson, J. A. Zollweg and K. E. Gubbins, *Mol. Phys.*, 1993, **78**, 591–618.
[^3]: M. Heier, S. Stephan, J. Liu, W. G. Chapman, H. Hasse and K. Langenbach, *Mol. Phys.*, 2018, **116**, 2083–2094.
[^4]: P. Rehner, G. Bauer and J. Gross, *Ind. Eng. Chem. Res.*, 2023, **62**, 5347–5357.
