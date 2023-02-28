# On Controller-Tuning with Time-Varying Optimization
This repository contains the code for the paper "On Controller-Tuning with Time-Varying Optimization (2022)" accepted at the [61st IEEE Conference on Decision and Control](https://cdc2022.ieeecss.org/).

![LQR_Regret_combined6](https://user-images.githubusercontent.com/49341051/158646082-e957109f-cd6a-4a43-8d78-6e9a373f4aab.png)

We propose a novel model for time-varying Bayesian optimization called **UI-TVBO** which leverages the concept of *uncertainty injection*. UI-TVBO is especially suitable for non-stationary time-varying optimization problems such as controller-tuning. Motivated by the LQR controller-tuning problem, we further include convexity constraints on the GP to increase sample efficiency. The model is implemented using [GPyTorch](https://gpytorch.ai) and [BoTorch](https://botorch.org).

If you find our code or paper useful, please consider citing
```
@inproceedings{brunzema2022controller,
  title={On controller tuning with time-varying bayesian optimization},
  author={Brunzema, Paul and Von Rohr, Alexander and Trimpe, Sebastian},
  booktitle={2022 IEEE 61st Conference on Decision and Control (CDC)},
  pages={4046--4052},
  year={2022},
  organization={IEEE}
}
```


## Install packages via pip

To install the necessary packages into an environment with python 3.9.10 use the `requirements.txt` file as

```
pip install -r requirements.txt
```

## Results and parameters

All the hyperparameters (Gamma prior, feasible set, etc.) used in the paper are specified in the corresponding jupyter notebooks (`*.ipynb`). The parameters of the simulated inverted pendulum are based on a hardware experiment and are specified in `src/objective_functions_LQR.py`.
To recreate our results, first install the necessary packages and then just run the notebooks. :)
