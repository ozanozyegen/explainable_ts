# Explainable Neural Networks for Time Series Forecasting

## Citation
Will be added
<!-- ```
@article{ozyegen2022evaluation,
  title={Evaluation of interpretability methods for multivariate time series forecasting},
  author={Ozyegen, Ozan and Ilic, Igor and Cevik, Mucahit},
  journal={Applied Intelligence},
  volume={52},
  number={5},
  pages={4727--4743},
  year={2022},
  publisher={Springer}
}
``` -->

### Experiments
- Train single model
    - `python exps/run_single.py`
- Tune a dataset-model pair via optuna framework
    - `python exps/run_tuning.py`
- Analyze the model performances
    - `python exps/review_results.ipynb`
- Analyze model forecasts and explanations
    - `python notebooks/polar_rnn_rossmann.ipynb`
    - A trained Rossmann $DNNLITS^{RNN}$ model checkpoint is available under `models/3r5uvsl1`

## Data Sources
### Rossmann
- https://www.kaggle.com/c/rossmann-store-sales
### Walmart
- https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

## Requirements
- python 3.8
- pytorch 1.11
- pytorch-forecasting 0.10.1
- wandb - Weights and Biases is used for tracking the experiments
- exp_ts.yml contains all the package dependencies