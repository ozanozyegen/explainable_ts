# Explainable Neural Networks for Time Series Forecasting

## Citation
[SSRN - DNNLITS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4179881)
```
@article{ozyegendnnlits,
  title={Dnnlits: Deep Neural Networks for Locally Interpretable Time Series Forecasting},
  author={Ozyegen, Ozan and Cevik, Mucahit and Basar, Ayse},
  journal={Available at SSRN 4179881}
}
``` 

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
