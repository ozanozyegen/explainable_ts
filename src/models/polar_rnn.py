"""
Polar RNN version
"""
from copy import copy, deepcopy
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot_date
import numpy as np
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.distributions as dists
import torch.nn as nn
from torch.nn.utils import rnn
from torch.utils.data.dataloader import DataLoader

from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    DistributionLoss,
    Metric,
    MultiLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import HiddenState, MultiEmbedding, get_rnn
from pytorch_forecasting.utils import apply_to_list, to_list

from models.layers import TimeDistributed


class PolarRNN(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        config: dict = {},
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, List[str]] = None,
        target_lags: Dict[str, List[int]] = {},
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Args:
            cell_type (str, optional): Recurrent cell type ["LSTM", "GRU"]. Defaults to "LSTM".
            hidden_size (int, optional): hidden recurrent size - the most important hyperparameter along with
                ``rnn_layers``. Defaults to 10.
            rnn_layers (int, optional): Number of RNN layers - important hyperparameter. Defaults to 2.
            dropout (float, optional): Dropout in RNN layers. Defaults to 0.1.
            static_categoricals: integer of positions of static categorical variables
            static_reals: integer of positions of static continuous variables
            time_varying_categoricals_encoder: integer of positions of categorical variables for encoder
            time_varying_categoricals_decoder: integer of positions of categorical variables for decoder
            time_varying_reals_encoder: integer of positions of continuous variables for encoder
            time_varying_reals_decoder: integer of positions of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            n_validation_samples (int, optional): Number of samples to use for calculating validation metrics.
                Defaults to None, i.e. no sampling at validation stage and using "mean" of distribution for logging
                metrics calculation.
            n_plotting_samples (int, optional): Number of samples to generate for plotting predictions
                during training. Defaults to ``n_validation_samples`` if not None or 100 otherwise.
            target (str, optional): Target variable or list of target variables. Defaults to None.
            target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped to list of time steps by
                which the variable should be lagged.
                Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
                add at least the target variables with the corresponding lags to improve performance.
                Defaults to no lags, i.e. an empty dictionary.
            loss (DistributionLoss, optional): Distribution loss function. Keep in mind that each distribution
                loss function might have specific requirements for target normalization.
                Defaults to :py:class:`~pytorch_forecasting.metrics.NormalDistributionLoss`.
            logging_metrics (nn.ModuleList, optional): Metrics to log during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
        """
        if loss is None:
            loss = RMSE()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if n_plotting_samples is None:
            if n_validation_samples is None:
                n_plotting_samples = n_validation_samples
            else:
                n_plotting_samples = 100
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        assert set(self.encoder_variables) - set(to_list(target)) - set(lagged_target_names) == set(
            self.decoder_variables
        ), "Encoder and decoder variables have to be the same apart from target variable"

        rnn_class = get_rnn(cell_type)
        cont_size = len(self.reals)
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        input_size = cont_size + cat_size
        history_size = self.hparams.config['history_size']
        target_size = self.hparams.config['target_size']
        # pidits
        self.alpha_rnn = rnn_class(
            input_size=input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
            bias=self.hparams.config['alpha_rnn_bias'],
        )
        self.beta_rnn = rnn_class(
            input_size=input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
            bias=self.hparams.config['beta_rnn_bias'],
        )
        rnn_out_size = (int(self.alpha_rnn.bidirectional) + 1) * self.hparams.hidden_size
        self.alpha_projector = TimeDistributed(nn.Linear(rnn_out_size, 1))
        self.alpha_attention = nn.Softmax(dim=1)
        self.beta_projector = TimeDistributed(nn.Linear(rnn_out_size, input_size))
        self.beta_activation = nn.Tanh()

        # add linear layers for argument projects
        if isinstance(target, str):  # single target
            # self.distribution_projector = nn.Linear(28, len(self.loss.distribution_arguments))
            self.output_projectors = nn.ModuleList(
                [nn.Linear(input_size * history_size + input_size, 1,
                           bias=self.hparams.config['distribution_projector_bias'])
                    for _ in range(target_size)]
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        config: dict = {},
        **kwargs,
    ):
        """
        Create model from dataset.
        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)
        Returns:
            DeepAR network
        """
        new_kwargs = {}
        if dataset.multi_target:
            new_kwargs.setdefault("loss", MultiLoss([NormalDistributionLoss()] * len(dataset.target_names)))
        new_kwargs.update(kwargs)
        assert not isinstance(dataset.target_normalizer, NaNLabelEncoder) and (
            not isinstance(dataset.target_normalizer, MultiNormalizer)
            or all([not isinstance(normalizer, NaNLabelEncoder) for normalizer in dataset.target_normalizer])
        ), "target(s) should be continuous - categorical targets are not supported"  # todo: remove this restriction
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            config=config, **new_kwargs
        )

    def construct_encoder_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Create input vector into RNN network
        Args:
            one_off_target: tensor to insert into first position of target. If None (default), remove first time step.
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        return input_vector

    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Create input vector into RNN network
        Args:
            one_off_target: tensor to insert into first position of target. If None (default), remove first time step.
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        return input_vector

    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"]
        input_vector = self.construct_encoder_vector(x["encoder_cat"], x["encoder_cont"])
        alpha_dec_output, _ = self.alpha_rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        beta_dec_output, _ = self.beta_rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        alpha_out = self.alpha_attention(self.alpha_projector(alpha_dec_output))
        beta_out = self.beta_activation(self.beta_projector(beta_dec_output))
        context = input_vector * alpha_out * beta_out
        if hasattr(self, '_interpret_dict'):
            self._add_interpret_dict('encoder_input_vector', input_vector)
            self._add_interpret_dict('alpha_out', alpha_out)
            self._add_interpret_dict('beta_out', beta_out)
            self._add_interpret_dict('encoder_context', context)
        return context

    def decode_all(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        idx: int = None,
    ):
        if context.ndim == 3:
            # context_sum = torch.sum(context, dim=1, keepdim=False)  # (bs, fs)
            context_sum = context.view(context.shape[0], -1)
        else:
            context_sum = context
        # Alternative option
        # context_repeat = torch.repeat_interleave(context_sum, )
        if idx is not None:  # Decode one step, runs on .predict()
            merged_context = torch.cat((context_sum, x[:,0,:]), dim=1)
            output = self.output_projectors[idx](merged_context)
            output = output.unsqueeze(dim=1)  # Add time dimension
            if hasattr(self, '_interpret_dict'):
                self._add_decoder_interpret(target=idx,
                                            merged_context=merged_context.clone().detach().numpy(),
                                            output=output.clone().detach().numpy(),
                                            W=self.output_projectors[idx].weight.clone().detach().numpy(),
                                            b=self.output_projectors[idx].bias)
            return output, context

        if isinstance(self.hparams.target, str):  # single target
            # output = self.distribution_projector(context)
            outputs = []
            for i, output_projector in enumerate(self.output_projectors):
                # Combine context with
                merged_context = torch.cat((context_sum, x[:, i, :]), dim=1)
                output = output_projector(merged_context)
                if hasattr(self, '_interpret_dict'):
                    self._add_decoder_interpret(target=i,
                                                merged_context=merged_context.clone().detach().numpy(),
                                                output=output.clone().unsqueeze(dim=-1).detach().numpy(),
                                                W=output_projector.weight.clone().detach().numpy(),
                                                b=output_projector)
                outputs.append(output)
            output = torch.stack(outputs, dim=1)
        else:
            output = [projector(context) for projector in self.distribution_projector]
        return output, context_sum

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: torch.Tensor,  # TODO: Rename context vector
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, _ = self.decode_all(input_vector, hidden_state, idx=None)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            # hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            hidden_state = hidden_state.repeat_interleave(n_samples, 0)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                if lagged_targets[-1].ndim == 3 and lagged_targets[-1].shape[-1] == 1:
                    lagged_targets[-1] = lagged_targets[-1][:, :, 0]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, hidden_state = self.decode_all(x, hidden_state, idx)
                # Here x.shape (n_samples*bs, 1, 87), prediction.shape (n_samples*bs, 1, 2)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
        return output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        context = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T,  # Last two values of the target in encoder x['encoder_cont'][:, -1, 1]
        )

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=context,
            n_samples=n_samples,
        )
        # return relevant part
        return self.to_network_output(prediction=output)
        # NOTE: mu, and sigma almost have the same range (~-670, 12k) Rossmann, same in deepar

    def create_log(self, x, y, out, batch_idx):
        n_samples = [self.hparams.n_validation_samples, self.hparams.n_plotting_samples][self.training]
        log = super().create_log(
            x,
            y,
            out,
            batch_idx,
            prediction_kwargs=dict(n_samples=n_samples),
            quantiles_kwargs=dict(n_samples=n_samples),
        )
        return log

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ) -> plt.Figure:
        # workaround for not being able to compute loss for single sample without parameters of distribution
        return super().plot_prediction(
            x, out, idx=idx, add_loss_to_title=False, show_future_observed=show_future_observed, ax=ax, **kwargs
        )

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        n_samples: int = None,  # NOTE: No Monte Carlo sampling
        interpret: bool = False,
    ):
        """
        predict dataloader
        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", "samples" or "raw", or tuple ``("raw", output_name)`` where
                output_name is a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            n_samples: number of samples to draw. Defaults to 100.
        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        if interpret:
            self._interpret_dict = dict(embedding_sizes=self.embeddings.embedding_sizes,
                                        x_categoricals=self.hparams.x_categoricals,
                                        x_reals=self.hparams.x_reals)

        if isinstance(mode, str):
            if mode in ["prediction", "quantiles"]:
                if mode_kwargs is None:
                    mode_kwargs = dict(use_metric=False)
                else:
                    mode_kwargs = deepcopy(mode_kwargs)
                    mode_kwargs["use_metric"] = False
            elif mode == "samples":
                mode = ("raw", "prediction")
        return super().predict(
            data=data,
            mode=mode,
            return_decoder_lengths=return_decoder_lengths,
            return_index=return_index,
            n_samples=n_samples,  # new keyword that is passed to forward function
            return_x=return_x,
            show_progress_bar=show_progress_bar,
            fast_dev_run=fast_dev_run,
            num_workers=num_workers,
            batch_size=batch_size,
            mode_kwargs=mode_kwargs,
        )

    def shap_fit_transform(self, batch_inp):
        self._shap_merger = InpMerger()
        return self._shap_merger.fit_transform(batch_inp)

    def lime_predict(self, merged_inp):
        inp = self._shap_merger.inverse_transform(merged_inp)
        output = self(inp)['prediction'].squeeze().detach().numpy()
        return output

    def shap_predict(self, target_idx):
        model = self

        def shap_predict_idx(merged_inp):
            inp = model._shap_merger.inverse_transform(merged_inp)
            output = model(inp)['prediction'].squeeze().detach().numpy()
            if output.ndim == 2:
                return output[:, target_idx]
            else:
                return output
        return shap_predict_idx

    def _add_interpret_dict(self, k, v: Union[torch.Tensor, np.array]):
        if k not in self._interpret_dict:
            self._interpret_dict[k] = v
        else:
            if type(v) == torch.Tensor:
                self._interpret_dict[k] = torch.concat((self._interpret_dict[k], v))
            elif type(v) == np.array:
                self._interpret_dict[k] = np.concat((self._interpret_dict[k], v))

    def _add_decoder_interpret(self, target: int,
                               merged_context: Union[torch.Tensor, np.array],
                               output=Union[torch.Tensor, np.array],
                               W=Union[torch.Tensor, np.array],
                               b=Union[torch.Tensor, np.array]):
        self._add_interpret_dict(f'{target}_merged_context', merged_context)
        self._add_interpret_dict(f'{target}_output', output)
        self._add_interpret_dict(f'{target}_W', W)
        self._add_interpret_dict(f'{target}_b', b)


class InpMerger:
    """ Helper class for transforming the input for SHAP """
    copy_keys = ['target_scale', 'encoder_lengths', 'decoder_lengths']

    def fit_transform(self, inp):
        """ Transform pytorch_forecasting input to a numpy array
            keep non-feature inputs (copy_keys) data in inp_dict for
            inverse_transformation later
        """
        self._saved_inp_dict = {k: v for k, v in inp.items() if k in self.copy_keys}
        self._dtypes = {k: v.dtype for k, v in inp.items() if v is not None}
        enc_cat, enc_cont = inp['encoder_cat'], inp['encoder_cont']
        dec_cat, dec_cont = inp['decoder_cat'], inp['decoder_cont']
        self._cont_feat, self._cat_feat = enc_cont.shape[-1], enc_cat.shape[-1]
        self.history_size, self.target_size = enc_cont.shape[1], dec_cont.shape[1]
        enc = torch.concat((enc_cont, enc_cat), axis=-1)
        dec = torch.concat((dec_cont, dec_cat), axis=-1)
        merged_inp = torch.concat((enc, dec), axis=1)
        merged_inp = merged_inp.reshape(merged_inp.shape[0], -1)
        return merged_inp.detach().numpy()

    def inverse_transform(self, merged_inp):
        """ Takes np.array input features and transforms back to the
            pytorch_forecasting input format
        """
        # Batch size might be changed, retrieve the batch size
        batch_size = merged_inp.shape[0]
        inp = self._saved_inp_dict.copy()
        merged_inp = merged_inp.reshape(batch_size,
                                        self.history_size + self.target_size,
                                        self._cont_feat + self._cat_feat)
        merged_inp = torch.Tensor(merged_inp)
        inp['encoder_cont'] = merged_inp[:, :self.history_size, :self._cont_feat]
        inp['encoder_cat'] = merged_inp[:, :self.history_size, self._cont_feat:]
        inp['decoder_cont'] = merged_inp[:, self.history_size:, :self._cont_feat]
        inp['decoder_cat'] = merged_inp[:, self.history_size:, self._cont_feat:]
        for k in ['encoder_cont', 'encoder_cat', 'decoder_cont', 'decoder_cat']:
            inp[k] = inp[k].type(self._dtypes[k])

        for k in self.copy_keys:
            # Adjust copy keys data based on the new batch size
            new_shape = tuple([batch_size] + [dim_sz for dim_sz in inp[k].shape[1:]])
            reshaped_arr = np.resize(inp[k].detach().numpy(), new_shape)
            inp[k] = torch.Tensor(reshaped_arr).type(self._dtypes[k])
        return inp
