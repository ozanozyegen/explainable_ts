from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import embedding, nn
from torch.utils.data.dataloader import DataLoader

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from models.polar_dense_layers import PolarDense, EmbNet


class MyDecoderMLP(BaseModelWithCovariates):
    """
    MLP on the decoder.

    MLP that predicts output only based on information available in the decoder.
    """

    def __init__(
        self,
        config: dict = {},
        out_activation: str = "sigmoid",
        drop_input: float = 0,
        activation_class: str = "ReLU",
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
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
        output_size: Union[int, List[int]] = 1,
        target: Union[str, List[str]] = None,
        loss: MultiHorizonMetric = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Args:
            activation_class (str, optional): PyTorch activation class. Defaults to "ReLU".
            hidden_size (int, optional): hidden recurrent size - the most important hyperparameter along with
                ``n_hidden_layers``. Defaults to 10.
            n_hidden_layers (int, optional): Number of hidden layers - important hyperparameter. Defaults to 2.
            dropout (float, optional): Dropout. Defaults to 0.1.
            norm (bool, optional): if to use normalization in the MLP. Defaults to True.
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
            output_size (Union[int, List[int]], optional): number of outputs (e.g. number of quantiles for
                QuantileLoss and one target or list of output sizes).
            target (str, optional): Target variable or list of target variables. Defaults to None.
            loss (MultiHorizonMetric, optional): loss: loss function taking prediction and targets.
                Defaults to QuantileLoss.
            logging_metrics (nn.ModuleList, optional): Metrics to log during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
        """
        if loss is None:
            loss = QuantileLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList(
                [SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.input_embeddings = MultiEmbedding(
            embedding_sizes={
                name: val
                for name, val in embedding_sizes.items()
                if name in self.decoder_variables + self.static_variables
            },
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )
        # define network
        if isinstance(self.hparams.output_size, int):
            mlp_output_size = self.hparams.output_size
        else:
            mlp_output_size = sum(self.hparams.output_size)

        cont_size = len(self.decoder_reals_positions)
        cat_size = sum([v[1]
                       for v in self.input_embeddings.embedding_sizes.values()])
        history_size = self.hparams.config['history_size']
        input_size = (cont_size + cat_size) * (history_size + 1)

        model_name = self.hparams.config.get('MODEL')
        if model_name == 'emb_net':
            self.mlp = EmbNet(
                config=self.hparams.config,
                input_size=input_size,
                norm=self.hparams.norm,
                activation_class=getattr(nn, self.hparams.activation_class),
                hidden_size=self.hparams.hidden_size,
                n_hidden_layers=self.hparams.n_hidden_layers,
                dropout=self.hparams.dropout,
            )
        elif model_name == 'polar_dense':
            self.mlp = PolarDense(
                config=self.hparams.config,
                input_size=input_size,
                output_size=self.hparams.config['target_size'],
                hidden_size=self.hparams.hidden_size,
                n_hidden_layers=self.hparams.n_hidden_layers,
                out_activation=getattr(nn, self.hparams.out_activation),
                drop_input=self.hparams.drop_input  # TODO: Not implemented yet
            )

    @property
    def decoder_reals_positions(self) -> List[int]:
        return [
            self.hparams.x_reals.index(name)
            for name in self.reals
            if name in self.decoder_variables + self.static_variables
        ]

    def construct_encoder_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor
    ) -> torch.Tensor:
        if len(self.categoricals) > 0:
            embeddings = self.input_embeddings(x_cat)
            flat_embeddings = torch.cat(
                [emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        return input_vector

    def construct_decoder_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.input_embeddings(x_cat)
        input_vector = torch.cat([x_cont[..., self.decoder_reals_positions]] +
                                 list(embeddings.values()), dim=-1)
        return input_vector

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        # x is a batch generated based on the TimeSeriesDataset
        encoder_input_vector = self.construct_encoder_vector(
            x['encoder_cat'], x['encoder_cont'])
        decoder_input_vector = self.construct_decoder_vector(
            x['decoder_cat'], x['decoder_cont'])
        inp = dict({'encoder_vec': encoder_input_vector,
                    'decoder_vec': decoder_input_vector})
        interpret = hasattr(self, '_interpret_dict')
        prediction = self.mlp(inp, interpret)
        if interpret:
            self._interpret_dict.update(self.mlp._interpret_dict)

        # cut prediction into pieces for multiple targets
        if self.n_targets > 1:
            prediction = torch.split(
                prediction, self.hparams.output_size, dim=-1)
        # NOTE: prediction.shape == (batch_size, target_size, 1)
        # We need to return a dictionary that at least contains the prediction
        # The parameter can be directly forwarded from the input.
        prediction = self.transform_output(
            prediction, target_scale=x["target_scale"])
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = cls.deduce_default_output_parameters(
            dataset, kwargs, QuantileLoss())
        kwargs.update(new_kwargs)
        return super().from_dataset(dataset, **kwargs)

    def predict(self,
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
                interpret: bool = False,
                **kwargs):
        if interpret:
            self._interpret_dict = dict(embedding_sizes=self.input_embeddings.embedding_sizes,
                                        x_categoricals=self.hparams.x_categoricals,
                                        x_reals=self.hparams.x_reals)
        return super().predict(data, mode, return_index,
                               return_decoder_lengths, batch_size, num_workers,
                               fast_dev_run, show_progress_bar, return_x,
                               mode_kwargs, **kwargs)

