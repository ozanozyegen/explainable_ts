import torch
import numpy as np
from torch import dropout, nn
from typing import Union, Dict
from pytorch_forecasting.models.mlp import FullyConnectedModule


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation_class: nn.ReLU,
        n_hidden_layers: int = 1,
        dropout: float = 0,
        norm: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_class = activation_class
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.norm = norm
        self.output_size = hidden_size

        module_list = [nn.Linear(self.input_size, hidden_size), activation_class()]
        if dropout is not None:
            module_list.append(nn.Dropout(dropout))
        if norm:
            module_list.append(nn.BatchNorm1d(hidden_size))

        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size),
                                activation_class()])
            if dropout is not None:
                module_list.append(nn.Dropout(dropout))
            if norm:
                module_list.append(nn.BatchNorm1d(hidden_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor):
        return self.sequential(x)


class EmbNet(nn.Module):
    def __init__(
        self,
        config,
        input_size,
        norm,
        activation_class,
        hidden_size,
        n_hidden_layers,
        dropout,
    ):
        super().__init__()
        self.fcn = FullyConnectedModule(
            norm=norm,
            activation_class=activation_class,
            input_size=input_size,  # //(config['history_size']+1),
            output_size=1,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    def forward(self, x: Dict[str, torch.Tensor]):
        batch_size = x['encoder_vec'].shape[0]
        predictions = []
        for target_time in range(x['decoder_vec'].shape[1]):
            encoder_vec = x['encoder_vec'].reshape(batch_size, -1)
            decoder_vec = x['decoder_vec'][:, target_time:target_time + 1, :].reshape(batch_size, -1)
            v = torch.cat([encoder_vec, decoder_vec], dim=-1)
            # v = decoder_vec
            predictions.append(self.fcn(v))
        return torch.stack(predictions, dim=1)


# class EmbNet(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         encoder: EmbeddingNetEncoder,
#         out_activation: Union[nn.Sigmoid, nn.Identity],
#         drop_input: float = 0,
#     ):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.encoder = encoder
#         self.encoder_out_size = self.encoder.output_size
#         self.out_activation = out_activation
#         self.drop_input = drop_input

#         module_list = []
#         # if drop_input is not None:
#         #     module_list.append(nn.Dropout(drop_input))
#         module_list.append(encoder)
#         module_list.extend([nn.Linear(self.encoder_out_size, 1),
#                             out_activation()])

#         self.sequential = nn.Sequential(*module_list)

#     def forward(self, x: Dict[str, torch.Tensor]):
#         # x of shape: batch_size x n_features
#         # output of shape: batch_size x hidden_size / 2**n_hidden_layers
#         return self.sequential(x)


class PolarDense(nn.Module):
    def __init__(
        self,
        config,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        out_activation: Union[nn.Sigmoid, nn.Identity],
        drop_input: float = 0,
    ):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.out_activation = out_activation()
        self.drop_input = drop_input
        self._interpret_dict = dict()

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            activation_class=nn.ReLU,
            n_hidden_layers=n_hidden_layers,
            dropout=self.config['dropout'],
            norm=self.config['norm']
        )
        self.encoder_out_size = self.encoder.output_size

        self.dense = nn.Linear(self.encoder_out_size, input_size)
        self.out_projectors = nn.ModuleList([
            nn.Linear(input_size,
                      1,
                      bias=self.config['distribution_projector_bias'])
            for _ in range(self.output_size)
        ])

    def forward(self, x: Dict[str, torch.tensor], interpret):
        batch_size = x['encoder_vec'].shape[0]
        predictions = []
        for target_time in range(x['decoder_vec'].shape[1]):
            encoder_vec = x['encoder_vec'].reshape(batch_size, -1)
            decoder_vec = x['decoder_vec'][:, target_time:target_time + 1, :].reshape(batch_size, -1)
            v = torch.cat([encoder_vec, decoder_vec], dim=-1)

            out_encoder = self.encoder(v)
            out_encoder_projection = self.dense(out_encoder)
            assert out_encoder_projection.shape == v.shape
            c = out_encoder_projection * v
            out = self.out_projectors[target_time](c)
            out_activation = self.out_activation(out)
            predictions.append(out_activation)
            # Fill interpret packet
            if interpret:
                self._add_decoder_interpret(
                    target=target_time,
                    merged_context=c.clone().detach().numpy(),
                    output=out.clone().unsqueeze(dim=-1).detach().numpy(),
                    W=self.out_projectors[target_time].weight.clone().detach().numpy(),
                    b=self.out_projectors[target_time].bias  # .clone().detach().numpy())
                )
        return torch.stack(predictions, dim=1)

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
