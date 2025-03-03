from torch import nn
from collections import OrderedDict


class HiddenBlock(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_layers: tuple[int:] = None,
                 dropout: float = 0.2):
        super(HiddenBlock, self).__init__()
        if hidden_layers is None:
            self.hidden_layers = []
        elif isinstance(hidden_layers, int):
            self.hidden_layers = [hidden_layers]
        else:
            self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.num_of_layers = len(self.hidden_layers)
        self.dropout = dropout

        self.dense_layers = self._build_dense_layers()

    def _build_dense_layers(self):
        layers = []
        if self.num_of_layers == 0:
            return nn.Identity()
        else:
            in_features = self.input_size
            for index, layer_size in enumerate(self.hidden_layers, start=1):
                layers.append((
                    f'dense_layer{index}',
                    nn.Sequential(
                        nn.Linear(in_features=in_features,
                                  out_features=layer_size),
                        nn.ReLU(),
                        nn.LayerNorm(layer_size),
                        nn.Dropout(p=self.dropout)
                    )
                ))
                in_features = layer_size
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.dense_layers(x)

    def summary(self):
        print(self)


class DenseEncoder(HiddenBlock):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 hidden_layers: tuple[int:] = None,
                 latent_space_dim: int = 2,
                 dropout: float = 0.2):
        self.input_shape = input_shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        super(DenseEncoder, self).__init__(input_size=input_size,
                                           hidden_layers=hidden_layers,
                                           dropout=dropout)
        self.latent_space_dim = latent_space_dim

        self.flatten = nn.Flatten()
        self.output_layer = self._build_output_layer()

    def _build_output_layer(self):
        if self.num_of_layers == 0:
            in_features = self.input_size
        else:
            in_features = self.hidden_layers[-1]
        return nn.Linear(in_features=in_features,
                         out_features=self.latent_space_dim)

    def forward(self, x):
        return self.output_layer(self.dense_layers(self.flatten(x)))


class DenseDecoder(HiddenBlock):

    def __init__(self,
                 output_shape: tuple[int, int, int],
                 hidden_layers: tuple[int:] = None,
                 latent_space_dim: int = 2,
                 dropout: float = 0.2):
        super(DenseDecoder, self).__init__(input_size=latent_space_dim,
                                           hidden_layers=hidden_layers,
                                           dropout=dropout)
        self.output_shape = output_shape
        self.latent_space_dim = latent_space_dim
        self.output_layer = self._build_output_layer()

    def _build_output_layer(self):
        if self.num_of_layers == 0:
            in_features = self.latent_space_dim
        else:
            in_features = self.hidden_layers[-1]
        out_features = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        return nn.Sequential(nn.Linear(in_features=in_features,
                                       out_features=out_features),
                             nn.Sigmoid())

    def forward(self, x):
        return self.output_layer(self.dense_layers(x)).view(x.size(0), *self.output_shape)


class DenseAutoencoder(nn.Module):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 encoder_hidden_layers: tuple[int:] = None,
                 decoder_hidden_layers: tuple[int:] = None,
                 latent_space_dim: int = 2,
                 dropout: float = 0.1):
        super(DenseAutoencoder, self).__init__()

        self.encoder = DenseEncoder(input_shape=input_shape,
                                    hidden_layers=encoder_hidden_layers,
                                    latent_space_dim=latent_space_dim,
                                    dropout=dropout)

        if decoder_hidden_layers is None:
            decoder_hidden_layers = self.encoder.hidden_layers[::-1]
        self.decoder = DenseDecoder(output_shape=input_shape,
                                    hidden_layers=decoder_hidden_layers,
                                    latent_space_dim=latent_space_dim,
                                    dropout=dropout)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def summary(self):
        print(self)
