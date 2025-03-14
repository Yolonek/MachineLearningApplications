import torch
import operator
from torch import nn
from collections import OrderedDict
from functools import reduce


class DenseBlock(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_layers: tuple[int:] = None,
                 dropout: float = 0.2):
        super(DenseBlock, self).__init__()
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


class DenseEncoder(DenseBlock):

    def __init__(self,
                 input_shape: tuple[int:],
                 hidden_layers: tuple[int:] = None,
                 latent_space_dimension: int = 2,
                 dropout: float = 0.2):
        self.input_shape = input_shape
        input_size = reduce(operator.mul, input_shape)
        super(DenseEncoder, self).__init__(input_size=input_size,
                                           hidden_layers=hidden_layers,
                                           dropout=dropout)
        self.latent_space_dim = latent_space_dimension

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


class DenseDecoder(DenseBlock):

    def __init__(self,
                 output_shape: tuple[int:],
                 hidden_layers: tuple[int:] = None,
                 latent_space_dimension: int = 2,
                 dropout: float = 0.2):
        super(DenseDecoder, self).__init__(input_size=latent_space_dimension,
                                           hidden_layers=hidden_layers,
                                           dropout=dropout)
        self.output_shape = output_shape
        self.latent_space_dim = latent_space_dimension
        self.output_layer = self._build_output_layer()

    def _build_output_layer(self):
        if self.num_of_layers == 0:
            in_features = self.latent_space_dim
        else:
            in_features = self.hidden_layers[-1]
        out_features = reduce(operator.mul, self.output_shape)
        return nn.Sequential(nn.Linear(in_features=in_features,
                                       out_features=out_features),
                             nn.Sigmoid())

    def forward(self, x):
        return self.output_layer(self.dense_layers(x)).view(x.size(0), *self.output_shape)


class DenseAutoencoder(nn.Module):

    def __init__(self,
                 input_shape: tuple[int:],
                 encoder_hidden_layers: tuple[int:] = None,
                 decoder_hidden_layers: tuple[int:] = None,
                 latent_space_dimension: int = 2,
                 dropout: float = 0.1):
        super(DenseAutoencoder, self).__init__()

        self.encoder = DenseEncoder(input_shape=input_shape,
                                    hidden_layers=encoder_hidden_layers,
                                    latent_space_dimension=latent_space_dimension,
                                    dropout=dropout)

        if decoder_hidden_layers is None:
            decoder_hidden_layers = self.encoder.hidden_layers[::-1]
        self.decoder = DenseDecoder(output_shape=input_shape,
                                    hidden_layers=decoder_hidden_layers,
                                    latent_space_dimension=latent_space_dimension,
                                    dropout=dropout)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def summary(self):
        print(self)


class DenseVariationalEncoder(DenseBlock):

    def __init__(self,
                 input_shape: tuple[int:],
                 hidden_layers: tuple[int:] = None,
                 latent_space_dimension: int = 2,
                 dropout: float = 0.2):
        self.input_shape = input_shape
        input_size = reduce(operator.mul, input_shape)
        super(DenseVariationalEncoder, self).__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            dropout=dropout)
        self.latent_space_dim = latent_space_dimension

        self.flatten = nn.Flatten()
        self.dense_mu, self.dense_logvar = self._build_variational_layer()

    def _build_variational_layer(self):
        if self.num_of_layers == 0:
            in_features = self.input_size
        else:
            in_features = self.hidden_layers[-1]
        mu_layer = nn.Linear(in_features=in_features,
                             out_features=self.latent_space_dim)
        logvar_layer = nn.Linear(in_features=in_features,
                                 out_features=self.latent_space_dim)
        return mu_layer, logvar_layer

    def forward(self, x):
        x = self.dense_layers(self.flatten(x))
        return self.dense_mu(x), self.dense_logvar(x)

    def summary(self):
        print(self)


class DenseVariationalAutoencoder(nn.Module):

    def __init__(self,
                 input_shape: tuple[int:],
                 encoder_hidden_layers: tuple[int:] = None,
                 decoder_hidden_layers: tuple[int:] = None,
                 latent_space_dimension: int = 2,
                 dropout: float = 0.1):
        super(DenseVariationalAutoencoder, self).__init__()

        self.encoder = DenseVariationalEncoder(
            input_shape=input_shape,
            hidden_layers=encoder_hidden_layers,
            latent_space_dimension=latent_space_dimension,
            dropout=dropout)

        if decoder_hidden_layers is None:
            decoder_hidden_layers = self.encoder.hidden_layers[::-1]
        self.decoder = DenseDecoder(
            output_shape=input_shape,
            hidden_layers=decoder_hidden_layers,
            latent_space_dimension=latent_space_dimension,
            dropout=dropout)

    @staticmethod
    def reparametrization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def encode_and_reparametrize(self, x):
        return self.reparametrization(*self.encoder(x))

    def forward_mu_logvar(self, mu, logvar):
        z = self.reparametrization(mu, logvar)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.forward_mu_logvar(mu, logvar), mu, logvar

    def summary(self):
        print(self)


class ConvolutionalBlock(nn.Module):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 convolutional_filters: tuple[int:],
                 convolutional_kernels: tuple[int:],
                 convolutional_strides: tuple[int:]):
        super(ConvolutionalBlock, self).__init__()

        self.input_shape = input_shape
        self.convolutional_filters = convolutional_filters
        self.convolutional_kernels = convolutional_kernels
        self.convolutional_strides = convolutional_strides
        self.num_conv_layers = len(self.convolutional_filters)

        self.conv_layers = self._build_convolutional_layers()

    def _build_convolutional_layers(self):
        layers = []
        in_channels = self.input_shape[0]
        for i in range(self.num_conv_layers):
            layers.append((
                f'conv{i + 1}',
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.convolutional_filters[i],
                        kernel_size=self.convolutional_kernels[i],
                        stride=self.convolutional_strides[i],
                        padding=(self.convolutional_kernels[i] - 1) // 2
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.convolutional_filters[i])
                )
            ))
            in_channels = self.convolutional_filters[i]
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.conv_layers(x)

    def summary(self):
        print(self)


class ConvolutionalEncoder(ConvolutionalBlock):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 convolutional_filters: tuple[int:],
                 convolutional_kernels: tuple[int:],
                 convolutional_strides: tuple[int:],
                 latent_space_dimension: int = 2):
        super(ConvolutionalEncoder, self).__init__(
            input_shape=input_shape,
            convolutional_filters=convolutional_filters,
            convolutional_kernels=convolutional_kernels,
            convolutional_strides=convolutional_strides
        )
        self.latent_space_dim = latent_space_dimension
        self.shape_before_bottleneck = None
        self.shape_flattened = None

        self.output_layer = self._build_output_layer()

    def _build_output_layer(self):
        dummy_input = torch.zeros(1, *self.input_shape)
        conv_out = self.conv_layers(dummy_input)
        self.shape_before_bottleneck = conv_out.shape[1:]
        self.shape_flattened = conv_out.numel()
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.shape_flattened,
                      out_features=self.latent_space_dim))

    def forward(self, x):
        return self.output_layer(self.conv_layers(x))

    def summary(self):
        print(self)


class ConvolutionalTransposeBlock(nn.Module):

    def __init__(self,
                 shape_before_bottleneck: tuple[int:],
                 convolutional_transpose_filters: tuple[int:],
                 convolutional_transpose_kernels: tuple[int:],
                 convolutional_transpose_strides: tuple[int:]):
        super(ConvolutionalTransposeBlock, self).__init__()
        self.shape_before_bottleneck = shape_before_bottleneck
        self.convolutional_transpose_filters = convolutional_transpose_filters
        self.convolutional_transpose_kernels = convolutional_transpose_kernels
        self.convolutional_transpose_strides = convolutional_transpose_strides
        self.num_convT_filters = len(convolutional_transpose_filters)

        self.convT_layers = self._build_convolutional_transpose_layers()

    def _build_convolutional_transpose_layers(self):
        layers = []
        in_channels = self.shape_before_bottleneck[0]
        for i in range(self.num_convT_filters):
            padding = (self.convolutional_transpose_kernels[i] - 1) // 2
            output_padding = (self.convolutional_transpose_strides[i] -
                              self.convolutional_transpose_kernels[i] + (2 * padding))
            layers.append((
                f'convTranspose{i + 1}',
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=self.convolutional_transpose_filters[i],
                        kernel_size=self.convolutional_transpose_kernels[i],
                        stride=self.convolutional_transpose_strides[i],
                        padding=padding,
                        output_padding=output_padding
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.convolutional_transpose_filters[i])
                )
            ))
            in_channels = self.convolutional_transpose_filters[i]
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.convT_layers(x)

    def summary(self):
        print(self)


class ConvolutionalDecoder(ConvolutionalTransposeBlock):

    def __init__(self,
                 latent_space_dimension: int,
                 shape_before_bottleneck: tuple[int:],
                 convolutional_transpose_filters: tuple[int:],
                 convolutional_transpose_kernels: tuple[int:],
                 convolutional_transpose_strides: tuple[int:],
                 out_channels: int = 3):
        super(ConvolutionalDecoder, self).__init__(
            shape_before_bottleneck=shape_before_bottleneck,
            convolutional_transpose_filters=convolutional_transpose_filters,
            convolutional_transpose_kernels=convolutional_transpose_kernels,
            convolutional_transpose_strides=convolutional_transpose_strides
        )
        self.latent_space_dimension = latent_space_dimension
        self.shape_before_bottleneck = shape_before_bottleneck
        self.out_channels = out_channels

        self.dense_layer = self._build_dense_layer()
        self.output_layer = self._build_output_layer()

    def _build_dense_layer(self):
        flattened_size = self.shape_before_bottleneck[0] * \
                         self.shape_before_bottleneck[1] * \
                         self.shape_before_bottleneck[2]
        return nn.Linear(in_features=self.latent_space_dimension,
                         out_features=flattened_size)

    def _build_output_layer(self):
        padding = (self.convolutional_transpose_kernels[-1] - 1) // 2
        output_padding = (self.convolutional_transpose_strides[-1] -
                          self.convolutional_transpose_kernels[-1] + (2 * padding))
        output_convT = nn.ConvTranspose2d(
            in_channels=self.convolutional_transpose_filters[-1],
            out_channels=self.out_channels,
            kernel_size=self.convolutional_transpose_kernels[-1],
            stride=self.convolutional_transpose_strides[-1],
            padding=padding,
            output_padding=output_padding
        )
        return nn.Sequential(output_convT, nn.Sigmoid())

    def forward(self, x):
        return self.output_layer(self.convT_layers(
            self.dense_layer(x).view(x.size(0), *self.shape_before_bottleneck)))

    def summary(self):
        print(self)


class ConvolutionalAutoencoder(nn.Module):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 convolutional_filters: tuple[int:],
                 convolutional_kernels: tuple[int:],
                 convolutional_strides: tuple[int:],
                 latent_space_dimension: int = 2,
                 convolutional_transpose_filters: tuple[int:] = None,
                 convolutional_transpose_kernels: tuple[int:] = None,
                 convolutional_transpose_strides: tuple[int:] = None):
        super(ConvolutionalAutoencoder, self).__init__()

        if convolutional_transpose_filters is None:
            convolutional_transpose_filters = convolutional_filters[::-1]
        if convolutional_transpose_kernels is None:
            convolutional_transpose_kernels = convolutional_kernels[::-1]
        if convolutional_transpose_strides is None:
            convolutional_transpose_strides = convolutional_strides[::-1]

        self.encoder = ConvolutionalEncoder(
            input_shape=input_shape,
            convolutional_filters=convolutional_filters,
            convolutional_kernels=convolutional_kernels,
            convolutional_strides=convolutional_strides,
            latent_space_dimension=latent_space_dimension
        )
        self.decoder = ConvolutionalDecoder(
            latent_space_dimension=latent_space_dimension,
            shape_before_bottleneck=self.encoder.shape_before_bottleneck,
            convolutional_transpose_filters=convolutional_transpose_filters,
            convolutional_transpose_kernels=convolutional_transpose_kernels,
            convolutional_transpose_strides=convolutional_transpose_strides,
            out_channels=input_shape[0]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def summary(self):
        print(self)


class ConvolutionalVariationalEncoder(ConvolutionalBlock):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 convolutional_filters: tuple[int:],
                 convolutional_kernels: tuple[int:],
                 convolutional_strides: tuple[int:],
                 latent_space_dimension: int = 2):
        super(ConvolutionalVariationalEncoder, self).__init__(
            input_shape=input_shape,
            convolutional_filters=convolutional_filters,
            convolutional_kernels=convolutional_kernels,
            convolutional_strides=convolutional_strides
        )
        self.latent_space_dim = latent_space_dimension
        self.shape_before_bottleneck = None
        self.shape_flattened = None

        self.flatten = nn.Flatten()
        self.dense_mu, self.dense_logvar = self._build_variational_layer()

    def _build_variational_layer(self):
        dummy_input = torch.zeros(1, *self.input_shape)
        conv_out = self.conv_layers(dummy_input)
        self.shape_before_bottleneck = conv_out.shape[1:]
        self.shape_flattened = conv_out.numel()
        mu_layer = nn.Linear(in_features=self.shape_flattened,
                             out_features=self.latent_space_dim)
        logvar_layer = nn.Linear(in_features=self.shape_flattened,
                                 out_features=self.latent_space_dim)
        return mu_layer, logvar_layer

    def forward(self, x):
        x = self.flatten(self.conv_layers(x))
        return self.dense_mu(x), self.dense_logvar(x)

    def summary(self):
        print(self)


class ConvolutionalVariationalAutoencoder(nn.Module):

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 convolutional_filters: tuple[int:],
                 convolutional_kernels: tuple[int:],
                 convolutional_strides: tuple[int:],
                 latent_space_dimension: int = 2,
                 convolutional_transpose_filters: tuple[int:] = None,
                 convolutional_transpose_kernels: tuple[int:] = None,
                 convolutional_transpose_strides: tuple[int:] = None):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        if convolutional_transpose_filters is None:
            convolutional_transpose_filters = convolutional_filters[::-1]
        if convolutional_transpose_kernels is None:
            convolutional_transpose_kernels = convolutional_kernels[::-1]
        if convolutional_transpose_strides is None:
            convolutional_transpose_strides = convolutional_strides[::-1]

        self.encoder = ConvolutionalVariationalEncoder(
            input_shape=input_shape,
            convolutional_filters=convolutional_filters,
            convolutional_kernels=convolutional_kernels,
            convolutional_strides=convolutional_strides,
            latent_space_dimension=latent_space_dimension
        )
        self.decoder = ConvolutionalDecoder(
            latent_space_dimension=latent_space_dimension,
            shape_before_bottleneck=self.encoder.shape_before_bottleneck,
            convolutional_transpose_filters=convolutional_transpose_filters,
            convolutional_transpose_kernels=convolutional_transpose_kernels,
            convolutional_transpose_strides=convolutional_transpose_strides,
            out_channels=input_shape[0]
        )

    @staticmethod
    def reparametrization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def encode_and_reparametrize(self, x):
        return self.reparametrization(*self.encoder(x))

    def forward_mu_logvar(self, mu, logvar):
        z = self.reparametrization(mu, logvar)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.forward_mu_logvar(mu, logvar), mu, logvar

    def summary(self):
        print(self)


class KLDivergence(nn.Module):

    def __init__(self, kld_weight=0.02):
        super(KLDivergence, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, mu, logvar):
        kl_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_batch, dim=0)
        return self.kld_weight * kl_loss


class VAELoss(nn.Module):

    def __init__(self, reconstruction_loss=nn.MSELoss(), kld_weight=0.02):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence = KLDivergence(kld_weight=kld_weight)

    def forward(self, r_x, x, mu, logvar):
        reconstruction_loss = self.reconstruction_loss(r_x, x)
        kl_loss = self.kl_divergence(mu, logvar)
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss.detach(), kl_loss.detach()


