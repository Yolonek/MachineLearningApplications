import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Dict
from collections import OrderedDict


class ModularVGG(nn.Module):
    """
    VGG model that can be arbitrarily modified
    """

    def __init__(self, input_layer_size: int, num_of_classes: int, image_size: int, config: Dict):
        super(ModularVGG, self).__init__()
        self.latest_layer_size = input_layer_size
        self.image_size = image_size

        self.conv_layer = nn.Sequential(
            *self._conv_layers(**config['CONV']),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            *self._linear_layers(**config['FC']),
            nn.Linear(in_features=self.latest_layer_size,
                      out_features=num_of_classes),
            nn.Softmax(dim=1)
        )

        self.initialize_weights()

    def _conv_layers(self, conv_layers: List[List[int]],
                     kernel_params: Dict,
                     batch_norm: bool = False,
                     activation: nn.Module = nn.ReLU(inplace=True)) -> List[nn.Module]:
        layers = []
        for conv_layer in conv_layers:
            for out_channels in conv_layer:
                layers += [nn.Conv2d(in_channels=self.latest_layer_size,
                                     out_channels=out_channels, **kernel_params), activation]
                self.latest_layer_size = out_channels
            layers += [nn.MaxPool2d(kernel_size=2)]
            layers += [nn.BatchNorm2d(num_features=self.latest_layer_size)] if batch_norm else []
        self.latest_layer_size = (self.image_size // 2 ** len(conv_layers)) ** 2 * self.latest_layer_size
        return layers

    def _linear_layers(self, layer_sizes: List[int],
                       dropout: float = 0.,
                       activation: nn.Module = nn.ReLU(inplace=True)) -> List[nn.Module]:
        layers = []
        for layer_size in layer_sizes:
            layers += [nn.Dropout(p=dropout)] if dropout > 0 else []
            layers += [nn.Linear(in_features=self.latest_layer_size,
                                 out_features=layer_size), activation]
            self.latest_layer_size = layer_size
        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_layer(x))


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes), nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual_block(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        activation = nn.ReLU()
        layer1 = self._make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(ResidualBlock, 512, num_blocks[3], stride=2)

        self.resnet_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), activation,
            layer1, layer2, layer3, layer4,
            nn.AvgPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * ResidualBlock.expansion, num_classes),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.resnet_layers(x))


def ResNet18():
    return ResNet([2, 2, 2, 2])


def ResNet34():
    return ResNet([3, 4, 6, 3])


def ResNet18Transfered():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64,
                            kernel_size=(3, 3), stride=(1, 1),
                            padding=(1, 1), bias=False)
    model.bn1 = nn.BatchNorm2d(64, eps=1e-05,
                               momentum=0.1, affine=True,
                               track_running_stats=True)
    model.relu = nn.Identity()
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(in_features=512, out_features=10)
    return model


class ModularNeuralNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layer_sizes: tuple[int] = (),
                 activation_function: nn.modules.activation = nn.Tanh()):
        super(ModularNeuralNetwork, self).__init__()
        if len(layer_sizes) == 0:
            self.layers = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=output_size),
                activation_function
            )
        elif len(layer_sizes) == 1:
            size = layer_sizes[0]
            self.layers = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=size),
                activation_function,
                nn.Linear(in_features=size, out_features=output_size),
                activation_function
            )
        else:
            layers = []
            for index, layer_size in enumerate(layer_sizes):
                if index == 0:
                    layer = nn.Linear(in_features=input_size, out_features=layer_size)
                else:
                    layer = nn.Linear(in_features=layer_sizes[index - 1],
                                      out_features=layer_size)
                layers += [layer, activation_function]
            layers += [nn.Linear(in_features=layer_sizes[-1],
                                 out_features=output_size),
                       activation_function]
            self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleCNN(nn.Module):
    """
    Simple CNN Model
    """

    def __init__(self):
        super().__init__()
        activation = ('relu', nn.ReLU())
        pool = ('pool', nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)),
            activation, pool,
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
            activation, pool
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(in_features=16*5*5, out_features=120)), activation,
            ('fc2', nn.Linear(in_features=120, out_features=80)), activation,
            ('fc3', nn.Linear(in_features=80, out_features=10)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_layer(x))

