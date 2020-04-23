from torch import nn
import torch.nn.functional as F
import torch


class DilatedConvolutional(nn.Module):
    """[summary]

        Args:
            embedding_size ([type]): [description]
    """

    def __init__(self, embedding_size, num_layers=10, num_channels=16, downsample=False):
        super(DilatedConvolutional, self).__init__()

        self.num_layers = num_layers
        self.main = nn.Sequential()
        if downsample:
            downsample_amount = 3
            downsample = nn.Sequential()
            for i in range(1, downsample_amount + 1):
                downsample.add_module('ds_conv{}'.format(i), nn.Conv1d(1, 1, 2, 1))
                downsample.add_module('ds_bn{}'.format(i), nn.BatchNorm1d(1))
                downsample.add_module('ds_relu{}'.format(i), nn.ReLU())
                downsample.add_module('ds_maxpool{}'.format(i), nn.MaxPool1d(2))

            self.main.add_module('downsample', downsample)

        self.main.add_module(
            'init_conv',
            nn.Conv1d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=3,
                bias=False
            )
        )
        self.main.add_module(
            'init_bn',
            nn.BatchNorm1d(num_channels)
        )

        self.main.add_module('init_relu', nn.ReLU())

        for i in range(1, self.num_layers):

            self.main.add_module(
                str(i),
                nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    bias=False
                )
            )
            self.main.add_module(str(i) + '_bn2', nn.BatchNorm1d(num_channels))
            # self.main.add_module(str(i) + '_in2', nn.InstanceNorm1d(num_channels))
            self.main.add_module(str(i) + '_relu', nn.ReLU())
            self.main.add_module(str(i) + '_maxpool', nn.MaxPool1d(2))

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, embedding_size, bias=False)

    def forward(self, input):
        output = self.main(input)
        pools = self.max_pool(output).squeeze(-1)
        output = nn.functional.tanh(self.fc(pools))
        # output = nn.functional.normalize(output, dim=-1,p=2)
        return output
