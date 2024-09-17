import torch
import torch.nn as nn
import gin

# TODO: add dynamic pooling parameters

@gin.configurable()
class ConvAutoEncoder(nn.Module):
    def __init__(self, embedding_size=10, sample_size=20, num_conv_layers=3, kernels=2, dims=(3, 8, 16, 16), pooling=False):
        super(ConvAutoEncoder, self).__init__()

        self.pooling = pooling
        self.embeddings_size = embedding_size
        self.sample_size = sample_size
        self.num_conv_layers = num_conv_layers

        if isinstance(kernels, int):
            self.kernels = [kernels] * num_conv_layers
        elif isinstance(kernels, tuple) and len(kernels) == num_conv_layers:
            self.kernels = kernels
        else:
            raise ValueError('kernels must be a int or a tuple of size num_layers')

        if isinstance(dims, tuple) and len(dims) == num_conv_layers + 1:
            self.dims = dims
        else:
            raise ValueError('dims must be tuple of size num_layers + 1')

        self.encoder = _ConvEncoder(
            embedding_size=self.embeddings_size,
            sample_size=self.sample_size,
            num_conv_layers = self.num_conv_layers,
            kernels=self.kernels,
            dims=self.dims,
            pooling=self.pooling
        )

        self.decoder = _ConvDecoder(
            embedding_size=self.embeddings_size,
            projection_dim=self.encoder.projection_dim,
            num_conv_layers=self.num_conv_layers,
            kernels=self.kernels,
            dims=self.dims,
            pooling=self.pooling
        )

    def forward(self, X):
        encoding, pool_indicies, sizes = self.encoder(X)
        X = self.decoder(encoding, pool_indicies, sizes)
        return X, encoding


class _ConvEncoder(nn.Module):

    def __init__(self, embedding_size, sample_size, num_conv_layers, kernels, dims, pooling):
        super(_ConvEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.sample_size = sample_size
        self.num_conv_layers = num_conv_layers
        self.kernels = kernels
        self.dims = dims
        self.pooling = False

        self.pool_idx = []

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pools = nn.ModuleList()

        output_size = self.sample_size
        for l in range(num_conv_layers):
            self.convs.append(
                nn.Conv1d(
                    self.dims[l],
                    self.dims[l + 1],
                    kernel_size=self.kernels[l]
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.dims[l + 1]))
            self.activations.append(nn.ReLU())
            if pooling:
                self.pools.append(nn.MaxPool1d(2, return_indices=True))
            else:
                self.pools.append(Empty(returns=1))

            output_size = self._output_size(
                output_size,
                self.kernels[l],
                stride=1,
                padding=0,
                pool_factor=2 if self.pooling else 1
            )

        self.projection_dim = output_size * self.dims[-1]
        self.linear = nn.Sequential(
            nn.Linear(self.projection_dim, embedding_size),
            nn.Dropout(0.4),
            nn.Sigmoid()
        )

    def _output_size(self, in_size, conv_kernel, stride, padding, pool_factor):
        out = int((in_size - conv_kernel + (2 * padding)) / stride) + 1
        out = int(out / pool_factor)
        return out

    def forward(self, X):

        pool_indicies = []
        sizes = []
        for l in range(self.num_conv_layers):
            X = self.convs[l](X)
            X = self.activations[l](X)
            sizes.append(X.size())
            X, pool_idx = self.pools[l](X)
            pool_indicies.append(pool_idx)

        X = self.linear(X.flatten(start_dim=1))

        return X, pool_indicies, sizes


class _ConvDecoder(nn.Module):

    def __init__(self, embedding_size, projection_dim, num_conv_layers, kernels, dims, pooling):
        super(_ConvDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.projection_dim = projection_dim
        self.num_conv_layers = num_conv_layers
        self.kernels = kernels
        self.dims = dims
        self.pooling = pooling

        self.unpools = nn.ModuleList()
        self.transpose_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_size, self.projection_dim),
            nn.Dropout(0.4),
            nn.Sigmoid()
        )

        for l in range(num_conv_layers).__reversed__():
            if self.pooling:
                self.unpools.append(nn.MaxUnpool1d(2))
            else:
                self.unpools.append(Empty())

            self.transpose_convs.append(
                nn.ConvTranspose1d(
                    self.dims[l + 1],
                    self.dims[l],
                    kernel_size=self.kernels[l]
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.dims[l]))
            self.activations.append(nn.ReLU())

    def forward(self, X, pool_indices, sizes):
        pool_indices.reverse()
        sizes.reverse()

        X = self.linear(X)
        X = X.view(X.shape[0], self.dims[-1], -1)
        for l in range(self.num_conv_layers):
            X = self.unpools[l](
                X,
                pool_indices[l],
                output_size=sizes[l]
            )
            X = self.transpose_convs[l](X)
            X = self.activations[l](X)

        return X

class Empty(nn.Module):
    def __init__(self, returns=0, *args, **kwargs):
        self.returns = returns
        super(Empty, self).__init__()

    def forward(self, X, *args, **kwargs):
        if self.returns == 0:
            return X
        else:
            ret = [None for _ in range(self.returns)]
            ret.insert(0, X)
            return tuple(ret)

