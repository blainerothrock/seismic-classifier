import pytest, os, torch
from torch.utils.data import DataLoader
from seisml.datasets import MarsInsight, mars_insight_transform
from seisml.networks import ConvAutoEncoder
from torchsummary import summary
from scipy import stats
import numpy as np

class TestConvAutoEncoder:

    def test_configuration(self):

        # (batch_size, channel, samples)
        sample = torch.rand(16, 3, 200)

        model = ConvAutoEncoder(embedding_size=10, sample_size=200, num_conv_layers=5, kernels=2, dims=(3, 8, 16, 16, 32, 32))
        model.eval()
        X, _ = model(sample)
        assert X.shape == sample.shape, 'model should produce the same size output'
        summary(model, sample)

        model = ConvAutoEncoder(embedding_size=10, sample_size=200, num_conv_layers=1, kernels=2, dims=(3, 8))
        model.eval()
        X, _ = model(sample)
        assert X.shape == sample.shape, 'model should produce the same size output'

        model = ConvAutoEncoder(embedding_size=10, sample_size=200, num_conv_layers=2, kernels=(2, 4), dims=(3, 8, 16))
        model.eval()
        X, _ = model(sample)
        assert X.shape == sample.shape, 'model should produce the same size output'

    def test_learning(self):
        epochs = 300

        ones = torch.rand(256, 3, 50)
        zeros = torch.zeros(256, 3, 50)
        sample = torch.cat([ones, zeros])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ConvAutoEncoder(embedding_size=10, sample_size=50, num_conv_layers=3, kernels=2, dims=(3, 6, 12, 24)).to(device)
        model.train()

        summary(model, sample)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        loss_history = []
        for epoch in range(epochs):
            X = sample.to(device)
            optimizer.zero_grad()

            output, _ = model(X)
            loss = loss_fn(output, X)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.detach().item())

        slope, _, _, _, _ = stats.linregress(np.arange(epochs), loss_history)
        assert slope <= 0, 'should be learning'