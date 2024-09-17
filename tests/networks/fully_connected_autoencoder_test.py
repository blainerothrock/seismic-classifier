import pytest, os, torch
from torch.utils.data import DataLoader
from seisml.datasets import MarsInsight, mars_insight_transform
from seisml.networks import FCAutoEncoder
from torchsummary import summary
from scipy import stats
import numpy as np

class TestFCAutoEncoder:

    def test_configuration(self):

        # sample (batch_size, size)
        sample = torch.rand(16, 128)

        layers = [128, 64, 32, 16]
        model = FCAutoEncoder(layers)
        model.eval()
        output, embedding = model(sample)
        assert output.shape == sample.shape, 'model should produce the same size output'
        assert embedding.shape[-1] == layers[-1], 'latent representation should match config'

        summary(model, sample)

    def test_learning(self):
        epochs = 300

        ones = torch.rand(256, 128)
        zeros = torch.zeros(256, 128)
        sample = torch.cat([ones, zeros])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        layers = [128, 64, 32, 16]
        model = FCAutoEncoder(layers).to(device)
        model.train()

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

