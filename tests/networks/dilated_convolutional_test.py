import pytest, os
from seisml.networks.dilated_convolutional import DilatedConvolutional
from seisml.datasets import TriggeredEarthquake, SiameseDataset
from seisml.utility.download_data import DownloadableData
from seisml.metrics.loss import DeepClusteringLoss
import torch
from torch.utils.data import DataLoader
from torchsummary import summary


class TestDilatedConvolutional:

    def test_paper_configuration(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )

        dl = DataLoader(ds, batch_size=1, num_workers=1, shuffle=True)

        model = DilatedConvolutional(embedding_size=10)
        data, label = next(iter(dl))
        summary(model, data.unsqueeze(1))

        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        assert num_params == 7440, 'number of params should match papers description'

    def test_learning(self):
        embedding_size = 10

        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )

        dl = DataLoader(ds, batch_size=32, num_workers=1, shuffle=True)
        sample_batch = next(iter(dl))

        model = DilatedConvolutional(embedding_size=embedding_size, downsample=False)
        test_data, test_label = next(iter(dl))
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(params, lr=0.01)
        l = DeepClusteringLoss()

        test_data = test_data.view(-1, 1, test_data.shape[-1])
        embedding_a = model(test_data)
        assert len(embedding_a[-1]) == embedding_size, 'output should match embedding size'
        _loss_a = l(embedding_a, test_label.float())

        for _ in range(10):
            data, label = sample_batch
            data = data.view(-1, 1, data.shape[-1])
            output = model(data)
            _loss = l(output, label.float())
            _loss.backward()
            opt.step()


        embedding_b = model(test_data)
        _loss_b = l(embedding_b, test_label.float())

        assert _loss_b < _loss_a, 'the model should learn something'

    def test_siamese_learning(self):
        embedding_size = 10

        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )
        ds = SiameseDataset(ds)

        dl = DataLoader(ds, batch_size=24, num_workers=1, shuffle=True)

        model = DilatedConvolutional(embedding_size=embedding_size, downsample=False)
        test_data, test_label = next(iter(dl))
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(params, lr=0.01)
        l = DeepClusteringLoss()

        test_data = test_data.view(-1, 1, test_data.shape[-1])
        embedding_a = model(test_data)
        assert len(embedding_a[-1]) == embedding_size, 'output should match embedding size'
        _loss_a = l(embedding_a, test_label.float())

        for _ in range(4):
            for data, label in dl:
                data = data.view(-1, 1, data.shape[-1])
                output = model(data)
                _loss = l(output, label.float())
                _loss.backward()
                opt.step()

        embedding_b = model(test_data)
        _loss_b = l(embedding_b, test_label.float())
