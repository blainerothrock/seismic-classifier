import pytest
import torch
import os
from torch.utils.data import DataLoader
from seisml.datasets import MarsInsight, mars_insight_transform
from seisml.utility.download_data import DownloadableData, download_data
from seisml.utility.utils import split_dataset


class TestMarsInsight:

    def test_get_item(self):
        data_dir = download_data(DownloadableData.MARS_INSIGHT_SAMPLE)
        ds = MarsInsight(
            file_path=os.path.join(data_dir, 'VDS.h5'),
            transform=mars_insight_transform()
        )
        dl, _= split_dataset(ds, training_split=1.0, shuffle=True)
        sample = next(iter(dl))

        print('data length', len(ds))

        assert isinstance(sample[0], torch.Tensor), 'data output should be tensor'
        assert len(sample[0]) == 3, 'should have 3 channels'
        assert torch.mean(sample[0][0]).isclose(torch.Tensor([0]), atol=1e-4), 'mean should be removed'
        assert torch.mean(sample[0][1]).isclose(torch.Tensor([0]), atol=1e-4), 'mean should be removed'
        assert torch.mean(sample[0][2]).isclose(torch.Tensor([0]), atol=1e-4), 'mean should be removed'
