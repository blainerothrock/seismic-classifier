import os
import gin
import obspy
import torch
import numpy as np
import h5py
from scipy import signal
import numpy as np
from torch.utils.data import Dataset
from seisml.core.transforms import DetrendFilter, DetrendType, ToTensor, Compose
from seisml.utility.download_data import download_and_verify, DownloadableData, downloadable_data_path

def mars_insight_transform():
    transforms = [
        ToTensor()
    ]

    return Compose(transforms)

# TODO: add download option
@gin.configurable()
class MarsInsight(Dataset):
    """
        unsupervised data set for Seismic data from the Mars Insight Lander. NOTE: data must be downloaded and prepared
        before running this dataset.
    :param data_dir: str: absolute path of the split data (indivisual mseed examples)
    :param transform: seis.core.Transform: transforms to be completed on each individial sample
    """

    def __init__(self, file_path, transform=mars_insight_transform()):
        self.file_path = os.path.expanduser(file_path)
        self.transform = transform

        self.file = h5py.File(self.file_path, 'r')
        self.data = self.file['seismic_data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # stream = np.read(self.files[i])
        # ts = []
        # for trace in stream:
        #     d = {self.transform.source: trace}
        #     self.transform(d)
        #     ts.append(d[self.transform.output])
        #
        # return torch.stack(ts)

        arr = self.data[i]

        for i, _ in enumerate(arr):
            arr[i] -= arr[i].mean()
            arr[i] /= arr[i].std() + 1e-6

        arr = torch.Tensor(arr)
        return arr