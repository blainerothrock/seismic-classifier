# from torch.utils.data import Dataset
# import os
# import pickle
# import numpy as np
# import random
#
# from seisml.core.transforms import ButterworthPassFilter
#
# class EarthquakeDataset(Dataset):
#     def __init__(self, folder, transforms='demean', length=20000, split='', augmentations='',
#                  filter_labels=('negative', 'positive'), mode='train'):
#         self.folder = folder
#         self.files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.p' in x])
#         random.seed(0)
#         random.shuffle(self.files)
#
#         filter_labels = list(filter_labels)
#         for fname in self.files:
#             label = fname.split('/')[-1].split('_')[0]
#             if label not in filter_labels:
#                 self.files.remove(fname)
#         self.labels = sorted(filter_labels)
#         self.length = length
#         self.transforms = transforms.split(':')
#         self.augmentations = augmentations.split(':')
#         self.split = split.split(':')
#         self.split, self.files = self.filter_files(split)
#         self.priors = self.get_prior()
#         self.mode = mode
#
#     def get_prior(self):
#         labels = np.zeros(len(self.labels))
#         for i in range(len(self.files)):
#             with open(self.files[i], 'rb') as f:
#                 earthquake = pickle.load(f)
#             label = earthquake['label']
#             index = self.labels.index(label)
#             one_hot = np.zeros(len(self.labels))
#             one_hot[index] = 1
#             labels += one_hot
#         priors = labels / np.sum(labels)
#         return priors
#
#     def __getitem__(self, i):
#         with open(self.files[i], 'rb') as f:
#             earthquake = pickle.load(f)
#         label = earthquake['label']
#         sac = earthquake['data']
#         index = self.labels.index(label)
#         one_hot = np.zeros(len(self.labels))
#         one_hot[index] = 1
#
#         sacs = self.pre_transform(sac, self.transforms)
#         data = [self.get_surface_window(s) for s in sacs]
#         data = np.stack(data, axis=0)
#
#         data = self.augment(data, self.augmentations)
#         data = self.get_target_length_and_transpose(data, self.length)
#         weight = 1.  # / self.priors[index]
#
#         data = self.post_transform(data, self.transforms)
#
#         return {'data': data, 'label': one_hot, 'weight': np.sqrt(weight)}
#
#     def __len__(self):
#         return len(self.files)
#
#     def get_surface_window(self, sac):
#         # velocities = [5.0, 2]
#         # velocities = [5.0, 2.5]
#
#         # start = int(sac.stats.sac['dist'] / velocities[0])
#         # stop = int(sac.stats.sac['dist'] / velocities[1])
#
#         #        t = sac.stats.starttime
#         #        sac.trim(t + start, t + stop, nearest_sample=False)
#         return sac.data
#
#     def toggle_split(self):
#         tmp = self.files
#         self.files = self.split
#         self.split = tmp
#
#     def filter_files(self, split):
#         split_files = []
#         remaining_files = []
#         for i, fname in enumerate(self.files):
#             with open(fname, 'rb') as f:
#                 earthquake = pickle.load(f)
#             # hack to skip noisy earthquake
#             if earthquake['name'] not in 'SAC_20010126_XF_prem':
#                 if earthquake['name'] in split:
#                     split_files.append(fname)
#                 else:
#                     remaining_files.append(fname)
#         return split_files, remaining_files
#
#     # TODO: move to util method
#     def get_target_length_and_transpose(self, data, target_length):
#         length = data.shape[-1]
#         if target_length == 'full':
#             target_length = length
#         if length > target_length:
#             # offset = int(data.argmax(axis=-1))
#             # data = np.pad(data, ((0, 0), (int(target_length / 2), int(target_length / 2))), mode='constant')
#             if self.mode == 'train':
#                 offset = np.random.randint(0, length - target_length)
#             else:
#                 offset = 0
#         else:
#             offset = 0
#         pad_length = max(target_length - length, 0)
#         pad_tuple = [(0, 0) for k in range(len(data.shape))]
#         pad_tuple[1] = (int(pad_length / 2), int(pad_length / 2) + (length % 2))
#         data = np.pad(data, pad_tuple, mode='constant')
#         data = data[:, offset:offset + target_length]
#         return data
#
#     @staticmethod
#     def pre_transform(sac, transforms):
#         data = {'raw': sac}
#         if 'demean' in transforms:
#             data['raw'].detrend(type='demean')
#         # if 'raw' in transforms:
#         #     data.append(sac)
#
#         tfm = ButterworthPassFilter('bandpass', min_freq=2.0, max_freq=8.0, zerophase=True)
#         data = tfm(data)
#
#         # if 'bandpass' in transforms:
#         #     sac_copy = copy.deepcopy(sac)
#         #     sac_copy.filter('bandpass', freqmin=2, freqmax=8, corners=2, zerophase=True)
#         #     data.append(sac_copy)
#         # if 'highpass' in transforms:
#         #     sac_copy = copy.deepcopy(sac)
#         #     sac_copy.filter('highpass', freq=2)
#         #     data.append(sac_copy)
#         # if 'lowpass' in transforms:
#         #     sac_copy = copy.deepcopy(sac)
#         #     sac_copy.filter('lowpass', freq=2)
#         #     data.append(sac_copy)
#         return [data['bandpass']]
#
#     @staticmethod
#     def post_transform(data, transforms):
#         if 'whiten' in transforms:
#             data -= data.mean()
#             data /= data.std() + 1e-6
#         return data
#
#     @staticmethod
#     def augment(data, augmentations):
#         coin_flip = np.random.random()
#         if coin_flip > 0.5:
#             if 'amplitude' in augmentations:
#                 start_gain, end_gain = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
#                 amplitude_mod = np.linspace(start_gain, end_gain, num=data.shape[-1])
#                 data *= amplitude_mod
#
#             if 'noise' in augmentations:
#                 std = data.std() * np.random.uniform(1, 2)
#                 mean = data.mean()
#                 noise = np.random.normal(loc=mean, scale=std, size=data.shape)
#                 data += noise
#
#         return data