import os
import gin
import concurrent.futures

import obspy
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def split_stream(file, raw_dir, starttime, endtime, save_dir, stride, length):

    stream = obspy.read(os.path.join(raw_dir, file))

    # merge to get 3 traces (BHU, BHW, BHV), fill gaps with 0
    stream = stream.merge(method=0, fill_value=0)

    hd5_path = os.path.join(save_dir, f'seis_{str(stream[0].stats.starttime)}_{str(stream[0].stats.endtime)}.hd5')

    # only include streams with 3 channels
    if len(stream) != 3:
        return

    # common start
    starts = [t.stats.starttime for t in stream]

    max_start = max(starts)
    if max_start > endtime:
        return

    start = max_start if max_start >= starttime else starttime

    # common end
    ends = [t.stats.endtime for t in stream]

    max_end = max(ends)
    if max_end < starttime:
        return

    end = max_end if max_end < endtime else endtime

    # trim to common start and end times
    stream.trim(start, end)
    splits = list(stream.slide(
        window_length=length - (1/20), # offset by freq to get a even num of samples
        step=stride,
        include_partial_windows=False
    ))

    def acceptable(stream):
        trace_sizes = list(map(lambda x: float(len(x.data)), stream))
        trace_channels = list(map(lambda x: x.meta.channel, stream))
        return len(set(trace_channels)) == 3 and set(trace_sizes) and \
                set(trace_sizes) == {length * stream[0].stats.sampling_rate}

    splits = list(filter(acceptable, splits))

    channels = ['BHU', 'BHW', 'BHV']
    def map_order_traces(stream):
        data = np.zeros((3, int(length * stream[0].stats.sampling_rate)))
        for trace in stream:
            data[channels.index(trace.stats.channel)] = trace.data
        return data

    splits = np.array(list(map(map_order_traces, splits)))

    print(hd5_path)
    f = h5py.File(hd5_path, 'w')
    f.create_dataset('seismic_data', (len(splits), 3, int(length * stream[0].stats.sampling_rate)), data=splits)

    return hd5_path

@gin.configurable()
def split_data(raw_dir, save_dir, stride=3, length=12, starttime=None, endtime=None):
    raw_dir = os.path.expanduser(raw_dir)
    save_dir = os.path.expanduser(save_dir)

    starttime = obspy.UTCDateTime(starttime)
    endtime = obspy.UTCDateTime(endtime)

    file_names = tuple(filter(lambda f: os.path.splitext(f)[1] == '.mseed', os.listdir(raw_dir)))
    file_count = len(file_names)
    print(f'num of files: {file_count}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pbar_split = tqdm(total=len(file_names))

    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures_to_file = {
            executor.submit(
                split_stream,
                file,
                raw_dir,
                starttime,
                endtime,
                save_dir,
                stride,
                length
            ): file for file in file_names
        }

        paths = []
        for future in concurrent.futures.as_completed(futures_to_file):
            pbar_split.update(1)
            try:
                data = future.result()
                if data is not None:
                    paths.append(data)
            except Exception as exc:
                print()


        entry_key = 'seismic_data'
        sources = []
        total_length = 0
        for i, file in enumerate(paths):
            with h5py.File(os.path.join(save_dir, file), 'r') as active_data:
                vsource = h5py.VirtualSource(active_data[entry_key])
                total_length += vsource.shape[0]
                sources.append(vsource)

        layout = h5py.VirtualLayout(shape=(total_length, sources[0].shape[1], sources[0].shape[2]), dtype=np.float)

        offset = 0
        for vsource in sources:
            length = vsource.shape[0]
            layout[offset: offset + length] = vsource
            offset += length

        with h5py.File(os.path.join(save_dir, 'VDS.h5'), 'w') as f:
            f.create_virtual_dataset(entry_key, layout, fillvalue=0)



if __name__ == '__main__':
    gin.parse_config_file('config_prepare.gin')
    split_data()
