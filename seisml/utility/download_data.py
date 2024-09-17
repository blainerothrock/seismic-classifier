import os, shutil
import tarfile
import requests
import hashlib
from enum import Enum

class DownloadableData(str, Enum):
    SAMPLE_DATA = 'triggered_earthquake_sample_data'
    TRIGGERED_EARTHQUAKE = 'triggered_earthquakes'
    MARS_INSIGHT_SAMPLE = 'mars_insight_sample'
    TRIGGERED_TREMOR_100HZ = 'triggered_tremor'
    TRIGGERED_TREMOR_SAMPLE = 'triggered_tremor_sample'
    TRIGGERED_TREMOR_20HZ = 'triggered_tremor_20hz'


def downloadable_data_path(downloadable_data):
    if downloadable_data == DownloadableData.SAMPLE_DATA:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquake/triggered_earthquake_sample_data.tar.gz'
    if downloadable_data == DownloadableData.TRIGGERED_EARTHQUAKE:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquake/triggered_earthquakes.tar.gz'
    if downloadable_data == DownloadableData.MARS_INSIGHT_SAMPLE:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/mars_insight_sample.tar.gz'
    if downloadable_data == DownloadableData.TRIGGERED_TREMOR_100HZ:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_tremor/triggered_tremor.tar.gz'
    if downloadable_data == DownloadableData.TRIGGERED_TREMOR_SAMPLE:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_tremor/triggered_tremor_sample.tar.gz'
    if downloadable_data == DownloadableData.TRIGGERED_TREMOR_20HZ:
        return 'https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_tremor/triggered_tremor_20hz.tar.gz'

DATA_PATH = os.path.expanduser('~/.seisml/data/')

def download_data(dd):
    return download_and_verify(dd.value, downloadable_data_path(dd))

def download_and_verify(name, download_path, force=False):
    print('downloading {}: '.format(name))
    if os.path.isdir(DATA_PATH + name) and not force:
        return os.path.join(DATA_PATH, name)
    if os.path.isdir(DATA_PATH + name) and force:
        print('  - removing existing {}'.format(name))
        shutil.rmtree(DATA_PATH + name)

    url = download_path
    target_path = os.path.join(DATA_PATH, '{}.tar.gz'.format(name))

    print('  - downloading')
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(res.raw.read())

        print('  - validating checksum')
        with open(target_path, 'rb') as f:
            data = f.read()
            md5_to_check = hashlib.md5(data).hexdigest()

        with open(os.path.join('seisml/utility/checksums', '{}.md5'.format(name)), 'r') as f:
            data = f.readlines()[0]
            data = data.split('  ')
            md5 = data[0]

        if md5 == md5_to_check:
            print('  - checksum verified')
        else:
            print('  - checksum invalid, aborting')
            os.remove(target_path)
            return

        print('  - extracting {}'.format(name))
        tf = tarfile.open(target_path, 'r:gz')
        tf.extractall(DATA_PATH)
        tf.close()
        os.remove(target_path)

        return os.path.join(DATA_PATH, name)

