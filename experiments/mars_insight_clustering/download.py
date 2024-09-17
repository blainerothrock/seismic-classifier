import csv, json, os, sys
import requests
from time import sleep
import gin

import concurrent.futures

import obspy
from obspy import read

from obspy.core.event import read_events
from obspy import read_inventory
from datetime import datetime, timedelta
from obspy.core import UTCDateTime

@gin.configurable()
def download_data_availability(data_path):
    data_path = os.path.expanduser(data_path)
    try:
        os.makedirs(data_path)
    except FileExistsError:
        pass

    payload = {
        'option': 'com_ajax',
        'data': 'ELYSE',
        'format': 'json',
        'module': 'seis_data_available'
    }
    r = requests.post('https://www.seis-insight.eu/en/science/seis-data/seis-data-availability', payload)
    with open(os.path.join(data_path, 'data_availability.json'), 'wb') as f:
        f.write(r.content)

@gin.configurable()
def split_availability(path, network, location, channel):
    path = os.path.expanduser(path)
    with open(os.path.join(path, 'data_availability.json'), 'r') as f:
        raw_ava = json.load(f)

    ava = []
    for t in raw_ava['data']:
        if t['network'] == network and t['location'] == location and t['channel'].startswith(channel):
            ava.append(t)

    with open(os.path.join(path, 'catelog.json'), 'w') as f:
        json.dump(ava, f)

    return ava


@gin.configurable(blacklist=['event'])
def download_mseed(event, channel, data_path):

    data_path = os.path.expanduser(data_path)

    try:
        os.makedirs(data_path)
    except FileExistsError:
        pass

    payload = {
        'network': event['network'],
        'station': event['station'],
        'startTime': event['startTime'],
        'endTime': event['endTime'],
        'location': event['location'],
        'channel': channel
    }

    req = requests.get('http://ws.ipgp.fr/fdsnws/dataselect/1/query', params=payload)
    file_name = '-'.join(
        [event['network'], event['station'], event['location'], event['startTime'], event['endTime']]) + '.mseed'
    print(f'{os.getpid()}: downloading: {file_name}')
    path = os.path.join(data_path, file_name)
    with open(path, 'wb') as c:
        c.write(req.content)
    return path


if __name__ == '__main__':
    gin.parse_config_file('config_download.gin')
    download_data_availability()
    ava = split_availability()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for event in ava:
            executor.submit(download_mseed, event)