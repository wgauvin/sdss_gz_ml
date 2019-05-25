# master.py
import numpy as np
import pandas as pd
import logging
import asyncio
import aiohttp
import aiofiles
import random
import async_timeout
import os
import json

from aiostream import stream, pipe
from google.cloud import pubsub

PROJECT = 'astroml-ast80014'
TOPIC_NAME = 'get-fits-image'
TOPIC = f'projects/{PROJECT}/topics/{TOPIC_NAME}'
bands = [ 'g', 'r', 'i']
base_url = 'http://dr15.sdss.org/sas/dr15/eboss'

def get_fields():
    df = pd.read_csv('../notebooks/data/input.csv')
    return df[['run', 'rerun', 'camcol', 'field']].drop_duplicates().reset_index(drop=True)

def generate_urls(row):
    items = []
    for band in bands:
        run = row.run
        rerun = row.rerun
        camcol = row.camcol
        field = row.field
        
        url = f'{base_url}/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2'
        out_dir = f'fits/{run}/{camcol}/{field}'
        filename = f'frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2'
        items.append({ 'url': url, 'run': run, 'rerun': rerun, 'camcol': camcol, 'field': field, 'out_dir': out_dir, 'filename': filename })

    return items

def publish_download_file_messages(fields):
    global total
    total = len(bands) * len(fields)
    for row in fields.itertuples():
        messages = generate_urls(row)
        for message in messages:
            publish_message(message)

publisher = pubsub.PublisherClient()
count = 0
total = 0

def publish_message(record):
    global count
    count += 1
    if count % 1000 == 0:
        print(f'Sent {count} of {total}')
    publisher.publish(TOPIC, json.dumps(record).encode('utf-8'))

def main():
    publish_download_file_messages(get_fields())

if __name__ == '__main__':
    main()
