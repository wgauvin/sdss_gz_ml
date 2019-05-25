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

import sdss_gz_data as sgd

from aiostream import stream, pipe
from google.cloud import pubsub

PROJECT = 'astroml-ast80014'
TOPIC_NAME = 'cutout-fits-image'
TOPIC = f'projects/{PROJECT}/topics/{TOPIC_NAME}'
BUCKET = 'wgauvin-astroml-ast80014'

def get_data(path='data/astromonical_data.csv.gz'):
    df = sgd.load_data(path)
    print(f'Length of data: {len(df)}')
    df = sgd.prepare_data(df)
    print(f'Length after prepared data: {len(df)}')

    df = df[df.galaxy_type != sgd.UNKNOWN_GALAXY_TYPE]
    print(f'Length of known galaxy types: {len(df)}')

    data = df[['objid','run','rerun','camcol','field','obj','ra','dec','petroRad_r']]
    fields = data[['run','camcol','field']].drop_duplicates().reset_index(drop=True)
    return fields, data

def generate_message(row, data):
    message = {
        'run': row.run,
        'camcol': row.camcol,
        'field': row.field
    }

    objects = [None] * len(data)

    for idx, obj in enumerate(data.itertuples()):
        curr_obj = {
            'objid': obj.objid,
            'ra': obj.ra,
            'dec': obj.dec,
            'petroRad_r': obj.petroRad_r
        }
        objects[idx] = curr_obj

    message['objects'] = objects

    return message

def publish_download_file_messages(fields, data):
    global total
    total = len(fields)
    print(f'Total number of messages to send: {total}')
    for row in fields.itertuples():
        selector = np.all([
            data.run == row.run,
            data.camcol == row.camcol,
            data.field == row.field
        ], axis=0)
        
        curr_field_data = data[selector]

        message = generate_message(row, curr_field_data)
        publish_message(message)

publisher = pubsub.PublisherClient()
count = 0
total = 0

def publish_message(record):
    global count
    count += 1
    if count % 100 == 0:
        print(f'Sent {count} of {total}')

    msg = json.dumps(record).encode("utf-8")
#    print(f'Sending message: {msg}')
    publisher.publish(TOPIC, msg)

def main():
    fields, data = get_data()
    publish_download_file_messages(fields, data)

if __name__ == '__main__':
    main()
