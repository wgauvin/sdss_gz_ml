#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import csv
import logging


# In[2]:


get_ipython().run_line_magic('ls', '-l data/')


# In[3]:


input_data = pd.read_csv('data/input.csv')


# In[26]:


fields = input_data[['run', 'rerun', 'camcol', 'field']].drop_duplicates().reset_index(drop=True)
fields


# In[29]:


fields.itertuples()


# In[92]:


bands = ['u', 'g', 'r', 'i', 'z']
base_url = 'http://dr15.sdss.org/sas/dr15/eboss'
# photoObj/frames/<rerun>/<run>/<camcol>/frame-{band}-<run-06d}-<camcol>-<field-04d}.fits.bz
url_format = '{}/photoObj/frames/{}/{}/{}/frame-{}-{:06d}-{}-{:04d}.fits.bz2'

urls = []
for index, row in fields[0:5].iterrows():
    for band in bands:
        run = row.run
        rerun = row.rerun
        camcol = row.camcol
        field = row.field
        
        url = url_format.format(base_url, rerun, run, camcol, band, run, camcol, field)
        urls.append(url)


# In[104]:


import asyncio
import aiohttp
import aiofiles
import random
import async_timeout
import os
from aiostream import stream, pipe


# In[7]:


urls


# In[121]:


async def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

async def generate_urls(record):
    (session, row) = record
    for band in bands:
        run = row.run
        rerun = row.rerun
        camcol = row.camcol
        field = row.field
        
        url = url_format.format(base_url, rerun, run, camcol, band, run, camcol, field)
        out_dir = 'fits/{}/{}/{}'.format(run, camcol, field)
        out_filename = '{}/frame-{}-{:06d}-{}-{:04d}.fits.bz2'.format(out_dir, band, run, camcol, field)
        yield { 'session': session, 'url': url, 'row': row, 'out_filename': out_filename }

async def handle_exception(url, e):
    print('Error retrieving {}'.format(url), e)
    return { 'error': e, 'url': url }

async def handle_http_error(url, response):
    print('Received error code {} while retrieving {}'.format(response.status, url))
    return { 'error': remotefile.status, 'url': url }

async def ignore_existing_file(record):
    return not os.path.exists(record['out_filename'])

async def retrieve_file(record):
    session = record['session']
    url = record['url']
    out_filename = record['out_filename']
    await ensure_dir(out_filename)

    print('Retriving url {}'.format(url))
    f = await aiofiles.open(out_filename, mode='wb')
    try:
        with async_timeout.timeout(120):
            async with session.get(url) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        await f.write(data)

                    print('Finished retriving {}'.format(out_filename))
                    return record
                else:
                    return await handle_http_error(url, response)
    except Exception as e:
        logging.exception(e)
        return await handle_exception(url, e)
    finally:
        await f.close()

async def download_files(urls):
    async with aiohttp.ClientSession() as session:
        await (stream.repeat(session)
           | pipe.zip(stream.iterate(fields.itertuples()))
           | pipe.flatmap(generate_urls)
           | pipe.filter(ignore_existing_file)
           | pipe.take(1000)
           | pipe.map(retrieve_file, ordered=False, task_limit=10))

def finished_downloading(_):
    end = time.time()
    print('Process took {:0.3f} seconds'.format(end - start))


# In[122]:


import time

start = time.time()

loop = asyncio.get_event_loop()
future = loop.create_task(download_files(urls))
future.add_done_callback(finished_downloading)


# In[ ]:




