# worker.py
from google.cloud import pubsub

import time
import asyncio
import aiohttp
import aiofiles
import random
import async_timeout
import logging
import json
import subprocess as sp
import numpy as np

from PIL import Image
from aiostream import stream, pipe

from astropy.io import fits
from sdss_gz_data import scale_rgb
import sdss_gz_data as sgd

PROJECT = 'astroml-ast80014'
SUBSCRIPTION_NAME = 'cutout-fits-image'
SUBSCRIPTION = f'projects/{PROJECT}/subscriptions/{SUBSCRIPTION_NAME}'
BUCKET = 'wgauvin-astroml-ast80014'
DATA_DIR = '/tmp'
SIZING_RATIO = 7.8567420132
CUTOUT_RATIO = 2 * SIZING_RATIO
CUTOUT_PIXEL_SIZE = 72
JPEG_SIZE = 144

TOP_COORD = int(CUTOUT_PIXEL_SIZE / 4)
BTTM_COORD = int(TOP_COORD + CUTOUT_PIXEL_SIZE / 2)

bucket = sgd.get_gcs_bucket()

async def cleanup(msg):
    import shutil

    shutil.rmtree(msg['workdir'], ignore_errors=True)

async def move_file_to_gcs(file, out_dir):
    print(f'Moving {file} to gs://{BUCKET}/{out_dir}/')

    blob_path = f'{out_dir}/{file}'
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file)

async def move_files_to_gcs(msg):
    run = msg['run']
    camcol = msg['camcol']
    field = msg['field']
    objid = msg['objid']
    fits_blob = f'fits/{run}/{camcol}/{field}/obj-{objid}.fits.bz2'
    jpeg_blob = f'fits/{run}/{camcol}/{field}/obj-{objid}.jpeg'

    await asyncio.gather(
        move_file_to_gcs(msg['cutout_file'], fits_blob),
        move_file_to_gcs(msg['jpeg_file'], jpeg_blob)
    )

    return msg

async def save_fits(data, wcs, filename):
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=True)

async def save_cutout(cutout, filename):
    await save_fits(cutout.data, cutout.wcs, filename)

async def get_object_msgs(msg):
    for obj in msg['objects']:
        obj_msg = {
            'run': msg['run'],
            'camcol': msg['camcol'],
            'field': msg['field'],
            'objid': obj['objid'],
            'ra': obj['ra'],
            'dec': obj['dec'],
            'petroRad_r': obj['petroRad_r'],
            'filenames': msg['filenames'],
            'workdir': msg['workdir']
        }
        yield obj_msg

async def create_jpeg(msg):
    objid = msg['objid']
    workdir = msg['workdir']

    filename = f'{workdir}/obj-{objid}.jpeg'
    
    with fits.open(msg['fits_image']) as fits_file:
        fits_file = fits.open(msg['fits_image'])
        data = fits_file[0].data
    
        data = data[:, TOP_COORD:BTTM_COORD, TOP_COORD:BTTM_COORD]
        jpeg_data = scale_rgb(data)

        image = Image.fromarray(np.transpose(jpeg_data))
        image = image.resize((JPEG_SIZE, JPEG_SIZE), Image.BICUBIC)
        image = image.transpose(Image.ROTATE_90)
        image.save(filename)

        msg['jpeg_file'] = filename

async def create_data_cube(msg):
    from aplpy.rgb import make_rgb_cube

    workdir = msg['workdir']
    objid = msg['objid']

    cutout_filenames = np.empty(3, dtype=object)
    
    for idx, band in enumerate(['i', 'r', 'g']):
        cutout_filenames[idx] = get_obj_filename(workdir, objid, band)
    
    outfile = get_obj_filename(workdir, objid)
    make_rgb_cube(cutout_filenames, outfile)

    msg['fits_image'] = outfile
    return msg

async def download_img(infile, outfile, verbose=False):
    import os

    if verbose:
        print(f'Downloading {infile}')
    
    if not os.path.isfile(outfile):
        blob = bucket.get_blob(infile)
        blob.download_to_filename(outfile)

async def get_file_from_gcs(msg, idx, band, filenames):
    run = msg['run']
    camcol = msg['camcol']
    field = msg['field']
    workdir = msg['workdir']

    file_dir = f'fits/{run}/{camcol}/{field}'
    filename = f'frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2'

    infile = f'{file_dir}/{filename}'
    outfile = f'{workdir}/{filename}'

    print(f'Downloading {infile}')
    await download_img(infile, outfile)
    filenames[idx] = outfile

async def create_workdir(msg):
    import os

    run = msg['run']
    camcol = msg['camcol']
    field = msg['field']

    workdir = f'{DATA_DIR}/{run}-{camcol}-{field}'

    try:
        os.makedirs(workdir, exist_ok=True)
    finally:
        msg['workdir'] = workdir

    return msg

def fill_nan(data):
    nans = np.isnan(data)
    data = np.nan_to_num(data)
    data[nans] = np.min(data)

    return data

def resize(data, size):
    return np.array(Image.fromarray(data).resize((size, size), Image.BICUBIC))

async def cutout_fits(msg):
    from astropy.coordinates import SkyCoord, ICRS
    import astropy.units as u
    from astropy.wcs import WCS
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.nddata import Cutout2D

    run = msg['run']
    camcol = msg['camcol']
    field = msg['field']
    objid = msg['objid']
    ra = msg['ra']
    dec = msg['dec']
    petroRad_r = msg['petroRad_r']
    fits_image = msg['fits_image']

    print(f'Creating cutout for - run: {run}, camcol: {camcol}, field: {field}, objid: {objid}, ra: {ra}, dec: {dec}, petroRad_r: {petroRad_r}')

    with fits.open(fits_image, memmap=True) as fits_file:
        hdu = fits_file[0]
        data = hdu.data
        wcs = WCS(hdu.header, naxis=2)
        
        angular_size = SIZING_RATIO * petroRad_r

        position = SkyCoord(ra=ra, dec=dec, frame=ICRS, unit=u.deg)
        cutout_size = u.Quantity((angular_size, angular_size), u.arcsec)

        out_data = np.empty((3, CUTOUT_PIXEL_SIZE, CUTOUT_PIXEL_SIZE))
        cutout_wcs = WCS(naxis=3)

        for idx in range(3):
            cutout = Cutout2D(data[idx], position=position, size=cutout_size, wcs=wcs)
            if (idx == 0):
                curr_wcs = cutout.wcs.wcs
                cutout_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', '']
                cutout_wcs.wcs.cdelt = np.array([curr_wcs.cdelt[0], curr_wcs.cdelt[1], 1.0])
                cutout_wcs.wcs.crpix = [curr_wcs.crpix[0], curr_wcs.crpix[1], 0.0]
                cutout_wcs.wcs.crval = [curr_wcs.crval[0], curr_wcs.crval[1], 0.0]
            
            temp_data = fill_nan(cutout.data)
            out_data[idx] = resize(temp_data, CUTOUT_PIXEL_SIZE)

        await save_fits(out_data, cutout_wcs, fits_image)

        msg['cutout_file'] = fits_image

        del hdu, data, wcs
        return msg

def get_obj_filename(workdir, objid, band=None):
    if (band is None):
        return f'{workdir}/obj-{objid}.fits.bz2'
    else:
        return f'{workdir}/obj-{objid}-{band}.fits.bz2'

async def cutout_img_band(msg, obj_msg, idx, band, filename):
    from astropy.coordinates import SkyCoord, ICRS
    import astropy.units as u
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D

    workdir = msg['workdir']

    objid = obj_msg['objid']
    ra = obj_msg['ra']
    dec = obj_msg['dec']
    petroRad_r = obj_msg['petroRad_r']

    angular_size = CUTOUT_RATIO * petroRad_r

    position = SkyCoord(ra=ra, dec=dec, frame=ICRS, unit=u.deg)
    cutout_size = u.Quantity((angular_size, angular_size), u.arcsec)

    with fits.open(filename, memmap=True) as fits_file:
        hdu = fits_file[0]
        data = hdu.data
        header = hdu.header
        wcs = WCS(header, naxis=2)

        cutout = Cutout2D(data, position=position, size=cutout_size, wcs=wcs)
        del hdu, data, header, wcs

        cutout_filename = get_obj_filename(workdir, objid, band)
        await save_cutout(cutout, cutout_filename)

        return msg

async def process_band(msg, idx, band):
    filenames = msg['filenames']
    await get_file_from_gcs(msg, idx, band, filenames)
    
    for obj_msg in msg['objects']:
        await cutout_img_band(msg, obj_msg, idx, band, filenames[idx])

async def handle_message_async(body):
    msg = json.loads(body.data.decode('utf-8'))

    print(f'Received msg: {msg}')
    msg = await create_workdir(msg)
    filenames = np.empty(3, dtype=object)
    msg['filenames'] = filenames

    await asyncio.gather(
        process_band(msg, 0, 'i'),
        process_band(msg, 1, 'r'),
        process_band(msg, 2, 'g')
    )

    await (stream.just(msg)
        | pipe.flatmap(get_object_msgs)
        | pipe.action(create_data_cube)
        | pipe.action(cutout_fits)
        | pipe.action(create_jpeg)
        | pipe.action(move_files_to_gcs))

    await cleanup(msg)

def finish_handle_msg(msg, start, fut):
    end = time.time()
    time_diff = end - start
    if fut.exception() is None:
        print(f'Finished successfully. Process took {time_diff:0.3f} seconds')
        msg.ack()
    else:
        print(f'Finished with error. Process took {time_diff:0.3f} seconds. NAKing message', fut.exception())
        msg.nack()

def handle_message(msg):
    start = time.time()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        future = loop.create_task(handle_message_async(msg))
        future.add_done_callback(lambda x: finish_handle_msg(msg, start, x))
        loop.run_until_complete(future)
    finally:
        loop.close()

def main():
    subscriber = pubsub.SubscriberClient()
    flow_control = pubsub.types.FlowControl(max_messages=2)
    # Subscription is a Future.
    print('Listening to subscription for messages')
    subscription = subscriber.subscribe(
            SUBSCRIPTION,
            callback=handle_message,
            flow_control=flow_control
        )

    try:
        subscription.result()
    except Exception as ex:
        print('Error occurred with subscription')
        logging.exception(ex)
        subscription.close()
        raise

if __name__ == '__main__':
    main()
