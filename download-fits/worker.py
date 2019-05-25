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

from aiostream import stream, pipe

PROJECT = 'astroml-ast80014'
SUBSCRIPTION_NAME = 'get-fits-image'
SUBSCRIPTION = f'projects/{PROJECT}/subscriptions/{SUBSCRIPTION_NAME}'
BUCKET = 'wgauvin-astroml-ast80014'

async def move_file_to_gcs(msg):
    body = json.loads(msg.data.decode('utf-8'))
    file = body['filename']
    out_dir = body['out_dir']
    print(f'Moving {file} to gs://{BUCKET}/{out_dir}/')

    cmd = f'gsutil mv /app/{file} gs://{BUCKET}/{out_dir}/ > /dev/null 2>&1'
    sp.run(cmd, shell=True, text=True, check=True)

    return msg

async def retrieve_file(msg):
    body = json.loads(msg.data.decode('utf-8'))

    async with aiohttp.ClientSession() as session:
        url = body['url']
        filename = body['filename']

        print(f'Retrieving {url}')
        f = await aiofiles.open(filename, mode='wb')
        try:
            with async_timeout.timeout(360):
                async with session.get(url) as response:
                    if response.status == 200:
                        async for data in response.content.iter_any():
                            await f.write(data)

                        print(f'Finished retriving {filename}')
                        return msg
                    else:
                        print(f'Received error code {response.status} while retrieving {url}')
                        raise ValueError(f'Received error code {response.status} while retrieving {url}')
        except Exception as e:
            logging.exception(e)
            raise
        finally:
            await f.close()

async def handle_message_async(msg):
    await (stream.just(msg)
        | pipe.action(retrieve_file)
        | pipe.action(move_file_to_gcs))

def finish_handle_msg(msg, start):
    end = time.time()
    print(f'Finished handling message. Process took {end - start:0.3f} seconds')
    msg.ack()

def handle_message(msg):
    start = time.time()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        future = loop.create_task(handle_message_async(msg))
        future.add_done_callback(lambda x: finish_handle_msg(msg, start))
        loop.run_until_complete(future)
    finally:
        loop.close()

def main():
    subscriber = pubsub.SubscriberClient()
    flow_control = pubsub.types.FlowControl(max_messages=10)
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
