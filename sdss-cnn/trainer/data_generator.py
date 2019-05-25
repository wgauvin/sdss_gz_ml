import numpy as np
import keras

from image_generator import ImageAugmentor
from PIL import Image

from sdss_gz_data import get_gcs_bucket
from keras.preprocessing.image import ImageDataGenerator

from astropy.io import fits

IMAGE_SIZE = 42

bucket = get_gcs_bucket()
image_augmentor = ImageAugmentor(0.1)

def fill_nan(data):
    nans = np.any([np.isnan(data), data == -100])
    data[nans] = 0
    data[nans] = np.min(data)

    return data, nans

def download_img(record, verbose=False):
    def _get_data(hdu, filename):
        data = hdu.data
        data, nans = fill_nan(data)

        if nans.any():
            hdu.data = data
            hdu.writeto(filename, overwrite=True)

        return data

    import os

    run = record.run
    camcol = record.camcol
    field = record.field
    objid = record.objid
    
    blob_dir = f'fits/{run}/{camcol}/{field}'
    filename = f'obj-{objid}.fits.bz2'
    blob_path = f'{blob_dir}/{filename}'
    
    if verbose:
        print(f'Downloading {blob_path}')
    
    if not os.path.isfile(filename):
        blob = bucket.get_blob(blob_path)

        try:
            if not blob.exists():
                inv_blob_path = f'${blob_dir}//tmp/{record.run}-{record.camcol}-{record.field}/obj-{record.objid}.fits.bz2'
                blob = bucket.get_blob(inv_blob_path)

            blob.download_to_filename(filename)
        except Exception as e:
            print(f'Error getting blob {blob_path}')
            raise e

    with fits.open(filename) as fits_file:
        return _get_data(fits_file[0], filename)

def augment_image(data):
    return image_augmentor.augment_image(data)

def crop_image(data, image_size):
    top_left = (72 - image_size)/2
    bottom_right = top_left + image_size
    
    output_data = np.zeros((3, image_size, image_size))
    
    for idx in range(3):
        img = Image.fromarray(data[idx])
        img = img.crop((top_left, top_left, bottom_right, bottom_right))
        output_data[idx] = np.array(img)
    
    return np.moveaxis(output_data, 0, -1)

def get_image(record, image_size, augment=False):
    data = download_img(record)
    if augment:
        data = augment_image(data)
    return crop_image(data, image_size)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 X,
                 y,
                 features,
                 feature_scaler,
                 image_min_val,
                 image_range,
                 batch_size=32,
                 shuffle=True,
                 random_state=42,
                 augment=True
                ):
        'Initialization'

        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.random_state = random_state
        self.shuffle = shuffle
        self.features = features
        self.features_length = len(features)
        self.listOfIndexes = X.index.values
        self.feature_scaler = feature_scaler
        self.augment = augment
        self.image_min_val = image_min_val
        self.image_range = image_range

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        length = int(np.floor(len(self.X) / self.batch_size))
        return length

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.listOfIndexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __scale_images(self, images):
        return (images - self.image_min_val)/self.image_range

    def __data_generation(self, indexes):
        X_images = np.empty((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
        X_features = np.empty((self.batch_size, self.features_length), dtype=float)
        y_galaxy_morph = np.empty((self.batch_size), dtype=float)
        y_redshift = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, index in enumerate(indexes):
            pandas_index = self.listOfIndexes[index]
            record = self.X.loc[pandas_index]
            try:
                # Inputs
                X_images[i,] = get_image(record, IMAGE_SIZE, augment=self.augment)
                X_features[i,] = record[self.features]

                # Outputs
                y_galaxy_morph[i] = self.y.loc[pandas_index]['galaxy_type']
                y_redshift[i] = self.y.loc[pandas_index]['z']
            except Exception as e:
                print(f'Error in processing record {record}. Removing from data', e)
                raise e

        X = {
            'input_1': self.__scale_images(X_images),
            'input_2': self.feature_scaler.transform(X_features)
        }

        y = {
            'output_1': y_galaxy_morph,
            'output_2': y_redshift
        }

        return X, y

    def get_source_data(self):
        data = self.X.copy()
        data['galaxy_type'] = self.y['galaxy_type']
        data['z'] = self.y['z']

        return data
