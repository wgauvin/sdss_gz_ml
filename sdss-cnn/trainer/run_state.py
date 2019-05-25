import sdss_gz_data as sgd
from sdss_gz_data import IMAGE_SIZE
from sdss_gz_data import UNKNOWN_GALAXY_TYPE

from data_generator import DataGenerator

import numpy as np

from sklearn.preprocessing import StandardScaler

import pickle

N_FOLDS=5

object_cols = [
    'objid',
    'run',
    'rerun',
    'field',
    'camcol'
]

features = [
    'deVAB_i',
    'expAB_g',
    'expAB_i',
    'expAB_z',
    'expRad_g',
    'expRad_u',
    'expRad_z',
    'fiberMag_g',
    'fiberMag_u',
    'fiberMag_z',
    'model_g_u_colour_index',
    'model_i_r_colour_index',
    'model_r_g_colour_index',
    'model_z_i_colour_index',
    'petroRad_r',
    'petro_R90_R50_ratio_g',
    'petro_R90_R50_ratio_i',
    'petro_r_g_colour_index',
    'psfMag_r'    
]

all_cols = object_cols + features

def load_run_state(filename):
    with open(filename, 'rb') as file:
        run_state = pickle.load(file)

    return run_state

def get_data(file):
    print('Loading data...')
    orig_data = sgd.load_data(file)
    print('Preparing data...')
    prepared_data = sgd.prepare_data(orig_data)
    print('Transforming data...')
    transformed_data = sgd.transform_data(prepared_data)
    known_galaxy_type_idx = transformed_data.galaxy_type != UNKNOWN_GALAXY_TYPE

    transformed_data = transformed_data[known_galaxy_type_idx]

    X = transformed_data[all_cols]
    y = transformed_data[['galaxy_type','z']]

    # X = X[known_galaxy_type_idx]
    # y = y[known_galaxy_type_idx]
    print(f'Number of records: {len(X)}')

    return X, y

class RunState:
    def __init__(self,
                 test_size=0.2,
                 random_state=42,
                 use_cnn=True,
                 use_features=True,
                 batch_size=32,
                 # allow this to be set from args!
                 image_min_val=None, #-2.533323287963867,
                 image_max_val=None #238.15689086914062,
                ):
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.use_cnn = use_cnn
        self.use_features = use_features
        self.__training_prepared = False
        self.__testing_prepared = False

        self.image_min_val = image_min_val
        self.image_max_val = image_max_val

    def __getstate__(self):
        state = self.__dict__.copy()
        state['__training_prepared'] = False
        state['__test_prepared'] = False

        # Don't store the train and test data... too big
        del state['X_train']
        del state['y_train']
        del state['X_test']
        del state['y_test']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_train_data(self, X, y):
        self.X_train = X
        self.y_train = y

        self.batch_size = min(len(self.X_train), self.batch_size)
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train[features])

        self.__training_prepared = True
        self.fit_images()
            
#        self.split_train_and_validation()

    def set_test_data(self, X, y):
        self.X_test = X
        self.y_test = y
        self.__testing_prepared = True

    def scale(self, input):
        return self.scaler.transform(input[features])

    def image_input_shape(self):
        return (IMAGE_SIZE, IMAGE_SIZE, 3)

    def features_input_shape(self):
        return (len(features), )

    def get_fold_generators(self, n_splits=N_FOLDS, num_bins=24):
        from sklearn.model_selection import StratifiedKFold
        from sdss_gz_data import binnify

        y_binned = binnify(self.y_train, num_bins=12)
        folds = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            ).split(self.X_train, y_binned,)

        for train_idx, valid_idx in folds:
            yield (self.__get_data_generator(train_idx), self.__get_validate_data_generator(valid_idx))

    def split_train_and_validation(self, split_size=0.2):
        if not self.__training_prepared:
            raise Exception('Run state in not a prepared state.')

        from sdss_gz_data import binnify

        indexes = np.arange(len(self.X_train))
        np.random.shuffle(indexes)

        end_idx = int(np.round(len(self.X_train) * split_size))
        validate_index = indexes[0:end_idx]
        train_index = indexes[end_idx:-1]


        X_t, X_v = self.X_train.iloc[train_index], self.X_train.iloc[validate_index]
        y_t, y_v = self.y_train.iloc[train_index], self.y_train.iloc[validate_index]

        self.curr_X_train = X_t
        self.curr_X_validate = X_v
        self.curr_y_train = y_t
        self.curr_y_validate = y_v

    def __get_data_generator(self, train_idx):
        if not self.__training_prepared:
            raise Exception('Run state in not a prepared state.')

        return DataGenerator(
                self.X_train.iloc[train_idx],
                self.y_train.iloc[train_idx],
                features=features,
                feature_scaler=self.scaler,
                batch_size=self.batch_size,
                augment=True,
                shuffle=True,
                image_min_val=self.image_min_val,
                image_range=self.image_range,
            )

    def __get_validate_data_generator(self, validate_idx):
        if not self.__training_prepared:
            raise Exception('Run state in not a prepared state.')

        return DataGenerator(
                self.X_train.iloc[validate_idx],
                self.y_train.iloc[validate_idx],
                features=features,
                feature_scaler=self.scaler,
                batch_size=len(validate_idx),
                augment=False,
                shuffle=False,
                image_min_val=self.image_min_val,
                image_range=self.image_range,
            )

    def get_test_generator(self):
        if not self.__testing_prepared:
            raise Exception('Run state in not a prepared state.')

        return DataGenerator(
                self.X_test,
                self.y_test,
                features=features,
                feature_scaler=self.scaler,
                batch_size=len(self.X_test),
                augment=False,
                shuffle=False,
                image_min_val=self.image_min_val,
                image_range=self.image_range,
            )

    def fit_images(self):
        def __fit_images():
            import data_generator

            if not self.__training_prepared:
                raise Exception('Run state in not a prepared state.')

            min_val = 1000
            max_val = -1000

            for record in self.X_train.itertuples():
                # Don't augment image while fitting images
                try:
                    img = data_generator.get_image(record, IMAGE_SIZE, augment=False)
                    min_val = min(min_val, np.min(img))
                    max_val = max(max_val, np.max(img))
                except:
                    self.X_train.drop(self.X_train.objid == record.objid)

            print(f'Min/max values of images: {min_val}/{max_val}')
            self.image_min_val = min_val
            self.image_range = max_val - min_val

        if (self.image_min_val is None or self.image_max_val is None):
            __fit_images()
        else:
            self.image_range = self.image_max_val - self.image_min_val

    def load_train_data(self, train_file):
        X, y = get_data(train_file)
        self.set_train_data(X, y)

    def load_test_data(self, test_file):
        print(f'Loading test file {test_file}')
        X, y = get_data(test_file)
        self.set_test_data(X, y)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=-1)
