from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import csv
import os

from astropy.io import fits

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

import keras
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import Input, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard

from tensorflow.python.lib.io import file_io
from tensorflow.core.framework.summary_pb2 import Summary
import tensorflow as tf

N_FOLDS=5
SPIRIAL_GALAXY_TYPE    = 0
ELLIPTICAL_GALAXY_TYPE = 1
UNKNOWN_GALAXY_TYPE    = 2

target_column = 'z'

CONFIDENCE_LEVEL = 0.8

def scaled_tanh(scale):
    def _scaled_tanh(x):
        return scale * K.tanh(x)

    return _scaled_tanh

def mae(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    return np.abs(y - y_pred).mean()

def mse(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    return np.square(y - y_pred).mean()

def rmse(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    return np.sqrt(mse(y, y_pred))  

def r_square(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()

    SS_res =  np.square(y - y_pred).sum()
    SS_tot = np.square(y - np.mean(y)).sum() 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def z_mae(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()

    diff = y - y_pred
    scale = 1 + y
    return np.abs(diff / scale).mean()

def z_mse(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()

    diff = y - y_pred
    scale = 1 + y
    return np.square(diff / scale).mean()

def redshift_err(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    scale = 1. + y_true

    return K.mean(diff / scale, axis=-1)

def print_accuracy(y, y_pred):
    print(f'mae:   {mae(y, y_pred)}')
    print(f'mse:   {mse(y, y_pred)}')
    print(f'rmse:  {rmse(y, y_pred)}')
    print(f'r2:    {r_square(y, y_pred)}')
    print(f'z_mae: {z_mae(y, y_pred)}')
    print(f'z_mse: {z_mse(y, y_pred)}')

def load_data(data_path):
    import gzip
    with file_io.FileIO(data_path, mode='rb') as file:
       with gzip.open(file, mode='rt') as f:
            return prepare_data(pd.read_csv(f))

def filter_data(data):
    # need to remove rows that have invalid 
    bands = ['u', 'g', 'r', 'i', 'z']
    base_fields = ['petroRad', 'petroR50', 'petroR90', 'expRad', 'deVRad',
            'petroMag', 'psfMag', 'expMag', 'dered', 'fiberMag', 'extinction',
            'expAB', 'expPhi', 'deVAB', 'deVPhi', 'fracDeV', 'stokes_q',
            'stokes_u', 'mag'
        ]

    filters = []
    for field in base_fields:
        for band in bands:
            field_name = f'{field}_{band}'
            filters.append(data[field_name] != -9999)

    result = data[np.all(filters, axis=0)].reset_index(drop=True)
    print(f'Filtered out {len(data) - len(result)} invalid records')
    return result

def prepare_data(data):
    data = data.copy()
    data = filter_data(data)
    num_of_high_z = len(data[data.z > 0.4])
    print(f'Number of high z galaxies = {num_of_high_z}')
    data = data[data.z <= 0.4].reset_index(drop=True)

    combined_spiral = data.spiralclock + data.spiralanticlock + data.edgeon
    data['galaxy_type'] = UNKNOWN_GALAXY_TYPE
    data['combined_spiral'] = combined_spiral
    data.loc[data.debiased_elliptical > CONFIDENCE_LEVEL, 'galaxy_type'] = ELLIPTICAL_GALAXY_TYPE
    data.loc[data.debiased_spiral > CONFIDENCE_LEVEL, 'galaxy_type'] = SPIRIAL_GALAXY_TYPE

    # Add petroR50/petroR90
    data['petro_R90_R50_ratio_u'] = data.petroR90_u / data.petroR50_u
    data['petro_R90_R50_ratio_g'] = data.petroR90_g / data.petroR50_g
    data['petro_R90_R50_ratio_r'] = data.petroR90_r / data.petroR50_r
    data['petro_R90_R50_ratio_i'] = data.petroR90_i / data.petroR50_i
    data['petro_R90_R50_ratio_z'] = data.petroR90_z / data.petroR50_z
    data['avg_petro_rad'] = (data.petroRad_u + data.petroRad_g + data.petroRad_r + data.petroRad_i + data.petroRad_z)/5
    data['avg_petro_R50'] = (data.petroR50_u + data.petroR50_g + data.petroR50_r + data.petroR50_i + data.petroR50_z)/5
    data['avg_petro_R90'] = (data.petroR90_u + data.petroR90_g + data.petroR90_r + data.petroR90_i + data.petroR90_z)/5
    data['avg_petro_R90_R50_ratio'] = data.avg_petro_R90 / data.avg_petro_R50

    data['u_g_colour_index'] = data.dered_u - data.dered_g
    data['g_r_colour_index'] = data.dered_g - data.dered_r
    data['r_i_colour_index'] = data.dered_r - data.dered_i
    data['i_z_colour_index'] = data.dered_i - data.dered_z

    # does average of stokes in different bands really matter?
    data['avg_stokes_u'] = (data.stokes_u_u + data.stokes_u_g + data.stokes_u_r + data.stokes_u_i + data.stokes_u_z)/5
    data['avg_stokes_q'] = (data.stokes_q_u + data.stokes_q_g + data.stokes_q_r + data.stokes_q_i + data.stokes_q_z)/5

    # Average of petro rad
    data['avg_petro_rad'] = (data.petroRad_u + data.petroRad_g + data.petroRad_r + data.petroRad_i + data.petroRad_z)/5

    # Petro Mag colour index
    data['petro_u_g_colour_index'] = data.petroMag_u - data.petroMag_g
    data['petro_g_r_colour_index'] = data.petroMag_g - data.petroMag_r
    data['petro_r_i_colour_index'] = data.petroMag_r - data.petroMag_i
    data['petro_i_z_colour_index'] = data.petroMag_i - data.petroMag_z

    # Stokes P
    data['stokes_p_u'] = np.sqrt(np.power(data.stokes_q_u, 2) + np.power(data.stokes_u_u, 2))
    data['stokes_p_g'] = np.sqrt(np.power(data.stokes_q_g, 2) + np.power(data.stokes_u_g, 2))
    data['stokes_p_i'] = np.sqrt(np.power(data.stokes_q_i, 2) + np.power(data.stokes_u_i, 2))
    data['stokes_p_r'] = np.sqrt(np.power(data.stokes_q_r, 2) + np.power(data.stokes_u_r, 2))
    data['stokes_p_z'] = np.sqrt(np.power(data.stokes_q_z, 2) + np.power(data.stokes_u_z, 2))

    num_of_elliptical = data[data.galaxy_type == ELLIPTICAL_GALAXY_TYPE].size
    num_of_spirial = data[data.galaxy_type == SPIRIAL_GALAXY_TYPE].size
    num_of_unknown = data[data.galaxy_type == UNKNOWN_GALAXY_TYPE].size
    total_count = data.size

    print(f'% elliptical:      {num_of_elliptical / total_count}')
    print(f'% spiral:          {num_of_spirial / total_count}')
    print(f'% unknown:         {num_of_unknown / total_count}')
    print(f'% spiral of known: {num_of_spirial / (num_of_elliptical + num_of_spirial)}')

    return data[data.galaxy_type != UNKNOWN_GALAXY_TYPE]

def split_train(data, features, random_state=None, num_bins=24, min_y=-3, max_y=3, test_size=0.2):
    X = data[features]
    y = data['z'].values

    # Need to stratify y
    y_scaler = PowerTransformer()
    y_norm = y_scaler.fit_transform(y.reshape(-1, 1))

    bins = np.linspace(-3, 3, num_bins)
    y_binned = np.digitize(y_norm, bins)

    print(f'Min z:             {np.min(y)}')
    print(f'Max z:             {np.max(y)}')
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_binned)

def scale_data(X):
    X_scaler = PowerTransformer()
    X_norm = X_scaler.fit_transform(X)

    return X_scaler, X_norm

def create_nn(input_shape, output_shape, num_hidden_layers=2, dense_units=1024, dropout_rate_1=0.1, dropout_rate_2=0.3, l1_reg=0.1, l2_reg=0.01, lr=0.00001, loss=None):
    input = Input(shape=input_shape, name='input_1')
    x = input

    for idx in range(1, 1 + num_hidden_layers):
        if idx > 1:
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate_1)(x)

        x = Dense(dense_units,
                kernel_initializer='random_normal',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                )(x)
        x = Activation('relu', name=f'activation_{idx}')(x)

    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_2)(x)
    output = Dense(output_shape,
              kernel_initializer='random_normal',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
             )(x)

    optimizer = Adam(lr=lr)
    
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=[redshift_err, 'mse', 'mae'])
    return model

def generate_features(bands=['u','g','i','r','z'], use_stokes=True, use_averages=False, use_normal_colour_index=True):
    features = []

    base_features = [
        'dered',
        'petroRad',
        'petroR50',
        'petroR90',
        'petro_R90_R50_ratio',
        'petroMag',
        'expRad',
        'deVRad',
        'psfMag',
        'expMag',
        'fiberMag',
        'extinction',
        'expAB',
        'expPhi',
        'deVAB',
        'deVPhi',
        'fracDeV',
        'mag'
    ]
    
    stokes_features = [
        'stokes_q',
        'stokes_u',
        'stokes_p'
    ]

    average_features = [
        'avg_petro_rad',
        'avg_petro_R50',
        'avg_petro_R90',
        'avg_petro_R90_R50_ratio'
    ]
    
    average_stokes_features = [
        'avg_stokes_q',
        'avg_stokes_u',
    ]
    
    valid_colour_indexes = [
        'u_g_colour_index',
        'g_r_colour_index',
        'r_i_colour_index',
        'i_z_colour_index',
    ]
    
    for band in bands:
        for base_feature in base_features:
            feature = '{}_{}'.format(base_feature, band)
            features.append(feature)
            
        if use_stokes:
            for stokes_feature in stokes_features:
                feature = '{}_{}'.format(stokes_feature, band)
                features.append(feature)
        
        for band2 in bands:
            feature = '{}_{}_colour_index'.format(band, band2)
            if feature in valid_colour_indexes:
                petro_feature = 'petro_{}'.format(feature)
                features.append(petro_feature)
                if use_normal_colour_index:
                    features.append(feature)

    if use_averages:
        features.extend(average_features)
        if use_stokes:
            features.extend(average_stokes_features)

    return features

def train(model, X_train, y_train, X_test, y_test, batch_size, epochs, callbacks=[], verbose=False):
    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(X_test, y_test),
                callbacks=callbacks)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test error:', score[1])

    return score

def get_folds(X, y, n_splits=N_FOLDS, seed=None, num_bins=24):
    if n_splits == 1:
        # fake it by getting the first 80% as X
        idx = int(len(X) * .8)
        train_idx = np.arange(idx)
        validate_idx = np.arange(idx, len(X))
        return [(train_idx, validate_idx)]
    else:
        bins = np.linspace(-3, 3, num_bins)
        y_binned = np.digitize(y, bins)
        
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y_binned,)
        return folds

def train_model(args):
    job_dir = args.job_dir
    metric_name = args.metric_name
    train_file = args.train_file
    model_name = args.model_name
    features = generate_features()

    log_path = f'{job_dir}/logs'
    if (args.verbose or args.hypertuning):
        print(args)
        print(f'Using logs_path located at {log_path}')

    data = load_data(train_file)

    X, X_test, y, y_test = split_train(data,
                            features,
                            random_state=args.seed,
                            num_bins=args.bins,
                            min_y=-3)

    X_scaler, X = scale_data(X)

    # create TensorBoard callback
    tensorboard_callback = TensorBoard(log_path)
    earlystopping_callback = EarlyStopping(patience=3, restore_best_weights=True)
    callbacks = [tensorboard_callback, earlystopping_callback]

    loss_val = 0
    error_val = 0
    models = np.empty(args.nfolds, dtype=Model)
    for idx, (train_idx, valid_idx) in enumerate(get_folds(X, y, n_splits=args.nfolds, seed=args.seed, num_bins=args.bins)):
        print(f'Fold {idx + 1}')
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        model = create_nn(
                (len(features),),
                1,
                num_hidden_layers=args.num_hidden_layers,
                dense_units=args.dense_units,
                dropout_rate_1=args.dropout_rate_1,
                dropout_rate_2=args.dropout_rate_2,
                l1_reg=args.l1_reg,
                l2_reg=args.l2_reg,
                lr=args.lr,
                loss=redshift_err
            )

        score = train(model,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    callbacks=callbacks,
                    verbose=args.verbose
                    )
        
        loss_val += score[0]
        error_val += score[1]
        print(f'Avg cross validation loss: {loss_val/(idx + 1)}')
        models[idx] = model

    loss_val /= args.nfolds
    error_val /= args.nfolds

    print(f'Final validation loss : {loss_val}')
    print(f'Final validation error: {error_val}')

    if args.predict:
        X_test_norm = X_scaler.transform(X_test)

        y_pred = np.zeros_like(y_test)
        for model in models:
            curr_pred = model.predict(X_test_norm)
            y_pred += curr_pred.ravel()/args.nfolds

        print_accuracy(y_test, y_pred)
        err = (y_pred - y_test)/(1 + y_test)
        print(f'Actual   : {y_test[0:20]}')
        print(f'Predicted: {y_pred[0:20]}')
        print(f'Err:       {err[0:20]}')

    if args.hypertuning:
        import hypertune

        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=metric_name,
            metric_value=error_val,
            global_step=args.epochs
        )

    model_file = f'{model_name}.h5'
    model.save(model_file)

    with file_io.FileIO(model_file, mode='rb') as input_f:
        with file_io.FileIO(f'{job_dir}/{model_file}', mode='wb+') as output_f:
            output_f.write(input_f.read())

def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Redshift NN Training Example')

    parser.add_argument('--job-dir',  # handled automatically by ML Engine
                        type=str,
                        default='./job',
                        help='GCS location to write checkpoints and export ' \
                             'models')
    parser.add_argument('--model-name',
                        type=str,
                        default="redshift_nn",
                        help='What to name the saved model file')
    parser.add_argument('--metric-name',
                        type=str,
                        default="redshift_nn_z_err",
                        help='The name of the metric for validation loss ')
    parser.add_argument('--train-file',
                        type=str,
                        default="./data/astromonical_data.csv.gz",
                        help='The location of the training file')
    parser.add_argument('--dense-units',
                        type=int,
                        default=512,
                        help='Number of dense units per layer (default: 512)')
    parser.add_argument('--dropout-rate-1',
                        type=float,
                        default=0.1,
                        help='Dropout rate for first hidden layer (default: 0.2)')
    parser.add_argument('--dropout-rate-2',
                        type=float,
                        default=0.3,
                        help='Dropout rate for second hidden layer (default: 0.2)')
    parser.add_argument('--l1-reg',
                        type=float,
                        default=0.0001,
                        help='L1 regularisation value (default: 0.01)')
    parser.add_argument('--l2-reg',
                        type=float,
                        default=0.01,
                        help='L2 regularisation value (default: 0.0001)')
    parser.add_argument('--lr',  # Specified in the config file
                        type=float,
                        default=0.00001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--num-hidden-layers',  # Specified in the config file
                        type=int,
                        default=1,
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--seed',
                        type=int,
                        help='random seed use for reproducability')
    parser.add_argument('--verbose',
                        type=bool,
                        default=False,
                        help='Whether to be versbose or not')
    parser.add_argument('--hypertuning',
                        type=bool,
                        default=False,
                        help='Whether hypertuning or not')
    parser.add_argument('--bins',
                        type=int,
                        default=24,
                        help='Number of bins to help stratify redshift (default: 24)')
    parser.add_argument('--nfolds',
                        type=int,
                        default=N_FOLDS,
                        help='Number of folds to do during KFolds. (default: 5)')
    parser.add_argument('--predict',
                        type=bool,
                        default=False,
                        help='Predict results')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.hypertuning:
        print('Running hyperparameter tuning job')
        if args.seed is None:
            print('Randing seed not set but due to hypertuning setting it to 42')
            args.seed = 42

    if args.seed:
        print('Seed is set. Making sure numpy, keras and tensorflow use the same seed')
        from tensorflow import set_random_seed
        np.random.seed(args.seed)
        set_random_seed(args.seed)

    train_model(args)

if __name__ == '__main__':
    main()
