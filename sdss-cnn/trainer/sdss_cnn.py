from __future__ import print_function

from run_state import RunState

import argparse
import numpy as np
import pandas as pd
import csv
import os

from astropy.io import fits

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard

from tensorflow.python.lib.io import file_io
from tensorflow.core.framework.summary_pb2 import Summary
import tensorflow as tf

import sdss_gz_data as sgd
from sdss_gz_data import SPIRIAL_GALAXY_TYPE
from sdss_gz_data import ELLIPTICAL_GALAXY_TYPE
from sdss_gz_data import redshift_err
from sdss_gz_data import z_err

from run_state import N_FOLDS

def init_run_state(args):
    print('Initiating run state...')
    print('')
    
    run_state = RunState()
    print('')
    print('Getting training data...')
    run_state.load_train_data(args.train_file)
    print('')
    print('Getting test data...')
    run_state.load_test_data(args.test_file)

    # TEMP - save run state
    run_state.save(args.run_state_file)

    return run_state

def galaxy_predictions(predictions):
    return np.round(predictions[0]).ravel().astype(int)

def redshift_predictions(predictions):
    return predictions[1].ravel()

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalisation=True,
                 conv_first=True,
                 l2_beta=0
                ):

    def apply_normalisation(x):
        if batch_normalisation:
            x = BatchNormalization()(x)
        
        return x
    
    def apply_activation(x):
        if activation is not None:
            x = Activation(activation)(x)
            
        return x
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(l2_beta),
                 )
    
    x = inputs
    if conv_first:
        x = apply_activation(apply_normalisation(conv(x)))
    else:
        x = conv(apply_activation(apply_normalisation(x)))
    
    return x

def get_config(stage, res_block, num_filters_in):
    num_filters_out = num_filters_in * 2
    activation = 'relu'
    batch_normalisation = True
    strides = 1
    
    if stage == 0:
        num_filters_out = num_filters_in * 4
        if res_block == 0:
            activation = None
            batch_normalisation = False
    else:
        if res_block == 0:
            strides = 2
            
    return num_filters_out, activation, batch_normalisation, strides

def main_block(inputs,
               num_filters_in,
               num_filters_out,
               strides,
               activation,
               batch_normalisation,
               l2_beta):
    y = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     kernel_size=1,
                     strides=strides,
                     activation=activation,
                     batch_normalisation=batch_normalisation,
                     conv_first=False,
                     l2_beta=l2_beta
                    )
    y = resnet_layer(inputs=y,
                     num_filters=num_filters_in,
                     conv_first=False,
                     l2_beta=l2_beta
                    )
    y = resnet_layer(inputs=y,
                     num_filters=num_filters_out,
                     kernel_size=1,
                     conv_first=False,
                     l2_beta=l2_beta
                    )
    
    return y

def residual_block(inputs, num_filters, strides):
    return resnet_layer(inputs=inputs,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalisation=None
                       )

def resnetV2(inputs, args, data_format='channels_last'):
    if (args.depth - 2) % 9 != 0:
        raise ValueError('Invalid depth, must be 9n+2')
    
    num_filters = 16
    num_res_blocks = int((args.depth - 2)/9)
    
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters,
                     conv_first=True,
                     l2_beta=args.l2_reg
                    )
    
    for stage in range(3):
        for res_block in range(num_res_blocks):
            num_filters_out, activation, batch_normalisation, strides = get_config(stage, res_block, num_filters)
            
            y = main_block(inputs=x,
                           num_filters_in=num_filters,
                           num_filters_out=num_filters_out,
                           strides=strides,
                           activation=activation,
                           batch_normalisation=batch_normalisation,
                           l2_beta=args.l2_reg
                          )
            
            if res_block == 0:
                x = residual_block(x, num_filters_out, strides)
                
            x = keras.layers.add([x, y])
        
        num_filters = num_filters_out
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return GlobalAveragePooling2D(data_format=data_format)(x)

def create_nn(run_state, args):
    if run_state.use_cnn:
        input_1 = Input(shape=run_state.image_input_shape(), name='input_1')
        cnn = resnetV2(input_1, args)

    if run_state.use_features:
        input_2 = Input(shape=run_state.features_input_shape(), name='input_2')

    if run_state.use_cnn:
        if run_state.use_features:
            x = keras.layers.concatenate([cnn, input_2])
        else:
            x = cnn
    else:
        if run_state.use_features:
            x = input_2
        else:
            raise Exception('Need at least one input')
    
    # # Make sure we normalise after concatination!!!
    # if (run_state.use_features and run_state.use_cnn):
    #     x = BatchNormalization()(x)

    for idx in range(args.num_hidden_layers - 1):
        x = Dense(args.dense_units,
                kernel_initializer='random_normal',
                kernel_regularizer=l2(args.l2_reg),
                name=f'hidden_layer_{idx + 1}',
                use_bias=True,
                activation='relu'
                )(x)
        x = BatchNormalization()(x)
        
        x = Dropout(args.dropout_rate_1)(x)

    x = Dense(args.dense_units,
              kernel_initializer='random_normal',
              kernel_regularizer=l2(args.l2_reg),
              name=f'hidden_layer_{args.num_hidden_layers}',
              use_bias=True,
              activation='relu'
             )(x)
    x = BatchNormalization()(x)

    x = Dropout(args.dropout_rate_2)(x)

    output_1 = Dense(1, # classification
                     kernel_initializer='random_normal',
                     kernel_regularizer=l2(args.l2_reg),
                     name='output_1',
                     use_bias=True,
                     activation='sigmoid', 
                    )(x)
    output_2 = Dense(1,
                     kernel_initializer='random_normal',
                     kernel_regularizer=l2(args.l2_reg),
                     name='output_2',
                     use_bias=True,
                     activation='linear'
                    )(x)

    optimizer = Adam(lr=args.lr)
    
    outputs = [
        output_1,
        output_2
    ]

    if run_state.use_cnn:
        if run_state.use_features:
            inputs = [
                input_1,
                input_2
            ]
        else:
            inputs = [ input_1 ]
    else:
        inputs = [ input_2 ]

    model = Model(inputs=inputs, outputs=outputs)
    loss_weights = { 
        'output_1': args.galaxy_loss_weight,
        'output_2': args.redshift_loss_weight
    }
    loss = {
        'output_1': 'binary_crossentropy',
        'output_2': redshift_err
    }
    metrics = {
        'output_1': 'accuracy',
        'output_2': redshift_err
    }
    
    model.compile(loss=loss,
                  optimizer=optimizer,
                  loss_weights=loss_weights,
                  metrics=metrics
                 )
    return model

def custom_err(galaxy_err, z_err):
    return (galaxy_err + z_err + np.sqrt((galaxy_err ** 2 + z_err ** 2)/2)) / 3

def train(model, data_generator, validation_generator, callbacks, epochs):
    history = model.fit_generator(
        generator=data_generator,
        validation_data=validation_generator,
        use_multiprocessing=False,
        epochs=epochs,
        callbacks=callbacks,
    )
    
    score = model.evaluate_generator(validation_generator, verbose=0)
    print(score)

    loss = score[0]
    galaxy_err = 1 - score[3]
    z_err = score[4]
    cust_err = custom_err(galaxy_err, z_err)

    print('Test loss: ', loss)
    print('Test Galaxy Morph. error: ', galaxy_err)
    print('Test redshift error: ', z_err)
    print('Test custom error: ', cust_err)
    return loss, galaxy_err, z_err, cust_err, history

def train_model(args):
    run_state = init_run_state(args)

    models = np.empty(args.nfolds, dtype=Model)
    fold_generators = run_state.get_fold_generators(n_splits=args.nfolds)

    earlystopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    callbacks = [ earlystopping_callback ]

    loss_val = 0
    galaxy_err_val = 0
    z_err_val = 0
    cust_err_val = 0
    histories = np.empty(args.nfolds, dtype=object)

    for idx, (data_generator, validation_generator) in enumerate(fold_generators):
        fold_num = idx + 1
        print('')
        model = create_nn(run_state, args)

        if idx == 0:
            model.summary()

        print(f'Training fold {fold_num}')
        loss, galaxy_err, z_err, cust_err, history = train(
            model,
            data_generator,
            validation_generator,
            callbacks=callbacks,
            epochs=args.epochs
        )

        loss_val += loss
        galaxy_err_val += galaxy_err
        z_err_val += z_err
        cust_err_val += cust_err

        print(f'Avg cross validation loss: {loss_val/fold_num}')
        print(f'Avg cross galaxy err: {galaxy_err_val/fold_num}')
        print(f'Avg cross redshift err: {z_err_val/fold_num}')
        print(f'Avg cross custom err: {cust_err_val/fold_num}')
        models[idx] = model
        histories[idx] = history

        model_file = f'{args.model_name}-{fold_num}.h5'
        model.save(model_file)

    with open('histories.pkl', 'wb') as f:
        import pickle
        pickle.dump(histories, f)

    return models

def predict(models, run_state):
    data_generator = run_state.get_test_generator()
    data = data_generator.get_source_data()

    y_test = run_state.y_test
    for idx, model in enumerate(models):
        fold_num = idx + 1
        print('')
        print(f'Processing fold {fold_num}')
        result = model.predict_generator(data_generator)

        z_predictions = redshift_predictions(result)
        z_errs = (z_predictions  - y_test["z"].values)/(1 + y_test["z"].values)

        data[f'pred_galaxy_type_val_{fold_num}'] = result[0].ravel()
        data[f'pred_galaxy_type_{fold_num}'] = galaxy_predictions(result)
        data[f'pred_photoz_{fold_num}'] = z_predictions
        data[f'pred_specz_photoz_err_{fold_num}'] = z_errs

        print(model.evaluate_generator(data_generator,))

    data.to_csv(f'results.csv.bz2', index=False)

def run_predict(args):
    from keras.models import load_model
    from keras.utils.generic_utils import get_custom_objects

    from run_state import load_run_state

    # load model
    get_custom_objects().update({"redshift_err": redshift_err})
    # Load a run state
    run_state = load_run_state(args.run_state_file)
    run_state.load_test_data(args.test_file)

    models = np.empty(args.nfolds, dtype=Model)
    for idx in range(args.nfolds):
        model_name = f'{args.saved_model_prefix}-{idx+1}.h5'
        models[idx] = load_model(model_name)

    predict(models, run_state)

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
                        default="sdss_cnn",
                        help='What to name the saved model file')
    parser.add_argument('--metric-name',
                        type=str,
                        default="sdss_cnn_z_err",
                        help='The name of the metric for validation loss ')
    parser.add_argument('--train-file',
                        type=str,
                        default="./data/astromonical_data.csv.gz",
                        help='The location of the training file')
    parser.add_argument('--test-file',
                        type=str,
                        default="./data/astromonical_data.csv.gz",
                        help='The location of the test file')
    parser.add_argument('--depth',
                        type=int,
                        default=20,
                        help='How deep of a ResnetV2 to use. (default: 20)')
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
                        default=0.000003,
                        help='learning rate (default: 0.00001)')
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
                        default=100,
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
    parser.add_argument('--saved-model-prefix',
                        type=str,
                        default='sdss_cnn',
                        help='Model to use for prediction')
    parser.add_argument('--galaxy-loss-weight',
                        type=float,
                        default=1.0,
                        help='Loss weight for galaxy classification. (default: 1.0)')
    parser.add_argument('--redshift-loss-weight',
                        type=float,
                        default=5.0,
                        help='Loss weight for redshift regression. (default: 5.0)')
    parser.add_argument('--run-state-file',
                        type=str,
                        default='run_state.pkl',
                        help='The file to store/load the run state the model')

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

    if args.predict:
        run_predict(args)
    else:
        train_model(args)

if __name__ == '__main__':
    main()
