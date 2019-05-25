import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io

SPIRIAL_GALAXY_TYPE    = 0
ELLIPTICAL_GALAXY_TYPE = 1
UNKNOWN_GALAXY_TYPE    = 2
CONFIDENCE_LEVEL       = 0.8
IMAGE_SIZE             = 42

def say_hello():
    print('Hello')

def load_data(data_path):
    """Loads data from given file path. This can also be
    a file on GCP. It is expected to be a gzipped file.

    Args:
        data_path (string): The path to the file

    Returns:
        DataFrame: a Pandas data frame with the prepared data which includes extra columns

    """
    def read_file(f):
        data = pd.read_csv(f)
        data.objid = data.objid.astype(str)

        return data

    with file_io.FileIO(data_path, mode='rb') as file:
        if (data_path.endswith('gz')):
            import gzip
            with gzip.open(file, mode='rt') as f:
                return read_file(f)
        elif (data_path.endswith('bz2')):
            # Need to save locallly
            import bz2
            with bz2.BZ2File(file, mode='r') as f:
                return read_file(f)
        else:
            return read_file(file)

def prepare_data(data, init_galaxy_type=True, has_specz=True):
    data = data.copy()
    data = filter_data(data, has_specz=has_specz)

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

    data['model_g_u_colour_index'] =  data.dered_g - data.dered_u
    data['model_r_g_colour_index'] =  data.dered_r - data.dered_g
    data['model_i_r_colour_index'] =  data.dered_i - data.dered_r
    data['model_z_i_colour_index'] =  data.dered_z - data.dered_i

    # does average of stokes in different bands really matter?
    data['avg_stokes_u'] = (data.stokes_u_u + data.stokes_u_g + data.stokes_u_r + data.stokes_u_i + data.stokes_u_z)/5
    data['avg_stokes_q'] = (data.stokes_q_u + data.stokes_q_g + data.stokes_q_r + data.stokes_q_i + data.stokes_q_z)/5

    # Average of petro rad
    data['avg_petro_rad'] = (data.petroRad_u + data.petroRad_g + data.petroRad_r + data.petroRad_i + data.petroRad_z)/5

    # Petro Mag colour index
    data['petro_g_u_colour_index'] = data.petroMag_g - data.petroMag_u
    data['petro_r_g_colour_index'] = data.petroMag_r - data.petroMag_g
    data['petro_i_r_colour_index'] = data.petroMag_i - data.petroMag_r
    data['petro_z_i_colour_index'] = data.petroMag_z - data.petroMag_i

    # psfMag colour index
    data['psfMag_g_u_colour_index'] = data.psfMag_g - data.psfMag_u
    data['psfMag_r_g_colour_index'] = data.psfMag_r - data.psfMag_g
    data['psfMag_i_r_colour_index'] = data.psfMag_i - data.psfMag_r
    data['psfMag_z_i_colour_index'] = data.psfMag_z - data.psfMag_i

    if init_galaxy_type:
        combined_spiral = data.spiralclock + data.spiralanticlock + data.edgeon
        data['combined_spiral'] = combined_spiral

        data['galaxy_type'] = UNKNOWN_GALAXY_TYPE
        data.loc[data.debiased_elliptical > CONFIDENCE_LEVEL, 'galaxy_type'] = ELLIPTICAL_GALAXY_TYPE
        data.loc[data.debiased_spiral > CONFIDENCE_LEVEL, 'galaxy_type'] = SPIRIAL_GALAXY_TYPE

        num_of_elliptical = data[data.galaxy_type == ELLIPTICAL_GALAXY_TYPE].size
        num_of_spirial = data[data.galaxy_type == SPIRIAL_GALAXY_TYPE].size
        num_of_unknown = data[data.galaxy_type == UNKNOWN_GALAXY_TYPE].size
        total_count = data.size

        print(f'% elliptical:      {num_of_elliptical / total_count}')
        print(f'% spiral:          {num_of_spirial / total_count}')
        print(f'% unknown:         {num_of_unknown / total_count}')
        print(f'% spiral of known: {num_of_spirial / (num_of_elliptical + num_of_spirial)}')

    return data

def filter_data(data, has_specz=True):
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

    radii_fields = ['petroRad', 'expRad', 'deVRad']
    for field in radii_fields:
        for band in bands:
            field_name = f'{field}_{band}'
            filters.append(data[field_name] > 0.0)

    filters.append(data['photoz'] != -9999)
    filters.append(data['photozErr'] != -9999)
    if has_specz:
        filters.append(data['zErr'] != -1)

    result = data[np.all(filters, axis=0)].reset_index(drop=True)
    if has_specz:
        num_of_high_z = len(result[result.z > 0.4])
        print(f'Number of high z galaxies = {num_of_high_z}')
        result = result[result.z <= 0.4].reset_index(drop=True)

    print(f'Filtered out {len(data) - len(result)} invalid records')

    return result

def generate_features(bands=['u','g','i','r','z'], use_stokes=True, use_averages=False, use_model_colour_index=True):
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
        # 'fracDeV',
        'mag'
    ]
    
    stokes_features = [
        'stokes_q',
        'stokes_u',
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
        'g_u_colour_index',
        'r_g_colour_index',
        'i_r_colour_index',
        'z_i_colour_index',
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
            feature = f'{band}_{band2}_colour_index'
            if feature in valid_colour_indexes:
                petro_feature = f'petro_{feature}'
                features.append(petro_feature)
                if use_model_colour_index:
                    features.append(f'model_{feature}')
                features.append(f'psfMag_{feature}')

    if use_averages:
        features.extend(average_features)
        if use_stokes:
            features.extend(average_stokes_features)

    return features

def select_features (data, bands=['u','g','i','r','z'], band_features=[], colour_indexes=[]):
    features = []
    valid_colour_indexes = [
        'g_u_colour_index',
        'r_g_colour_index',
        'i_r_colour_index',
        'z_i_colour_index',
    ]

    for band in bands:
        for band_feature in band_features:
            features.append(f'{band_feature}_{band}')

    for colour_index_prefix in colour_indexes:
        for colour_index in valid_colour_indexes:
            features.append(f'{colour_index_prefix}_{colour_index}')

    return data[features], features

def transform_data(data, bands=['u','g','i','r','z']):
    def power(scale):
        def _power(x):
            return x ** scale

        return _power

    data = data.copy()

    config = [
        {
            'func': np.log,
            'feature_config': {
                'band_features': [
                    'petroRad',
                    'petroR50',
                    'petroR90',
                    'petro_R90_R50_ratio',
                ],
                'features': [
                    'avg_petro_rad',
                    'avg_petro_R50',
                    'avg_petro_R90',
                    'avg_petro_R90_R50_ratio',
                ]
            }
        },
        {
            'func': np.log1p,
            'feature_config': {
                'band_features': [
                    'deVRad',
                    'expRad',
                ],
                'features': []
            }
        },
        {
            'func': np.tanh,
            'feature_config': {
                'band_features': [
                    'stokes_q',
                    'stokes_u'
                ],
                'features': [
                    'avg_stokes_q',
                    'avg_stokes_u'
                ]
            }
        },
        {
            'func': power(1/6),
            'feature_config': {
                'band_features': [
                    'extinction',
                ],
                'features': [
                ]
            }
        },
        # {
        #     'func': power(1/3),
        #     'feature_config': {
        #         'band_features': [
        #         ],
        #         'features': [
        #             'photoz',
        #             'z'
        #         ]
        #     }
        # },
    ]

    for entry in config:
        func = entry['func']
        for band in bands:
            for base_feature in entry['feature_config']['band_features']:
                feature = f'{base_feature}_{band}'
                data[feature] = data[feature].apply(func)
        
        for feature in entry['feature_config']['features']:
            data[feature] = data[feature].apply(func)
    
    return data

def classification_scores(y_test, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    recall = tp/(tp + fn)
    specificity = tn/(tn + fp)
    precision = tp/(tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1 = 2 * precision * recall / (precision + recall)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    results = {
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'mcc': mcc
    }

    return results

def binnify(y, num_bins=24, min_y=-3, max_y=3):
    # Need to stratify y, but should scale y to make it a better distribution
    y_norm = y['z'] ** 1/3
    
    bins = np.linspace(min(y_norm - 1e-5), max(y_norm + 1e-5) , num_bins)
    y_binned = np.digitize(y_norm, bins)
    y_binned = y_binned + y['galaxy_type'] * num_bins # this account for both redshift and galaxy classification
    
    return y_binned

def split_train(X, y, random_state=None, num_bins=24, min_y=-3, max_y=3, test_size=0.2):
    from sklearn.model_selection import train_test_split
    y_binned = binnify(y, num_bins=num_bins, min_y=min_y, max_y=max_y)
    
    print(np.unique(y_binned))
    min_z = np.min(y['z'])
    max_z = np.max(y['z'])

    print(f'Min z:             {min_z}')
    print(f'Max z:             {max_z}')
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_binned)

def redshift_err(y_true, y_pred):
    from keras import backend as K

    diff = K.abs(y_pred - y_true)
    scale = 1. + y_true

    return K.mean(diff / scale, axis=-1)

def z_err(preds, train_data):
    if hasattr(train_data,'label'):
        train_data = train_data.label
    train_data = train_data.ravel()
    preds = preds.ravel()

    error = np.abs((train_data - preds)/(1 + train_data)).mean()
    return 'z_err', error, False

def z_err_stats(actual, predicted):
    err = np.abs((predicted - actual)/(1 + actual))
    num_cat_failure = len(err[err > 0.15])
    num_records = len(err)
    
    catastrophic_failure = num_cat_failure / num_records
    
    return err.mean(), np.std(err), np.min(err), np.max(err), catastrophic_failure

def z_pop_std(y, yhat):
    sum_errs = np.sum((y - yhat)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)
    return stdev

def scale_rgb(data, sigma=1/3, gains=[0.9,1.1,1.8], gamma=0.1):
    from numpy import arcsinh

    min = 0
    max = np.max(data)

    R_IDX = 0
    G_IDX = 1
    B_IDX = 2
    
    if min < 0:
        data = data - min
        max = max - min
        min = 0

    r = data[R_IDX].copy()
    g = data[G_IDX].copy()
    b = data[B_IDX].copy()

    slope = 255 / arcsinh((max - min)/sigma)

    mean = (r + g + b)/3
    mean[mean < min] = 0
    r[mean == 0] = 0
    g[mean == 0] = 0
    b[mean == 0] = 0
    
    scale = slope * arcsinh((mean - min) / sigma) / mean

    r = (r * scale).astype(int)
    g = (g * scale).astype(int)
    b = (b * scale).astype(int)
    
    r = (r * gains[R_IDX]).astype(int)
    g = (g * gains[G_IDX]).astype(int)
    b = (b * gains[B_IDX]).astype(int)
    
    r += (gamma * (r - g)).astype(int)
    b += (gamma * (b - g)).astype(int)

    r[r < 0] = 0
    r[r > 255] = 255
    g[g < 0] = 0
    g[g > 255] = 255
    b[b < 0] = 0
    b[b > 255] = 255
    
    result = np.empty(data.shape, dtype=np.uint8)
    result[0] = r
    result[1] = g
    result[2] = b

    return result

def get_gcs_bucket():
    from google.cloud import storage

    gcs_client = storage.Client()
    return gcs_client.get_bucket('wgauvin-astroml-ast80014')
