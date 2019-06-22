'''
Detect anomalies on a HPC supercomputer node
Andrea Borghesi
    University of Bologna
'''
import os
import sys
import time
import datetime
import pickle
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K 
from keras import regularizers
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.mixture import GMM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import util

_epochs = 2
_batch_size = 64

_base_dir = './'
_data_dir = _base_dir + 'data/'

'''
Initial and final date & time of the time frame considered
'''
start_time = datetime.datetime.strptime(
    "2018-03-03 00:00:01","%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime(
    "2018-05-30 05:00:01","%Y-%m-%d %H:%M:%S")

'''
Specify the types of frequency governors adopted; conservative is default.
There are two possible types of anomalies: powersave and performance
To each node in the data set is associated a anomaly type
'''
_freqGov = {'davide16' : 'powersave', 'davide17' : 'powersave', 
        'davide18' : 'powersave', 'davide19' : 'powersave', 
        'davide26' : 'performance', 'davide27' : 'performance',
        'davide28' : 'performance', 'davide29' : 'performance',
        'davide30' : 'performance', 'davide31' : 'performance',
        'davide32' : 'performance', 'davide33' : 'performance',
        'davide34' : 'performance', 'davide42' : 'powersave',
        'davide45' : 'powersave'}

'''
Data collected contains also idle periods
These periods are of little interest for the anomaly types considered in this
work, hence the idle periods can be directly removed from the data set
(Leaving the idle periods might cause the anomaly detection approach to identify
as anomalous all non-idle periods, since the supercomputer was in idle state for
most of the considered time frame)
'''
_remove_idle = True

'''
The data is collected by EXAMON monitoring system with sampling rate that can
vary from 50ms (fine-grained power measurements) to 5-10 seconds (performance
counters, IPMI, etc). But this data was to large to be stored long-term, hence
it was aggregated by computing the average values over 5 minutes (300s)
periods. This aggregated data is used in this script for anomaly detection.
'''
_phys_sampling_time = 300

'''
Create an autoencoder to evaluate correlations between measurements, extract
features 
- data set not seen as a time series, but rather each time stamp with its
    measurements values is data point
'''
def correlation_autoencoder(node, df, timestamps, series_labels, gaps, idle_periods, 
        active_idle=[], ts_idle=[], ts_noIdle=[]):
    st_train_set = str(timestamps[0]).replace('-','').replace(
                ':','').replace(' ','_')
    et_train_set = str(timestamps[-1]).replace('-','').replace(
                ':','').replace(' ','_')

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape
    input_data = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, df_anomalies,
            test_noAnomalies, test_anomalies) = util.split_dataset(df_tensor,
                    ts_noIdle, _freqGov[node])

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    encoded = Dense(n_features * 10, activation='relu',
                    activity_regularizer=regularizers.l1(1e-5))(input_data)
    decoded = Dense(n_features, activation='linear')(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    history = autoencoder.fit(x_train, x_train, epochs=_epochs, 
            batch_size=_batch_size, shuffle=True, 
            validation_data=(x_test, x_test),
            sample_weight=np.asarray(active_idle),
            verbose=1)

    # Errors distributions
    decoded_normal = autoencoder.predict(df_noAnomalies)
    decoded_anomal = autoencoder.predict(df_anomalies)
    decoded_normal_test = autoencoder.predict(test_noAnomalies)

    util.error_distribution_2_class_varyThreshold(test_noAnomalies,
            decoded_normal_test, df_anomalies, decoded_anomal, 
            node, df_noAnomalies, decoded_normal)

    # plot errors
    decoded_features = autoencoder.predict(df_tensor)
    stats_res = util.evaluate_predictions(
            decoded_features, df_tensor, gaps)
    util.plot_errors_DL(stats_res, series_labels, timestamps,
            _phys_sampling_time, gaps, idle_periods, [], ts_noIdle, ts_idle,
            test_idx, _freqGov[node])
    return stats_res

'''
Semi-supervised approach based on autoencoder NN
'''
def semi_supervised_ae_based(node, df_obj):
    (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, scaler
            ) = (df_obj['df'], df_obj['timestamps'], df_obj['idle_periods'],
                    df_obj['active_idle'], df_obj['ts_noIdle'], 
                    df_obj['ts_idle'], df_obj['scaler'])

    gaps = util.find_gaps_timeseries(timestamps, _phys_sampling_time)
    res = correlation_autoencoder(node, df, timestamps, df.columns, 
            gaps, idle_periods, [], ts_idle, ts_noIdle)

def main(argv):
    print("================================================================")

    node = argv[0]
    exp_mode = int(argv[1])

    df_already_prepared, df_file = util.check_df(_data_dir, start_time, 
            end_time, node, _remove_idle)

    if not df_already_prepared:
        print("Retrieve data")
        data_ipmi, data_bbb, data_occ = util.retrieve_data(
                start_time, end_time, node, _data_dir)
        phys_data = util.create_df(data_ipmi, data_bbb, data_occ)

        df = pd.DataFrame(phys_data[node])
        df = df.transpose()
        (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, 
                scaler) = util.prepare_dataframe(df, _remove_idle)

        df_obj = {}
        df_obj['df'] = df
        df_obj['timestamps'] = timestamps
        df_obj['idle_periods'] = idle_periods
        df_obj['active_idle'] = active_idle
        df_obj['ts_noIdle'] = ts_noIdle
        df_obj['ts_idle'] = ts_idle
        df_obj['scaler'] = scaler

        with open(df_file, 'wb') as handle:
            pickle.dump(df_obj, handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(df_file, 'rb') as handle:
            df_obj = pickle.load(handle, encoding='latin1')

    if exp_mode == 0:
        # Semi-supervised learning with autoencoder based model
        semi_supervised_ae_based(node, df_obj)

if __name__ == '__main__':
    main(sys.argv[1:])
