'''
Analyse davide off-line physical collected data (basically 
time-series) in order to extract interesting features
- work with single nodes (one AE per node)

Author: Andrea Borghesi, andrea.borghesi3@unibo.it,
    University of Bologna
Date: 20171030
'''
import os
import sys
import my_util
import math
import random
import time
import datetime
import pickle
import configparser
from collections import deque
import collections
from decimal import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Sequential, Model
from keras import backend as K 
from keras import optimizers, initializers, regularizers
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input
from keras.layers import UpSampling1D, Lambda, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from cassandra.auth import PlainTextAuthProvider
from sklearn import tree, preprocessing, metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, median_absolute_error
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.externals.six import StringIO  
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GMM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pydot 

latentVar_dir = '/home/b0rgh/data_collection_analysis/data_analysis/davide/'
latentVar_dir += 'anomaly_detection/latentVar_plot_perEpoch/'

_ncores = 16

# number of previous timeseries values to use as input features
#_window_size = 16     # CNN regressor
#_window_size = 1      # predict as last value
_window_size = 4      # AE
# the size of the sliding window that gets convolved with each position 
# along each instance. 
##_filter_length = 2
_filter_length = 2
# The number of different filters to learn
_nb_filter = 4
#_nb_filter = 2

_encoding_dim = 10
_epochs = 100
_batch_size = 64
_n_behave_cluster = 3

# dimension of latent variable (VAE/AAE)
_n_z = 16
#_m = 32
_vae_loss_kl_weight = 1
_vae_loss_recon_weight = 20
_vae_lr=0.001 
_vae_decay = 0.005

# Rolling Mean Window Size
N = 1

# frequency to plot the latent representation
_plot_interval = 1

#my_home_dir = '/home/b0rgh/'
my_home_dir = '/media/b0rgh/Elements/'
raw_data_dir = my_home_dir + 'rawData_logs_eurora/'
raw_data_dir_davide = raw_data_dir + "davide/"
offline_jobs_dir = raw_data_dir_davide + 'aggregate_job_info/'
offline_phys_dir = raw_data_dir_davide + 'physical_measures/'
plot_dir = raw_data_dir_davide + 'plots/predictions/'

offline_phys_file = offline_phys_dir + 'phys_measures_'

script_dir = '/home/b0rgh/data_collection_analysis/data_analysis/davide/'
script_dir += 'anomaly_detection/'
trained_model_dir = script_dir + 'trained_AEs/'

# OCC measurements appear to have a 'wrong' timestamp, one second later
# than the real value -- we need to have an end time that takes this into 
# account and it's one second later than desired

start_time = datetime.datetime.strptime(
    "2018-03-03 00:00:01","%Y-%m-%d %H:%M:%S")
#start_time = datetime.datetime.strptime(
#    "2018-04-10 00:00:01","%Y-%m-%d %H:%M:%S")

#end_time = datetime.datetime.strptime(
#    "2018-04-23 05:00:01","%Y-%m-%d %H:%M:%S")
#end_time = datetime.datetime.strptime(
#    "2018-05-26 05:00:01","%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime(
    "2018-05-30 05:00:01","%Y-%m-%d %H:%M:%S")

# must be a subset of [start_time, end_time]
test_start_time = datetime.datetime.strptime(
    "2017-10-01 00:00:01","%Y-%m-%d %H:%M:%S")
test_end_time = datetime.datetime.strptime(
    "2017-12-15 22:00:01","%Y-%m-%d %H:%M:%S")
test_start_time = start_time
test_end_time = end_time

test_nodes = ['davide10']
#test_nodes = ['davide42','davide45','davide16','davide17','davide18','davide19',
#    'davide26','davide27','davide28','davide29','davide10','davide11',
#    'davide12','davide13']
#test_nodes = ['davide19','davide45']

'''
Specify the types of frequency governors adopted; conservative is default.
Some nodes only have one types of anomaly (i.e. davide16 only power save)
Others have different anomalies: davide17 has powersave and then performance
    davide27 has performance and then powersave
'''
#_freqGov = 'powersave'
#_freqGov = 'performance'
#_freqGov = 'ondemand'
_freqGov = 'power_perf'
#_freqGov = 'perf_power'

_remove_idle = True

_split_random = 1

_test_train_split_time = datetime.datetime.strptime(
    "2018-04-12 00:00:00","%Y-%m-%d %H:%M:%S")

_phys_sampling_time = 300
JOBS_NSAMPLES = 100

_start_timer = 0

_data_set_size = 'FULL'
#_data_set_size = 1000

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def sample_z(args):
    mu, log_sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    #eps = K.random_normal(shape=(batch, dim), mean=0., stddev=.5)
    return mu + K.exp(0.5 * log_sigma) * eps

'''
Create an autoencoder to evaluate correlations between measurements, extract
    features 
    - data set not seen as a time series, but rather each time stamp with its
        measurements values is data point
'''
def correlation_autoencoder(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[], set_size='FULL'):
    st_train_set = str(timestamps[0]).replace('-','').replace(
                ':','').replace(' ','_')
    et_train_set = str(timestamps[-1]).replace('-','').replace(
                ':','').replace(' ','_')
    ae_name = ('AE_semisuper_rndmSplit_{}trainNode_10x_{}ep_{}bs_{}rm'
            '_ST{}_ET{}_{}s.h5'.format(node, _epochs, _batch_size, N, 
                st_train_set, et_train_set, set_size))
    model_name = trained_model_dir + ae_name

    # rolling mean
    #df = df.rolling(N, min_periods=1).mean()

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape
    #print(n_samples, n_features)

    # even when the active_idle list is present it might have a different
    # number of values compared to n_samples (this really should't happen..)
    #if len(active_idle) != n_samples

    input_data = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    if _split_random == 1:  # random, but exclude anomalies
        test_idx = 0
        #(x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
        #        df_anomalies) = my_util.split_dataset(
        #        df_tensor, ts_noIdle, _freqGov)
        (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
                df_anomalies, test_noAnomalies, test_anomalies
                ) = my_util.split_dataset_2(
                df_tensor, ts_noIdle, _freqGov)

    else:  # split in consecutive time-frames (before-after begin anomalies)
        prev_ts = ts_noIdle[0]
        for idx in range(len(ts_noIdle)):
            if prev_ts < _test_train_split_time <= ts_noIdle[idx]:
                test_idx = idx
                prev_ts = ts_noIdle[idx]
                break
        test_size = n_samples - test_idx
        x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    # (ALMOST) ALL DATA TO TRAIN SET
    #test_size = int(0.0001 * n_samples)           
    #if test_size == 0:
    #    test_size = 1
    #x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    if set_size != 'FULL':
        if set_size > len(x_train):
            return (-1, -1, -1, -1, -1, -1, -1)
        else:
            x_train = x_train[np.random.choice(
                x_train.shape[0], set_size, replace=False), :]

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)
    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    if os.path.isfile(model_name):
        autoencoder = load_model(model_name)
        ae_model_train_timer = time.time() 
    else:
        # Sparse
        encoded = Dense(n_features * 10, activation='relu',
                        activity_regularizer=regularizers.l1(1e-5))(input_data)
        decoded = Dense(n_features, activation='linear')(encoded)

        #encoded = Dense(n_features * 5, activation='relu',
        #                activity_regularizer=regularizers.l1(1e-5))(input_data)
        #encoded = Dense(n_features * 10, activation='relu',
        #                activity_regularizer=regularizers.l1(1e-5))(encoded)
        #encoded = Dense(n_features * 20, activation='relu',
        #                activity_regularizer=regularizers.l1(1e-5))(encoded)
        #decoded = Dense(n_features * 10, activation='relu',
        #                activity_regularizer=regularizers.l1(1e-5))(encoded)
        #decoded = Dense(n_features * 5, activation='relu',
        #                activity_regularizer=regularizers.l1(1e-5))(decoded)
        #decoded = Dense(n_features, activation='linear',
        #                activity_regularizer=regularizers.l1(1e-5))(decoded)

        # NOT Sparse
        #encoded = Dense(n_features, activation='relu')(input_data)
        #encoded = Dense(n_features/2, activation='relu')(encoded)
        #encoded = Dense(n_features/4, activation='relu')(encoded)
        #encoded = Dense(n_features/8, activation='relu')(encoded)
        #decoded = Dense(n_features/4, activation='relu')(encoded)
        #decoded = Dense(n_features/2, activation='relu')(decoded)
        #decoded = Dense(n_features, activation='linear')(decoded)

        #encoded = Dense(n_features / 8, activation='relu')(input_data)
        #decoded = Dense(n_features, activation='linear')(encoded)

        autoencoder = Model(input_data, decoded)

        ae_model_creation_timer = time.time() 
        ae_model_creation_time = ae_model_creation_timer - data_split_timer 
        print(">>> AE model created in %s s" % (ae_model_creation_time))

        #autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
        history = autoencoder.fit(x_train, x_train, epochs=_epochs, 
                batch_size=_batch_size, shuffle=True, 
                validation_data=(x_test, x_test),
                sample_weight=np.asarray(active_idle),
                verbose=1)

        ae_model_train_timer = time.time() 
        ae_model_train_time = ae_model_train_timer - ae_model_creation_timer
        print(">>> AE model trained in %s s" % (ae_model_train_time))

        autoencoder.save(model_name)

    # Errors distributions
    decoded_normal = autoencoder.predict(df_noAnomalies)
    decoded_anomal = autoencoder.predict(df_anomalies)
    decoded_normal_test = autoencoder.predict(test_noAnomalies)

    prediction_timer = time.time() 
    prediction_time = prediction_timer - ae_model_train_timer
    print(">>> Predictions obtained in %s s" % (prediction_time))

    plot_distrib = 1

    if plot_distrib == 1:  # study distribution of errors
        #my_util.error_distribution(df_noAnomalies, decoded_normal, 
        #        df_anomalies, decoded_anomal, test_nodes[0])
        #my_util.error_distribution_2_class(test_noAnomalies,
        #        decoded_normal_test, df_anomalies, decoded_anomal, 
        #        test_nodes[0], df_noAnomalies, decoded_normal)
        my_util.error_distribution_2_class_varyThreshold(test_noAnomalies,
                decoded_normal_test, df_anomalies, decoded_anomal, 
                test_nodes[0], df_noAnomalies, decoded_normal)

        error_distrib_timer = time.time()
        error_distrib_time = error_distrib_timer - prediction_timer
        print(">>> Error distribution analysed in %s s" % (error_distrib_time))
        stats_res = None

    elif plot_distrib == 0:  # plot errors
        # plot errors
        decoded_features = autoencoder.predict(df_tensor)
        stats_res = my_util.evaluate_predictions(
                decoded_features, df_tensor, gaps)
        #stats_res = my_util.evaluate_predictions_rolling_avg(
        #        decoded_features, df_tensor, gaps, N)
        prediction_timer = time.time() 
        prediction_time = prediction_timer - ae_model_train_timer
        print(">>> Predictions obtained in %s s" % (prediction_time))

        #my_util.visualize_train_history(history, [])
        #my_util.print_result(stats_res, series_labels, False)

        str_rolling = ''
        if N != 1:
            str_rolling = "Rolling Mean Window %s" % N
        my_util.plot_errors_DL(stats_res, series_labels, timestamps,
                _phys_sampling_time, gaps, idle_periods, [], ts_noIdle, ts_idle,
                test_idx, _freqGov, str_rolling)
    return stats_res

'''
Create a VAE with semisupervised scheme
'''
def VAE_semisupervised(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):
    ae_name = ('VAE_semisuper_rndmSplit_{}trainNode_DeBnDrDeDrDe_{}z_{}lrw_{}'
            'lkw_{}lr_{}dec_{}ep_{}bs_{}rm.h5'.format(node, _n_z, 
                _vae_loss_recon_weight, _vae_loss_kl_weight, _vae_lr, 
                _vae_decay, _epochs, _batch_size, N))
    model_name = trained_model_dir + ae_name

    if N != 1: # no rolling avg
        df = df.rolling(N, min_periods=1).mean()
    df_tensor = df.values
    n_samples, n_features = df_tensor.shape
    inputs = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)
    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    #if os.path.isfile(model_name):
    if False:
        vae = load_model(model_name, custom_objects={'vae_loss': vae_loss})
        ae_model_train_timer = time.time() 
    else:
        encoding_dim = int(n_features / 2)

        # NOT Sparse
        h_q_0 = Dense(encoding_dim, activation='elu')(inputs)
        h_q_0_bn_1 = BatchNormalization()(h_q_0)
        h_q_0_drop_1 = Dropout(0.2)(h_q_0_bn_1)
        h_q_1 = Dense(encoding_dim//2, activation='elu')(h_q_0_drop_1)
        h_q_1_drop_1 = Dropout(0.1)(h_q_1)
        h_q = Dense(encoding_dim//4, activation='elu')(h_q_1_drop_1)

        mu = Dense(_n_z, activation='linear')(h_q)
        log_sigma = Dense(_n_z, activation='linear')(h_q)

        # Sample z ~ Q(z|X)
        z = Lambda(sample_z, output_shape=(_n_z,))([mu, log_sigma])
        encoder = Model(inputs, [mu, log_sigma, z], name='encoder')
        latent_inputs = Input(shape=(_n_z,), name='z_sampling')

        x_1 = Dense(encoding_dim//4, activation='elu')(latent_inputs)
        dec_drop_0 = Dropout(0.2)(x_1)
        x_0 = Dense(encoding_dim//2, activation='elu')(dec_drop_0)
        dec_drop_1 = Dropout(0.2)(x_0)
        x = Dense(encoding_dim, activation='elu')(dec_drop_1)
        outputs = Dense(n_features, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # Overall VAE model, for reconstruction and training
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs)

        ae_model_creation_timer = time.time() 
        ae_model_creation_time = ae_model_creation_timer - data_split_timer 
        print(">>> AE model created in %s s" % (ae_model_creation_time))

        def vae_loss(y_true, y_pred):
            recon = K.mean(K.square(y_pred - y_true))
            kl = 0.5*K.sum(
                    K.exp(2*log_sigma) + K.square(mu) -1-2 * log_sigma,axis=1)
            return (_vae_loss_recon_weight*n_features*recon + 
                    _vae_loss_kl_weight*kl)

        adam = optimizers.Adam(lr=_vae_lr, decay = _vae_decay)
        vae.compile(optimizer=adam, loss=vae_loss)

        early_stopping = EarlyStopping(
                monitor='val_loss', patience=10, min_delta=1e-5) 
        reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2) 

        history = vae.fit(x_train, x_train, epochs=_epochs, 
                batch_size=_batch_size, shuffle=True, 
                validation_data=(x_test, x_test),
                sample_weight=np.asarray(active_idle),verbose=1, 
                callbacks=[early_stopping, reduce_lr])

        ae_model_train_timer = time.time() 
        ae_model_train_time = ae_model_train_timer - ae_model_creation_timer
        print(">>> AE model trained in %s s" % (ae_model_train_time))
        vae.save(model_name)

    # Errors distributions
    decoded_normal = vae.predict(df_noAnomalies)
    decoded_anomal = vae.predict(df_anomalies)
    decoded_normal_test = vae.predict(test_noAnomalies)

    prediction_timer = time.time() 
    prediction_time = prediction_timer - ae_model_train_timer
    print(">>> Predictions obtained in %s s" % (prediction_time))

    my_util.error_distribution_2_class_varyThreshold(test_noAnomalies,
            decoded_normal_test, df_anomalies, decoded_anomal, 
            test_nodes[0], df_noAnomalies, decoded_normal)

    error_distrib_timer = time.time()
    error_distrib_time = error_distrib_timer - prediction_timer
    print(">>> Error distribution analysed in %s s" % (error_distrib_time))
 

'''
Create an autoencoder to evaluate correlations between measurements, extract
    features 
    - data set not seen as a time series, but rather each time stamp with its
        measurements values is data point
    - Variational Autoencoder
    see https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
        https://blog.keras.io/building-autoencoders-in-keras.html
        https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''
def VAE(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):

    if N != 1: # no rolling avg
        df = df.rolling(N, min_periods=1).mean()

    df_tensor = df.values

    n_samples, n_features = df_tensor.shape

    inputs = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    idx_normal_anomalies = []
    for i in test_train_or_else_idxs:
        if i == 0 or i == 1:
            idx_normal_anomalies.append(0)
        elif i == 2:
            idx_normal_anomalies.append(1)
        elif i == 3:
            idx_normal_anomalies.append(2)

    # (ALMOST) ALL DATA TO TRAIN SET
    #test_size = int(0.0001 * n_samples)           
    test_size = int(0.001 * n_samples)           
    if test_size == 0:
        test_size = 1
    x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])
    idxs_train, idxs_test = (idx_normal_anomalies[:-test_size],
            idx_normal_anomalies[-test_size:])

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    #encoding_dim = _encoding_dim
    encoding_dim = int(n_features / 2)

    # NOT Sparse
    #h_q_0 = Dense(encoding_dim, activation='elu')(inputs)
    #h_q_0_bn_1 = BatchNormalization()(h_q_0)
    #h_q_0_drop_1 = Dropout(0.2)(h_q_0_bn_1)
    #h_q_1 = Dense(encoding_dim//2, activation='elu')(h_q_0_drop_1)
    #h_q_1_drop_1 = Dropout(0.1)(h_q_1)
    #h_q = Dense(encoding_dim//4, activation='elu')(h_q_1_drop_1)

    h_q_0 = Dense(encoding_dim, activation='elu')(inputs)
    h_q_0_drop_1 = Dropout(0.2)(h_q_0)
    h_q = Dense(encoding_dim//2, activation='elu')(h_q_0_drop_1)

    mu = Dense(_n_z, activation='linear')(h_q)
    log_sigma = Dense(_n_z, activation='linear')(h_q)

    # Sample z ~ Q(z|X)
    z = Lambda(sample_z, output_shape=(_n_z,))([mu, log_sigma])

    encoder = Model(inputs, [mu, log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(_n_z,), name='z_sampling')

    #x_1 = Dense(encoding_dim//4, activation='elu')(latent_inputs)
    #dec_drop_0 = Dropout(0.2)(x_1)
    #x_0 = Dense(encoding_dim//2, activation='elu')(dec_drop_0)
    #dec_drop_1 = Dropout(0.2)(x_0)
    #x = Dense(encoding_dim, activation='elu')(dec_drop_1)
    #outputs = Dense(n_features, activation='sigmoid')(x)

    x_1 = Dense(encoding_dim//2, activation='elu')(latent_inputs)
    dec_drop_0 = Dropout(0.2)(x_1)
    x = Dense(encoding_dim, activation='elu')(dec_drop_0)
    outputs = Dense(n_features, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # Overall VAE model, for reconstruction and training
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)
    #plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
    #vae.summary()

    ae_model_creation_timer = time.time() 
    ae_model_creation_time = ae_model_creation_timer - data_split_timer 
    print(">>> AE model created in %s s" % (ae_model_creation_time))

    '''
    Loss function to be used with VAE
    (see https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)
    - Calculate loss = reconstruction loss + KL loss for each data in minibatch
    '''
    def vae_loss(y_true, y_pred):
        #recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        recon = K.mean(K.square(y_pred - y_true))
        #recon = K.mean(K.abs(y_pred - y_true))

        #kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma,axis=1)
        kl = 0.5*K.sum(
                K.exp(2 * log_sigma) + K.square(mu) - 1 - 2 * log_sigma,axis=1)

        return _vae_loss_recon_weight*n_features*recon + _vae_loss_kl_weight*kl

    adam = optimizers.Adam(lr=_vae_lr, decay = _vae_decay)
    vae.compile(optimizer=adam, loss=vae_loss)

    class AnalyseLatentVar(Callback):
        def on_train_begin(self, logs={}):
            self.epoch_counter = 1

        def on_epoch_end(self, epoch, logs={}):

            if self.epoch_counter % _plot_interval == 0:
                z_mean, z_std, z = self.model.get_layer("encoder").predict(
                        df_tensor)
                plot_name = latentVar_dir + 'plot_%03d.png' % self.epoch_counter 

                my_util.plot_latent_var(z_mean, _n_z, idx_normal_anomalies, 
                        plot_name, self.epoch_counter)
                self.epoch_counter += 1

    early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=1e-5) 
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2) 

    analyse_latent_var = AnalyseLatentVar()
    history = vae.fit(x_train, x_train, epochs=_epochs, 
            batch_size=_batch_size, shuffle=True, 
            validation_data=(x_test, x_test),
            sample_weight=np.asarray(active_idle),verbose=1, 
            #callbacks=[early_stopping, reduce_lr, analyse_latent_var])
            callbacks=[early_stopping, reduce_lr])
            #callbacks=[])

    z_mean, z_std, z = encoder.predict(df_tensor, batch_size=_batch_size)

    centroids, pred_labels = my_util.cluster_latent_var(_n_behave_cluster, 
            z_mean, idx_normal_anomalies)
    ##my_util.plot_latent_var_withCluster(z_mean, _n_z, idx_normal_anomalies, 
    ##        'cluster_plot.png', centroids, pred_labels)
    my_util.plot_cluster_bars(z_mean, _n_z, idx_normal_anomalies, 
            pred_labels)
    #my_util.plot_latent_var(z_mean, _n_z, idx_normal_anomalies, '', -1, True)


'''
Create sparse AE and consider the activation patterns of its hidden layers.
    - check if with similar input the same hidden neurons fire together; this
    would identify "cluster" of behaviour
'''
def cluster_with_hidden_layer(df, timestamps, series_labels, test_node,
        gaps, idle_periods, active_idle=[], ts_idle=[], ts_noIdle=[]):

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    test_size = int(0.0001 * n_samples)           
    if test_size == 0:
        test_size = 1
    #x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])
    x_train, x_test = (df_tensor, df_tensor[-test_size:])

    input_data = Input(shape=(n_features,))
    encoded = Dense(n_features * 10, activation='relu',
            activity_regularizer=regularizers.l1(1e-5))(input_data)
    decoded = Dense(n_features, activation='sigmoid')(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

    #plot_model(autoencoder, to_file='model.png')

    history = autoencoder.fit(x_train, x_train, epochs=_epochs, 
            batch_size=_batch_size, shuffle=True, 
            validation_data=(x_test, x_test))

    #ae_name = 'AE_sparse_mutinode_encDim_519_nEpochs_100_nn_26_adam_MAE.h5'
    #autoencoder = load_model(ae_name)
    autoencoder.summary()

    decoded_features = autoencoder.predict(x_train)

    intermediate_layer_model = Model(inputs=autoencoder.input,
            outputs=autoencoder.get_layer("dense_1").output)
    intermediate_output = intermediate_layer_model.predict(x_train)
    intermediate_layer_model.summary()

    kmeans = KMeans(n_clusters=_n_behave_cluster)
    kmeans.fit(intermediate_output)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    cluster_predictions = kmeans.predict(intermediate_output)

    stats_res = my_util.evaluate_predictions(decoded_features, x_train, gaps)

    # !!! works only on single nodes !!!
    my_util.group_jobs_cluster(decoded_features, x_train, gaps, stats_res, 
            cluster_predictions, timestamps, test_node, active_idle, 
            ts_noIdle, ts_idle, test_node, test_start_time, test_end_time)

    my_util.combine_predictions_and_cluster(decoded_features, x_train, gaps,
            stats_res, cluster_predictions, timestamps, test_node,
            active_idle, ts_noIdle, ts_idle, _freqGov)

    #out_cl_layer = Dense(n_behave_cluster, activation='softmax')(
    #        cl_input_data)
    #clustering_model = Model(cl_input_data, out_cl_layer)
    #clustering_model.summary()

    return 1

'''
Create sparse AE and consider the activation patterns of its hidden layers.
    - check if with similar input the same hidden neurons fire together; this
    would identify "cluster" of behaviour
    - split data set in training and test set
'''
def cluster_with_hidden_layer_split(df, timestamps, series_labels, test_node,
        gaps, idle_periods, active_idle=[], ts_idle=[], ts_noIdle=[]):

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    input_data = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    if _split_random == 1:  # random, but exclude anomalies
        test_idx = 0
        x_train, x_test = my_util.split_dataset(df_tensor, ts_noIdle, _freqGov)
    else:  # split in consecutive time-frames (before-after begin anomalies)
        prev_ts = ts_noIdle[0]
        for idx in range(len(ts_noIdle)):
            if prev_ts < _test_train_split_time <= ts_noIdle[idx]:
                test_idx = idx
                prev_ts = ts_noIdle[idx]
                break
        print(">> Timestamp train/test split " + str(ts_noIdle[test_idx]))
        test_size = n_samples - test_idx
        x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    # (ALMOST) ALL DATA TO TRAIN SET
    #test_size = int(0.0001 * n_samples)           
    #if test_size == 0:
    #    test_size = 1
    #x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    encoded = Dense(n_features * 10, activation='relu',
            activity_regularizer=regularizers.l1(1e-5))(input_data)
    decoded = Dense(n_features, activation='sigmoid')(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

    #plot_model(autoencoder, to_file='model.png')

    history = autoencoder.fit(x_train, x_train, epochs=_epochs, 
            batch_size=_batch_size, shuffle=True, 
            validation_data=(x_test, x_test))

    #ae_name = 'AE_sparse_mutinode_encDim_519_nEpochs_100_nn_26_adam_MAE.h5'
    #autoencoder = load_model(ae_name)
    autoencoder.summary()

    decoded_features = autoencoder.predict(df_tensor)

    intermediate_layer_model = Model(inputs=autoencoder.input,
            outputs=autoencoder.get_layer("dense_1").output)
    intermediate_output = intermediate_layer_model.predict(df_tensor)
    intermediate_layer_model.summary()

    my_util.look_at_hidden_layer(intermediate_output, _freqGov, ts_noIdle)

    #kmeans = KMeans(n_clusters=_n_behave_cluster)
    #kmeans.fit(intermediate_output)
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_

    cluster_alg = SpectralClustering(n_clusters=_n_behave_cluster, 
            eigen_solver='arpack', affinity="nearest_neighbors")
    cluster_alg.fit(intermediate_output)
    labels = cluster_alg.labels_

    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    #cluster_predictions = kmeans.predict(intermediate_output)
    cluster_predictions = cluster_alg.labels_.astype(np.int)

    stats_res = my_uti34l.evaluate_predictions(decoded_features, df_tensor, gaps)

    my_util.combine_predictions_and_cluster_split(decoded_features, df_tensor,
            gaps, stats_res, cluster_predictions, timestamps, test_node,
            ts_noIdle, ts_idle, _freqGov)

    return 1

def settrainable(model, toset):
    for layer in model.layers:
        layer.trainable = toset
        model.trainable = toset
'''
Create an autoencoder to evaluate correlations between measurements, extract
    features 
    - data set not seen as a time series, but rather each time stamp with its
        measurements values is data point
    - Adversarial AE
    see https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py
    https://github.com/greentfrapp/keras-aae/blob/master/keras-aae.py
'''
def AAE1(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):

    if N != 1: # no rolling avg
        df = df.rolling(N, min_periods=1).mean()

    df_tensor = df.values

    n_samples, n_features = df_tensor.shape

    inputs = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    idx_normal_anomalies = []
    for i in test_train_or_else_idxs:
        if i == 0 or i == 1:
            idx_normal_anomalies.append(0)
        elif i == 2:
            idx_normal_anomalies.append(1)
        elif i == 3:
            idx_normal_anomalies.append(2)

    # (ALMOST) ALL DATA TO TRAIN SET
    test_size = int(0.0001 * n_samples)           
    if test_size == 0:
        test_size = 1
    x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    encoding_dim = n_features // 2

    # NOT Sparse
    h_q_0 = Dense(encoding_dim, activation='elu')(inputs)
    h_q_0_bn_1 = BatchNormalization()(h_q_0)
    h_q_0_drop_1 = Dropout(0.2)(h_q_0_bn_1)

    h_q_1 = Dense(encoding_dim//2, activation='elu')(h_q_0_drop_1)
    #h_q_1_bn_1 = BatchNormalization()(h_q_1)
    #h_q_1_drop_1 = Dropout(0.1)(h_q_1_bn_1)

    h_q = Dense(encoding_dim//4, activation='elu')(h_q_1)
    mu = Dense(_n_z, activation='linear')(h_q)
    log_sigma = Dense(_n_z, activation='linear')(h_q)

    # Sample z ~ Q(z|X)
    z = Lambda(sample_z, output_shape=(_n_z,))([mu, log_sigma])

    encoder = Model(inputs, [mu, log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(_n_z,), name='z_sampling')
    x_1 = Dense(encoding_dim//4, activation='elu')(latent_inputs)
    x_0 = Dense(encoding_dim//2, activation='elu')(x_1)
    x = Dense(encoding_dim, activation='elu')(x_0)
    outputs = Dense(n_features, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')

    # discriminator
    h_d_0 = Dense(encoding_dim, activation='elu')(latent_inputs)
    h_d_1 = Dense(encoding_dim//2, activation='elu')(h_d_0)
    h_d_2 = Dense(encoding_dim//4, activation='elu')(h_d_1)
    h_d_3 = Dense(1, activation='sigmoid')(h_d_2)

    discr = Model(latent_inputs, h_d_3, name='discriminator')

    outputs_enc_discr = discr(encoder(inputs)[2])
    enc_discr = Model(inputs, outputs_enc_discr)

    # Overall AE model, for reconstruction and training
    outputs = decoder(encoder(inputs)[2])
    ae = Model(inputs, outputs)

    ae_model_creation_timer = time.time() 
    ae_model_creation_time = ae_model_creation_timer - data_split_timer 
    print(">>> AE model created in %s s" % (ae_model_creation_time))

    ae.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")
    discr.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
    #enc_discr.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
    enc_discr.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")

    # TRAINING
    for epochnumber in range(_epochs):
        np.random.shuffle(x_train)
        for i in range(int(len(x_train) / _batch_size)):
            settrainable(vae, True)
            settrainable(encoder, True)
            settrainable(decoder, True)
            batch = x_train[i*_batch_size:i*_batch_size+_batch_size]
            ae.train_on_batch(batch, batch)

            settrainable(discr, True)
            #print(batch)
            batchpred = encoder.predict(batch)[2]
            fakepred = np.random.standard_normal((_batch_size,_n_z))
            #print(len(batchpred))
            ##print(len(fakepred))
            discbatch_x = np.concatenate([batchpred, fakepred])
            discbatch_y = np.concatenate([np.zeros(_batch_size), 
                np.ones(_batch_size)])
            discr.train_on_batch(discbatch_x, discbatch_y)

            settrainable(encoder, True)
            settrainable(enc_discr, True)
            settrainable(discr, False)
            enc_discr.train_on_batch(batch, np.ones(_batch_size))

        z_mean, z_std, _ = encoder.predict(df_tensor)
        plot_name = latentVar_dir + 'plot_%03d.png' % epochnumber
        my_util.plot_latent_var(z_mean, _n_z, idx_normal_anomalies, 
                plot_name, epochnumber)

        print("Epoch: %d/%d" % (epochnumber, _epochs))
        print("\tReconstruction Loss: %s" % vae.evaluate(
            x_train, x_train, verbose=0))
        print("\tAdversarial Loss: %s" % enc_discr.evaluate(
            x_train, np.ones(len(x_train)), verbose=0))

    z_mean, z_std, _ = encoder.predict(df_tensor, batch_size=_batch_size)

    my_util.cluster_latent_var(_n_behave_cluster, z_mean, z_std, 
            idx_normal_anomalies)

'''
Create an autoencoder to evaluate correlations between measurements, extract
    features 
    - data set not seen as a time series, but rather each time stamp with its
        measurements values is data point
    - Adversarial AE
    - second version
    see: https://github.com/greentfrapp/keras-aae/blob/master/keras-aae.py
'''
def AAE2(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):

    if N != 1: # no rolling avg
        df = df.rolling(N, min_periods=1).mean()

    df_tensor = df.values

    n_samples, n_features = df_tensor.shape

    inputs = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    idx_normal_anomalies = []
    for i in test_train_or_else_idxs:
        if i == 0 or i == 1:
            idx_normal_anomalies.append(0)
        elif i == 2:
            idx_normal_anomalies.append(1)
        elif i == 3:
            idx_normal_anomalies.append(2)

    # (ALMOST) ALL DATA TO TRAIN SET
    test_size = int(0.0001 * n_samples)           
    if test_size == 0:
        test_size = 1
    x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    encoding_dim = n_features // 2

    # NOT Sparse
    henc_0 = Dense(encoding_dim, activation='elu')(inputs)
    #h_q_0_drop_1 = Dropout(0.2)(h_q_0_bn_1)
    henc_1 = Dense(encoding_dim//2, activation='elu')(henc_0)
    #h_q_1_drop_1 = Dropout(0.1)(h_q_1_bn_1)
    henc_2 = Dense(encoding_dim//4, activation='elu')(henc_1)

    z = Dense(_n_z, activation='elu')(henc_2)

    #encoder = Model(inputs, z, name='encoder')

    latent_inputs = Input(shape=(_n_z,))
    hdec_0 = Dense(encoding_dim//4, activation='elu')(latent_inputs)
    hdec_1 = Dense(encoding_dim//2, activation='elu')(hdec_0)
    hdec_2 = Dense(encoding_dim, activation='elu')(hdec_1)
    outputs = Dense(n_features, activation='sigmoid')(hdec_2)

    #decoder = Model(latent_inputs, outputs, name='decoder')

    encoding_dim = n_features//2

    encoder = Sequential()
    encoder.add(Dense(encoding_dim, input_shape=(n_features,), activation='relu'))
    encoder.add(Dense(encoding_dim//2, activation='relu'))
    encoder.add(Dense(_n_z, activation=None))
        
    decoder = Sequential()
    decoder.add(Dense(encoding_dim//2, input_shape=(_n_z,), activation='relu'))
    decoder.add(Dense(encoding_dim, activation='relu'))
    decoder.add(Dense(n_features, activation='sigmoid'))

    discr = Sequential()
    discr.add(Dense(encoding_dim, input_shape=(_n_z,), activation='relu'))
    discr.add(Dense(encoding_dim, activation='relu'))
    discr.add(Dense(1, activation='sigmoid'))

    outputs = decoder(encoder(inputs))
    aae = Model(inputs, outputs)

    ae_model_creation_timer = time.time() 
    ae_model_creation_time = ae_model_creation_timer - data_split_timer 
    print(">>> AE model created in %s s" % (ae_model_creation_time))

    aae.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")
    discr.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
    discr.trainable = False

    discr.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")

    # training
    for epochnumber in range(_epochs):
        aae_losses = []
        discr_losses = []
        np.random.shuffle(x_train)
        for i in range(int(len(x_train) / _batch_size)):
            batch = x_train[i*_batch_size:i*_batch_size+_batch_size]
            aae_history = aae.fit(x=batch, y=batch, epochs=1, 
                    batch_size=_batch_size, validation_split=0.0, verbose=0)

            fake_latent = encoder.predict(batch)
            discr_input = np.concatenate((fake_latent, 
                np.random.randn(_batch_size, _n_z) * 5.))
            discr_labels = np.concatenate((np.zeros((_batch_size, 1)), 
                np.ones((_batch_size, 1))))
            discr_history = discr.fit(x=discr_input, y=discr_labels, epochs=1, 
                    batch_size=_batch_size, validation_split=0.0, verbose=0)

            aae_losses.append(aae_history.history["loss"])
            discr_losses.append(discr_history.history["loss"])

        z_temp = encoder.predict(df_tensor)
        plot_name = latentVar_dir + 'plot_%03d.png' % epochnumber
        my_util.plot_latent_var(z_temp, _n_z, idx_normal_anomalies, 
                plot_name, epochnumber)

        print("Epoch: %d/%d" % (epochnumber, _epochs))
        print("\tAutoencoder Loss: {}".format(np.mean(aae_losses)))
        print("\tDiscriminator Loss: {}".format(np.mean(discr_losses)))

    z = encoder.predict(df_tensor, batch_size=_batch_size)
    my_util.cluster_latent_var(_n_behave_cluster, z, [], 
            idx_normal_anomalies)

'''
Create an autoencoder to evaluate correlations between measurements, extract
    features 
    - data set not seen as a time series, but rather each time stamp with its
        measurements values is data point
    - Adversarial AE
    - third version
https://github.com/bstriner/keras-adversarial/blob/master/examples/example_aae.py
'''
def AAE3(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):

    z_distr = True

    if N != 1: # no rolling avg
        df = df.rolling(N, min_periods=1).mean()
    df_tensor = df.values
    n_samples, n_features = df_tensor.shape
    inputs = Input(shape=(n_features,))

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    idx_normal_anomalies = []
    for i in test_train_or_else_idxs:
        if i == 0 or i == 1:
            idx_normal_anomalies.append(0)
        elif i == 2:
            idx_normal_anomalies.append(1)
        elif i == 3:
            idx_normal_anomalies.append(2)

    # (ALMOST) ALL DATA TO TRAIN SET
    test_size = int(0.0001 * n_samples)           
    if test_size == 0:
        test_size = 1
    x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])

    # active_idle is a list that specifies if the sample correspond to 
    # a moment of activity or idleness
    # if the list is empty we assume that all samples are active
    if len(active_idle) == 0:
        active_idle = [1] * len(x_train)

    data_split_timer = time.time() 
    data_split_time = data_split_timer - ref_timer 
    print(">>> Data set split in %s s" % (data_split_time))

    encoding_dim = 500

    enc_0 = Dense(encoding_dim, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(inputs)
    enc_1 = Dense(encoding_dim//2, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(enc_0)

    if z_distr:
        mu = Dense(_n_z, activation=None, 
                activity_regularizer=regularizers.l1(1e-6))(enc_1)
        log_sigma = Dense(_n_z, activation=None, 
                activity_regularizer=regularizers.l1(1e-6))(enc_1)
        z = Lambda(sample_z, output_shape=(_n_z,))([mu, log_sigma])
        encoder = Model(inputs, [mu, log_sigma, z], name='encoder')
    else:
        z = Dense(_n_z, activation=None, 
                activity_regularizer=regularizers.l1(1e-6))(enc_1)
        encoder = Model(inputs, z, name='encoder')

    latent_inputs = Input(shape=(_n_z,), name='latent_inputs')
    dec_0 = Dense(encoding_dim//2, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(latent_inputs)
    dec_1 = Dense(encoding_dim, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(dec_0)
    outputs = Dense(n_features, activation='sigmoid')(dec_1)
    decoder = Model(latent_inputs, outputs, name='decoder')

    if z_distr:
        outputs = decoder(encoder(inputs)[2])
    else:
        outputs = decoder(encoder(inputs))
    ae = Model(inputs, outputs)

    discr_0 = Dense(encoding_dim, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(latent_inputs)
    discr_1 = Dense(encoding_dim//2, activation='elu', 
            activity_regularizer=regularizers.l1(1e-6))(discr_0)
    discr_outs = Dense(1, activation='sigmoid', 
            activity_regularizer=regularizers.l1(1e-6))(discr_1)
    discriminator = Model(latent_inputs, discr_outs, name='discr')

    optimizer = Adam(0.001, 0.9)
    #optimizer = Adam(0.0002, 0.5)

    discriminator.trainable = False
    discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])

    if z_distr:
        discr_outputs = discriminator(encoder(inputs)[2])
    else:
        discr_outputs = discriminator(encoder(inputs))

    aae = Model(inputs, [outputs, discr_outputs])
    aae.compile(loss=['mse', 'binary_crossentropy'], 
            #loss_weights=[0.999, 0.001], optimizer=optimizer)
            loss_weights=[0.99, 0.01], optimizer=optimizer)

    valid = np.ones((_batch_size, 1))
    fake = np.zeros((_batch_size, 1))
    for epoch in range(_epochs):
        # Select a random batch
        idx = np.random.randint(0, x_train.shape[0], _batch_size)
        x = x_train[idx]

        if z_distr:
            latent_fake_mu, latent_fake_std, latent_fake, = encoder.predict(x)
        else:
            latent_fake = encoder.predict(x)
        latent_real = np.random.normal(size=(_batch_size, _n_z))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(latent_real, valid)
        d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #discriminator.trainable = False
        #discriminator.compile(loss='binary_crossentropy',
        #        optimizer=optimizer, metrics=['accuracy'])

        # Train the generator
        g_loss = aae.train_on_batch(x, [x, valid])

        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
            epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

        if epoch % _plot_interval == 0:
            if z_distr:
                z_mean, z_std, z = aae.get_layer("encoder").predict(df_tensor)
            else:
                z = aae.get_layer("encoder").predict(df_tensor)
            plot_name = latentVar_dir + 'plot_%03d.png' % epoch
            my_util.plot_latent_var(z, _n_z, idx_normal_anomalies, 
                    plot_name, epoch)

    if z_distr:
        z_mean, z_std, z = aae.get_layer("encoder").predict(df_tensor)
        centroids, pred_labels = my_util.cluster_latent_var(
                _n_behave_cluster, z_mean, idx_normal_anomalies)
        my_util.plot_latent_var_withCluster(z_mean, _n_z, 
                idx_normal_anomalies,'cluster_plot.png',centroids, pred_labels)
    else:
        z = aae.get_layer("encoder").predict(df_tensor)
        centroids, pred_labels = my_util.cluster_latent_var(
                _n_behave_cluster, z, idx_normal_anomalies)
        my_util.plot_latent_var_withCluster(z, _n_z, 
                idx_normal_anomalies,'cluster_plot.png',centroids, pred_labels)

'''
Apply directly clustering techniques to data set
- no AE
'''
def pure_clustering(df, timestamps, series_labels, gaps, idle_periods, 
        ref_timer, active_idle=[], ts_idle=[], ts_noIdle=[]):
    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    idx_normal_anomalies = []
    for i in test_train_or_else_idxs:
        if i == 0 or i == 1:
            idx_normal_anomalies.append(0)
        elif i == 2:
            idx_normal_anomalies.append(1)
        elif i == 3:
            idx_normal_anomalies.append(2)

    # (ALMOST) ALL DATA TO TRAIN SET
    test_size = int(0.001 * n_samples)           
    if test_size == 0:
        test_size = 1
    x_train, x_test = (df_tensor[:-test_size], df_tensor[-test_size:])
    idxs_train, idxs_test = (idx_normal_anomalies[:-test_size],
            idx_normal_anomalies[-test_size:])

    my_util.plot_2d_proj(df_tensor, idx_normal_anomalies)
    sys.exit()

    n_clusters = 3

    ###########################################################################
    # DBSCAN
    #db = DBSCAN(eps=0.08, min_samples=10).fit(df_tensor)
    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    #labels = db.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    ###########################################################################

    ###########################################################################
    # KMeans
    #kmeans = KMeans(n_clusters=n_clusters)
    #kmeans.fit(df_tensor)
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    ###########################################################################

    ###########################################################################
    # Spectral Clustering
    #spectral = SpectralClustering(
    #        n_clusters=n_clusters, eigen_solver='arpack',
    #        affinity="nearest_neighbors")
    #spectral.fit(df_tensor)
    #labels = spectral.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    ###########################################################################

    ###########################################################################
    # Agglomerative Clustering
    aggCluster = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    aggCluster.fit(df_tensor)
    labels = aggCluster.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    ###########################################################################
    print("# unique labels: %s" % labels_unique)
    print("# Cluster input: %s" % n_clusters)
    print("# Cluster found: %s" % n_clusters_)

    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
            idx_normal_anomalies, labels, labels=[0])
    precision_A1, recall_A1, fscore_A1, xyz = precision_recall_fscore_support(
            idx_normal_anomalies, labels, labels=[1])
    precision_A2, recall_A2, fscore_A2, xyz = precision_recall_fscore_support(
            idx_normal_anomalies, labels, labels=[2])
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            idx_normal_anomalies, labels, average='weighted')

    print("Cluster evaluation: Multi-classes")
    print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
            recall_N, fscore_N))
    print("Anomaly 1: precision %f, recall %f, F-score %f" % (precision_A1, 
            recall_A1, fscore_A1))
    print("Anomaly 2: precision %f, recall %f, F-score %f" % (precision_A2, 
            recall_A2, fscore_A2))
    print("All Classes Weighted: precision %f, recall %f, F-score %f" % (
            precision_W, recall_W, fscore_W))

    my_util.plot_cluster_bars(df_tensor, n_features, idx_normal_anomalies,
            labels)

'''
Experiments with AE working on a single node
'''
def exps_AE_single_node(df_obj, ref_time):
    (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, scaler
            ) = (df_obj['df'], df_obj['timestamps'], df_obj['idle_periods'],
                    df_obj['active_idle'], df_obj['ts_noIdle'], 
                    df_obj['ts_idle'], df_obj['scaler'])

    # create timeseries
    timeseries = df.values

    gaps = my_util.find_gaps_timeseries(timestamps, _phys_sampling_time)
    find_gaps_timer = time.time() 
    find_gaps_time = find_gaps_timer - df_prepare_timer
    print(">>> Gaps found in %s s" % (find_gaps_time))

    # reduce the number of features
    delete_col = 0
    if delete_col == 1:
        c_to_del = []
        for c in df.columns:
            if 'notbze' in c:
                c_to_del.append(c)
            if 'fan' in c:
                c_to_del.append(c)
            if 'volt' in c:
                c_to_del.append(c)
            if 'sleepcn' in c:
                c_to_del.append(c)
            if 'system' in c:
                c_to_del.append(c)
            #if 'cmbw' in c:
            #    c_to_del.append(c)
            if 'cmbw_p0_2' in c:
                c_to_del.append(c)
            if 'cmbw_p0_3' in c:
                c_to_del.append(c)
            if 'cmbw_p0_4' in c:
                c_to_del.append(c)
            if 'cmbw_p0_5' in c:
                c_to_del.append(c)
            if 'cmbw_p0_6' in c:
                c_to_del.append(c)
            if 'cmbw_p0_7' in c:
                c_to_del.append(c)
            if 'cmbw_p0_8' in c:
                c_to_del.append(c)
            if 'cmbw_p0_9' in c:
                c_to_del.append(c)
            if 'cmbw_p0_14' in c:
                c_to_del.append(c)
            if 'cmbw_p0_15' in c:
                c_to_del.append(c)
            #if 'ambient' in c:
            #    c_to_del.append(c)
            if 'freq' in c:
                c_to_del.append(c)
            if 'mwr' in c:
                c_to_del.append(c)
            if 'mrd' in c:
                c_to_del.append(c)
            if 'notfin' in c:
                c_to_del.append(c)
            if 'vdd' in c:
                c_to_del.append(c)
            if 'util' in c:
                c_to_del.append(c)
            if 'wink' in c:
                c_to_del.append(c)
            if 'power' in c:
                c_to_del.append(c)
            if 'ips' in c:
                c_to_del.append(c)
            if 'pwr' in c:
                c_to_del.append(c)
            if 'temp_p0' in c:
                c_to_del.append(c)
        df = my_util.drop_stuff(df, c_to_del)

    res = correlation_autoencoder(df, timestamps, df.columns, 
            gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle, 
            _data_set_size)
    #res = VAE_semisupervised(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)
    #compare_different_AE(df, timestamps, df.columns, gaps,idle_periods)
    #res = modes_finder_AE(df, timestamps, df.columns, gaps, idle_periods)
    #res = cluster_with_hidden_layer(df, timestamps, df.columns, test_nodes[0],
    #        gaps, idle_periods, [], ts_idle, ts_noIdle)
    #res = cluster_with_hidden_layer_split(df, timestamps, df.columns, 
    #        test_nodes[0], gaps, idle_periods, [], ts_idle, ts_noIdle)
    #res = VAE(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)
    #res = AAE1(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)
    #res = AAE2(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)
    #res = AAE3(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)
    #res = pure_clustering(df, timestamps, df.columns, 
    #        gaps, idle_periods, find_gaps_timer, [], ts_idle, ts_noIdle)

'''
Supervised learning as discussed by Tuncer et al. in
Diagnosing Performance Variations in HPC Applications Using Machine Learning
- we cannot compute the set of features proposed by the authors and 
    we simply use average features (5 minutes average stored in cassandra)
'''
def supervised_classification(df_obj, ref_time):
    (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, scaler
            ) = (df_obj['df'], df_obj['timestamps'], df_obj['idle_periods'],
                    df_obj['active_idle'], df_obj['ts_noIdle'], 
                    df_obj['ts_idle'], df_obj['scaler'])

    feature_names = list(df.columns.values)

    gaps = my_util.find_gaps_timeseries(timestamps, _phys_sampling_time)
    find_gaps_timer = time.time() 
    find_gaps_time = find_gaps_timer - df_prepare_timer
    print(">>> Gaps found in %s s" % (find_gaps_time))

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    labels_binaryClass, labels_multiClass = my_util.get_labels(
            df_tensor, ts_noIdle, _freqGov)
    get_labels_timer = time.time() 
    get_labels_time = get_labels_timer - find_gaps_timer
    print(">>> Labels found in %s s" % (get_labels_time))


    train_data, test_data, train_target, test_target = train_test_split(
            #df_tensor, labels_binaryClass, test_size = 0.3, random_state = 42)
            df_tensor, labels_multiClass, test_size = 0.3, random_state = 42)

    print("Train data len %s" % len(train_data))
    print("Test data len %s" % len(test_data))

    classifier = RandomForestClassifier()
    #classifier = DecisionTreeClassifier()
    classifier.fit(train_data,train_target)
    class_train_timer = time.time() 
    class_train_time = class_train_timer - get_labels_timer 
    print(">>> Classifier trained in %s s" % (class_train_time))

    predictions = classifier.predict(test_data)

    #precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
    #        test_target, predictions, average='binary', pos_label=0)
    #precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
    #        test_target, predictions, average='binary', pos_label=1)

    #print("len test data = %s" % (len(test_data)))
    #print("Test data")
    #print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
    #        recall_N, fscore_N))
    #print("Anomaly: precision %f, recall %f, F-score %f" % (precision_A, 
    #        recall_A, fscore_A))

    #predictions_all = classifier.predict(df)
    #precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
    #        labels_binaryClass, predictions_all, average='binary', pos_label=0)
    #precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
    #        labels_binaryClass, predictions_all, average='binary', pos_label=1)

    #print("len all data = %s" % (len(df)))
    #print("All data")
    #print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
    #        recall_N, fscore_N))
    #print("Anomaly: precision %f, recall %f, F-score %f" % (precision_A, 
    #        recall_A, fscore_A))

    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
            test_target, predictions, labels=[0])
    precision_A1, recall_A1, fscore_A1, xyz = precision_recall_fscore_support(
            test_target, predictions, labels=[1])
    precision_A2, recall_A2, fscore_A2, xyz = precision_recall_fscore_support(
            test_target, predictions, labels=[2])
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            test_target, predictions, average='weighted')

    print("Classifier evaluation: Multi-classes")
    print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
            recall_N, fscore_N))
    print("Anomaly 1: precision %f, recall %f, F-score %f" % (precision_A1, 
            recall_A1, fscore_A1))
    print("Anomaly 2: precision %f, recall %f, F-score %f" % (precision_A2, 
            recall_A2, fscore_A2))
    print("All Classes Weighted: precision %f, recall %f, F-score %f" % (
            precision_W, recall_W, fscore_W))

'''
Experiments with alternative semi-supervised approaches.
The algorithms are trained with normal behaviour and have to distinguish
anomalous points (no distinction between anomaly types)
'''
def semisupervised_classification_singleRun(df_obj, ref_time):
    (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, scaler
            ) = (df_obj['df'], df_obj['timestamps'], df_obj['idle_periods'],
                    df_obj['active_idle'], df_obj['ts_noIdle'], 
                    df_obj['ts_idle'], df_obj['scaler'])
    df_tensor = df.values
    # DATA SPLIT IN TEST/TRAIN
    test_idx = 0
    (x_train, x_test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, test_anomalies
            ) = my_util.split_dataset_2(
            df_tensor, ts_noIdle, _freqGov)

    gaps = my_util.find_gaps_timeseries(timestamps, _phys_sampling_time)
    find_gaps_timer = time.time() 
    find_gaps_time = find_gaps_timer - df_prepare_timer

    ###################################################################
    # SVM One Class with sklearn; at test time, 1 means that it predicts a 
    # normal class, -1 is an anomaly
    #normal_class = 1
    #anomaly_class = -1
    ##classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    #classifier = svm.OneClassSVM(nu=0.5, kernel="poly", gamma=0.1)
    #classifier.fit(x_train)
    #predictions_NA = classifier.predict(test_noAnomalies)
    #predictions_A = classifier.predict(test_anomalies)
    #predictions_NA_A = np.concatenate((predictions_NA, predictions_A), axis=0)
    ###################################################################

    ###################################################################
    # Gaussian Model Mixture
    classifier = GMM(n_components=2, covariance_type='full')
    #classifier.fit(x_train)
    classifier.fit(df_tensor)
    predictions_NA = classifier.predict(test_noAnomalies)
    predictions_A = classifier.predict(test_anomalies)
    predictions_train = classifier.predict(x_train)
    predictions_NA_A = np.concatenate((predictions_NA, predictions_A), axis=0)
    # it's not clear what is the value of the normal class with GMM..
    # here we generously assume that it classified correctly the majority of
    # training samples without anomaly and we infer the corresponding classes
    n_0 = 0
    n_1 = 0
    for p in predictions_train:
        if p == 1:
            n_1 +=1
        else:
            n_0 +=1
    if n_0 > n_1:
        normal_class = 0
        anomaly_class = 1
    else:
        normal_class = 1
        anomaly_class = 0
    ###################################################################

    ###################################################################
    # Isolation Forest
    #classifier = IsolationForest(contamination=0.1)
    #classifier.fit(x_train)
    #predictions_NA = classifier.predict(test_noAnomalies)
    #predictions_A = classifier.predict(test_anomalies)
    #predictions_NA_A = np.concatenate((predictions_NA, predictions_A), axis=0)
    #normal_class = 1
    #anomaly_class = -1
    ###################################################################

    ###################################################################
    # Elliptic Envelope
    #classifier = EllipticEnvelope(contamination=0.1)
    #classifier.fit(x_train)
    #predictions_NA = classifier.predict(test_noAnomalies)
    #predictions_A = classifier.predict(test_anomalies)
    #predictions_NA_A = np.concatenate((predictions_NA, predictions_A), axis=0)
    #normal_class = 1
    #anomaly_class = -1
    ###################################################################

    class_train_timer = time.time() 
    class_train_time = class_train_timer - find_gaps_timer

    labels_test_A = [anomaly_class for t in test_anomalies]
    labels_test_NA = [normal_class for t in test_noAnomalies]
    labels_test_NA_A = labels_test_NA + labels_test_A

    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
            labels_test_NA_A, predictions_NA_A, labels=[normal_class])
    precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
            labels_test_NA_A, predictions_NA_A, labels=[anomaly_class])
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            labels_test_NA_A, predictions_NA_A, average='weighted')

    return (precision_N, recall_N, fscore_N, precision_A, recall_A, fscore_A,
            precision_W, recall_W, fscore_W)

def semisupervised_classification(df_obj, ref_time, node, 
        acc_res_file='acc_result'):
    n_iter = 10
    list_precision_N = []
    list_recall_N = []
    list_fscore_N = []
    list_precision_A = []
    list_recall_A = []
    list_fscore_A = []
    list_precision_W = []
    list_recall_W = []
    list_fscore_W = []
    print("{}".format(node))
    for i in range(n_iter):
        print("\tIter n. {}".format(i))
        (pN, rN, fN, pA, rA, fA, pW, rW, fW
                ) = semisupervised_classification_singleRun(df_obj, ref_time)
        list_precision_N.append(pN)
        list_recall_N.append(rN)
        list_fscore_N.append(fN)
        list_precision_A.append(pA)
        list_recall_A.append(rA)
        list_fscore_A.append(fA)
        list_precision_W.append(pW)
        list_recall_W.append(rW)
        list_fscore_W.append(fW)
    precision_N = np.mean(np.asarray(list_precision_N))
    recall_N = np.mean(np.asarray(list_recall_N))
    fscore_N = np.mean(np.asarray(list_fscore_N))
    precision_A = np.mean(np.asarray(list_precision_A))
    recall_A = np.mean(np.asarray(list_recall_A))
    fscore_A = np.mean(np.asarray(list_fscore_A))
    precision_W = np.mean(np.asarray(list_precision_W))
    recall_W = np.mean(np.asarray(list_recall_W))
    fscore_W = np.mean(np.asarray(list_fscore_W))
    print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
            recall_N, fscore_N))
    print("Anomaly: precision %f, recall %f, F-score %f" % (precision_A, 
            recall_A, fscore_A))
    print("All Classes Weighted: precision %f, recall %f, F-score %f" % (
            precision_W, recall_W, fscore_W))

    wrtxt = "\n{}\n".format(node)
    wrtxt+="Normal: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
            precision_N, recall_N, fscore_N)
    wrtxt+="Anomaly: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
            precision_A, recall_A, fscore_A)
    wrtxt+="Weighted: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
            precision_W, recall_W, fscore_W)
    #with open(acc_res_file, 'a') as of:
    #    of.write(wrtxt)
    return (precision_N, recall_N, fscore_N, precision_A, recall_A, fscore_A,
            precision_W, recall_W, fscore_W)

def semisupervised_classification_multinode(df_obj, df_prepare_timer, nodes):
    rdir = '/home/b0rgh/data_collection_analysis/data_analysis/davide/'
    rdir += 'anomaly_detection/results/semisupervised/'
    #rdir += 'elliptic_envelope/'
    #acc_res_file = rdir + 'accuracy_01contamination_multiRuns'
    #rdir += 'isolationForest/'
    #acc_res_file = rdir + 'accuracy_01contamination_multiRuns'
    #rdir += 'svm_one_class/'
    #acc_res_file = rdir + 'accuracy_poly_kernel_05nu_01gamma_multiRuns'
    ##acc_res_file = rdir + 'accuracy_rbf_kernel_01nu_01gamma_multiRuns'
    rdir += 'gmm/'
    #acc_res_file = rdir + 'accuracy_full_covar_multiRuns'
    #acc_res_file = rdir + 'accuracy_tied_covar_multiRuns'
    acc_res_file = rdir + 'accuracy_diag_covar_multiRuns'
    #acc_res_file = rdir + 'accuracy_spherical_covar_multiRuns'

    list_precision_N = []
    list_recall_N = []
    list_fscore_N = []
    list_precision_A = []
    list_recall_A = []
    list_fscore_A = []
    list_precision_W = []
    list_recall_W = []
    list_fscore_W = []
    for node in nodes:
        (pN, rN, fN, pA, rA, fA, pW, rW, fW
                ) = semisupervised_classification(df_obj, df_prepare_timer, 
                        node, acc_res_file)
        list_precision_N.append(pN)
        list_recall_N.append(rN)
        list_fscore_N.append(fN)
        list_precision_A.append(pA)
        list_recall_A.append(rA)
        list_fscore_A.append(fA)
        list_precision_W.append(pW)
        list_recall_W.append(rW)
        list_fscore_W.append(fW)
    precision_N = np.mean(np.asarray(list_precision_N))
    recall_N = np.mean(np.asarray(list_recall_N))
    fscore_N = np.mean(np.asarray(list_fscore_N))
    precision_A = np.mean(np.asarray(list_precision_A))
    recall_A = np.mean(np.asarray(list_recall_A))
    fscore_A = np.mean(np.asarray(list_fscore_A))
    precision_W = np.mean(np.asarray(list_precision_W))
    recall_W = np.mean(np.asarray(list_recall_W))
    fscore_W = np.mean(np.asarray(list_fscore_W))
    #wrtxt = "\nAverage\n"
    #wrtxt+="Normal: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
    #        precision_N, recall_N, fscore_N)
    #wrtxt+="Anomaly: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
    #        precision_A, recall_A, fscore_A)
    #wrtxt+="Weighted: precision {0:.3f}, recall {1:.3f}, F-score {2:.3f}\n".format(
    #        precision_W, recall_W, fscore_W)
    #with open(acc_res_file, 'a') as of:
    #    of.write(wrtxt)

if __name__ == "__main__":  
    print("================================================================")
    _start_timer = time.time()

    # Experiments with AE working on a single node
    # suppose we have only one node
    node = test_nodes[0]

    df_already_prepared, offline_df_file = my_util.check_df(start_time, 
            end_time, node, _remove_idle)

    if not df_already_prepared:
        print("Retrieve data")
        data_ipmi, data_bbb, data_occ = my_util.retrieve_data(
                start_time, end_time, test_nodes)
        retrieve_timer = time.time() 
        retrieve_time = retrieve_timer - _start_timer
        print(">>> Retrieval completed in %s s" % (retrieve_time))
        phys_data = my_util.create_df(data_ipmi, data_bbb, data_occ)
        df_create_timer = time.time() 
        df_create_time = df_create_timer - retrieve_timer

        df = pd.DataFrame(phys_data[node])

        df = df.transpose()

        (df, timestamps, idle_periods, active_idle, ts_noIdle, ts_idle, 
                scaler) = my_util.prepare_dataframe(df, _remove_idle)
        df_prepare_timer = time.time() 
        df_prepare_time = df_prepare_timer - df_create_timer
        print(">>> DF prepared in %s s" % (df_prepare_time))

        df_obj = {}
        df_obj['df'] = df
        df_obj['timestamps'] = timestamps
        df_obj['idle_periods'] = idle_periods
        df_obj['active_idle'] = active_idle
        df_obj['ts_noIdle'] = ts_noIdle
        df_obj['ts_idle'] = ts_idle
        df_obj['scaler'] = scaler

        with open(offline_df_file, 'wb') as handle:
            pickle.dump(df_obj, handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(offline_df_file, 'rb') as handle:
            df_obj = pickle.load(handle, encoding='latin1')
        df_prepare_timer = time.time() 
        df_prepare_time = df_prepare_timer - _start_timer
        print(">>> DF loaded in %s s" % (df_prepare_time))

    exp_mode = 0
    if exp_mode == 0:
        exps_AE_single_node(df_obj, df_prepare_timer)
    elif exp_mode == 1:
        # Experiments with supervised learning (single nodes)
        supervised_classification(df_obj, df_prepare_timer)
    elif exp_mode == 2:
        # Experiments with allternative methods for semi-supervised learning
        #semisupervised_classification(df_obj, df_prepare_timer, test_nodes[0])
        semisupervised_classification_multinode(
                df_obj, df_prepare_timer, test_nodes)


