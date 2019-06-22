'''
Miscellaneous utility functions

Author: Andrea Borghesi
    University of Bologna
'''

import os
import sys
import math
from decimal import *
import collections
import operator
import json
import pickle
import datetime
import argparse
import configparser
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pylab as pl
import time
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np
import scipy.stats as stats
import itertools as it

_threshold_feat = 0.15

'''
The following list indicates  the period when anomalies were injected in DAVIDE
nodes (mis-configurations)
The dates are converted to UTC
'''
_anomaly_periods = [
        (datetime.datetime.strptime("2018-04-12 11:50:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-04-13 09:08:00","%Y-%m-%d %H:%M:%S")
            ),
        (datetime.datetime.strptime("2018-04-14 10:07:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-04-14 17:15:00","%Y-%m-%d %H:%M:%S")
            ),
        (datetime.datetime.strptime("2018-04-16 10:37:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-04-16 13:00:00","%Y-%m-%d %H:%M:%S")
            ),
        (datetime.datetime.strptime("2018-04-17 07:32:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-04-20 17:00:00","%Y-%m-%d %H:%M:%S")
            ),
        (datetime.datetime.strptime("2018-04-24 10:20:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-05-04 06:58:00","%Y-%m-%d %H:%M:%S")
            ),
        (datetime.datetime.strptime("2018-05-23 17:15:00","%Y-%m-%d %H:%M:%S"),
            # fake date -- still currently still ongoing (2018-05-28)
            datetime.datetime.strptime("2018-05-26 00:00:00","%Y-%m-%d %H:%M:%S")
            )
        ]

def unix_time_millis(dt):
    return long((dt - epoch).total_seconds() * 1000.0)
def millis_unix_time(millis):
    seconds = millis / 1000
    return epoch + datetime.timedelta(seconds=seconds)

'''
Load information from physical sensors
'''
def load_data(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None

'''
Drop unused features and rows with NaN
'''
def drop_stuff(df, features_to_be_dropped):
    for fd in features_to_be_dropped:
        if fd in df:
            del df[fd]
    new_df = df.dropna(axis=0, how='all')
    new_df = new_df.dropna(axis=1, how='all')
    new_df = new_df.fillna(0)
    return new_df

'''
Pre-process input data.
Encode the categorical features
'''
def preprocess_noScaling(df, categorical_features, continuous_features):
    for c in categorical_features:
        df = encode_category(df, c)
    return df

'''
Pre-process input data.
Scale continuous features and encode the categorical ones
'''
def preprocess(df, categorical_features, continuous_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    for c in categorical_features:
        df = encode_category(df, c)
    return df, scaler

'''
Encode categorical values into with one-hot encoding
'''
def encode_category(data_set_in, c):
    if c not in data_set_in:
        return data_set_in
    dummy_cols =  pd.get_dummies(data_set_in[c], dummy_na=False)
    data_set = pd.concat([data_set_in, dummy_cols], axis=1)
    del data_set[c]
    return data_set

'''
Evaluate prediction
'''
def evaluate_predictions(predicted, actual, gaps):
    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    nb_samples, nb_series = actual.shape
    abs_errors = {}
    p_abs_errors = {}
    sp_abs_errors = {}
    squared_errors = {}
    MAE = {}
    MAPE = {}
    SMAPE = {}
    MSE = {}
    RMSE = {}
    R2 = {}

    actual_t = {}
    pred_t = {}

    for j in range(nb_series):
        abs_errors[j] = []
        p_abs_errors[j] = []
        sp_abs_errors[j] = []
        squared_errors[j] = []
        actual_t[j] = []
        pred_t[j] = []

    for i in range(nb_samples):
        for j in range(nb_series):
            abs_errors[j].append(abs(predicted[i][j] - actual[i][j]))
            squared_errors[j].append((predicted[i][j] - actual[i][j])*
                (predicted[i][j] - actual[i][j]))
            if actual[i][j] != 0:
                p_abs_errors[j].append((abs(predicted[i][j]-actual[i][j]))* 
                        100 / actual[i][j])
            sp_abs_errors[j].append((abs(predicted[i][j]-actual[i][j])) * 100 / 
                (predicted[i][j] + actual[i][j]))
            actual_t[j].append(actual[i][j])
            pred_t[j].append(predicted[i][j])

    for j in range(nb_series):
        MAE[j] = Decimal(np.mean(np.asarray(abs_errors[j])))
        MAPE[j] = Decimal(np.mean(np.asarray(p_abs_errors[j])))
        SMAPE[j] = Decimal(np.nanmean(np.asarray(sp_abs_errors[j])))
        MSE[j] = Decimal(np.mean(np.asarray(squared_errors[j])))
        RMSE[j] = Decimal(math.sqrt(MSE[j]))
        R2[j] = r2_score(actual[j], predicted[j])

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors
    stats_res["R2"] = R2

    cumul_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    std_feature_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    avg_feature_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
            cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

    # 'normalize' cumulated errors
    cumul_errors_norm = []
    for ce in cumul_errors:
        cumul_errors_norm.append(ce / len(stats_res["ABS_ERRORS"]))
    stats_res["CUMUL_ABS_ERRORS"] = cumul_errors
    stats_res["CUMUL_ABS_ERRORS_NORM"] = cumul_errors_norm

    return stats_res

'''
Plot errors generated with DL models (prediction errors..)
'''
def plot_errors_DL(stats_res, series_labels, timestamps, 
        sampling_time, gaps, idle_periods, samples_idleness=[], 
        ts_noIdle=[], ts_idle=[], test_idx=-1, freqGov="conservative",
        title="Error"):
    if len(ts_noIdle) > 0:
        plot_errors_DL_fixIdle(stats_res, series_labels, timestamps, 
                sampling_time, gaps, idle_periods, ts_noIdle, ts_idle, 
                test_idx, freqGov, title)
        return
    cumul_errors = []
    time_idxs = []
    for i in range(len(stats_res["ABS_ERRORS"][0])):
        cumul_errors.append(0)
        time_idxs.append(i)
    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
            cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

    # 'normalize' cumulated errors
    cumul_errors_norm = []
    for ce in cumul_errors:
        cumul_errors_norm.append(ce / len(stats_res["ABS_ERRORS"]))

    if len(samples_idleness) > 0:
        for i in range(len(cumul_errors_norm)):
            print("Error norm: %s; Idle: %s" % (
                cumul_errors_norm[i], samples_idleness[i]))

    fig = plt.figure()
    if len(timestamps) != len(cumul_errors_norm):
        if len(timestamps) > len(cumul_errors_norm):
            dif = len(timestamps) - len(cumul_errors_norm)
            timestamps = timestamps[dif:]
        else:
            dif = len(cumul_errors_norm) - len(timestamps)
            cumul_errors_norm = cumul_errors_norm[dif:]

    plt.plot(timestamps, cumul_errors_norm)
    if test_idx != -1:
        test_timestamp = ts_noIdle[test_idx]
        plt.axvline(x=test_timestamp, linewidth=2, 
                linestyle='--', color='k')

    ax = fig.add_subplot(111)
    for st, et in gaps:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='/', 
                color='grey')
        ax.add_patch(rect)
    for st, et in idle_periods:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='//', 
                color='yellow')
        ax.add_patch(rect)
    add_freq_govs_to_plot(freqGov, ax)
    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(title)
    plt.show()

'''
[s0, s1, s2, s3, s5,  ..]
s -> (s0, s1), (s1, s2), (s2, s3), ...
'''
def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def is_in_timeseries(timestamps, ts, sampling_time):
    for ts_1, ts_2 in pairwise(timestamps):
        if(ts_1 <= ts <= ts_2 and 
                (ts_1 - ts_2).total_seconds() <= sampling_time):
            return True
    return False

'''
Find gaps in time series.
Given a set of time stamps (our time series), find missing values 
'''
def find_gaps_timeseries(timestamps, sampling_time):
    gap_list = []
    for ts1, ts2 in pairwise(timestamps):
        tdif = (ts2 - ts1).total_seconds()
        if tdif > 1.1*sampling_time:
            gap_list.append((ts1,ts2))
    return gap_list

'''
Find time periods when a node was practically empty 
Return a list of time periods [(st_0, et_0), (st_1,et_1),...]
- a node is considered idle if all of its core have a very low load
'''
def find_idle_periods(df, timestamps, scaled):
    idle_periods = []
    idle_timestamps = []
    load_cols = []
    for c in df:
        if "util" in c:
            load_cols.append(c)
    prev = False
    if scaled:
        load_low = 0.1
    else:
        load_low = 10
    for index, row in df.iterrows():
        node_idle = is_node_idle(row, load_low, load_cols)
        if node_idle and not prev:
            start_ts = index
            prev_ts = index
            prev = True
            idle_timestamps.append(index)
        elif node_idle and prev:
            prev_ts = index
            idle_timestamps.append(index)
        elif not node_idle and prev:
            idle_periods.append((start_ts, prev_ts))
            prev = False
    # last round can be not considered properly -- add it now
    if prev:
        idle_periods.append((start_ts, prev_ts))

    return idle_periods

'''
Return if the particular timestamp correspond to an idle period
'''
def is_node_idle(df_row, idle_threshold, load_cols):
    cumul_load = 0
    for c in load_cols:
        cumul_load += df_row[c]
    if cumul_load < idle_threshold:
        return True
    else:
        return False

'''
Count number of idle periods in a data frame and return list where
each element corresponds to a row of the data frame and has value 1
if in the corresponding period the node was active, 0 if it was idle
OUT: list where each elements specifies if during the corresponding
    (same index) timestamp the node was active (1) or not (0)
'''
def count_idle_periods(df, timestamps, scaled):
    n_idle = 0
    n_tot = 0
    load_cols = []
    active_idle = []
    for c in df:
        if "util" in c:
            load_cols.append(c)
    prev = False
    if scaled:
        load_low = 0.1
    else:
        load_low = 10
    for index, row in df.iterrows():
        node_idle = is_node_idle(row, load_low, load_cols)
        n_tot += 1
        if node_idle:
            n_idle += 1
            active_idle.append(0)
        else:
            active_idle.append(1)

    print("# Periods: %s" % n_tot)
    print("# Idle Periods: %s" % n_idle)
    print("Perc. Idle: %s" % ((n_idle / float(n_tot))*100))
    return active_idle

'''
Add frequency governor types to plot
'''
def add_freq_govs_to_plot(freqGov, ax):
    if freqGov == 'performance':
        freqGov_periods = _anomaly_periods
        color = 'blue'
    elif freqGov == 'powersave':
        freqGov_periods = _anomaly_periods
        color = 'red'
    else:
        freqGov_periods = []

    for st, et in freqGov_periods:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, -0.002), dif, 0.012, fill=True, 
                color=color)
        ax.add_patch(rect)

'''
Retrieve physical data from Cassandra or local host
'''
def retrieve_data(start_time, end_time, node, data_dir):
    phys_data = {}
    phys_data_ipmi = {}
    phys_data_bbb = {}
    phys_data_occ = {}

    # IPMI data
    ipmi_file = data_dir + node + '_ipmi_'
    ipmi_file += str(start_time).replace('-','').replace(
            ':','').replace(' ','_') + '_'
    impi_file += str(end_time).replace('-','').replace(
            ':','').replace(' ','_') + '.pickle'
    phys_data_ipmi[node] = load_data(ipmi_file)
    if phys_data_ipmi[node] == None:
        print("--- IPMI data for node {} unavailable ---".format(node))

    # OCC data
    occ_file = data_dir + node + '_occ_'
    occ_file += str(start_time).replace('-','').replace(
            ':','').replace(' ','_') + '_'
    occ_file += str(end_time).replace('-','').replace(
            ':','').replace(' ','_') + '.pickle'
    phys_data_occ[node] = load_data(occ_file)
    if phys_data_occ[node] == None:
        print("--- OCC data for node {} unavailable ---".format(node))

    # BBB data (BeagleBoneBlack, fine grained power measurements)
    bbb_file = data_dir + node + '_bbb_'
    bbb_file += str(start_time).replace('-','').replace(
            ':','').replace(' ','_') + '_'
    bbb_file += str(end_time).replace('-','').replace(
            ':','').replace(' ','_') + '.pickle'
    phys_data_bbb[node] = load_data(bbb_file)
    if phys_data_bbb[node] == None:
        print("--- BBB data for node {} unavailable ---".format(node))

    return phys_data_ipmi, phys_data_bbb, phys_data_occ

'''
Fix wrong values, sort, merge
'''
def create_df(phys_data_ipmi, phys_data_bbb, phys_data_occ):
    print('\t >> Creating df')
    # OCC measurements appear to have a 'wrong' timestamp, one second later
    # than the real value 
    nodes = phys_data_ipmi.keys()
    phys_data_occ_fixed = {}
    for n in nodes:
        print("\t\t >> Fixing node %s" % n)
        phys_data_occ_fixed[n] = {}
        for timestamp in phys_data_occ[n].keys():
            if str(timestamp).split(':')[-1] != '00':
                new_timestamp = timestamp - datetime.timedelta(seconds=1)
            else:
                new_timestamp = timestamp
            phys_data_occ_fixed[n][new_timestamp] = phys_data_occ[n][timestamp]

    for n in nodes:
        print("\t\t >> Sorting node %s" % n)
        phys_data_ipmi[n] = collections.OrderedDict(
                sorted(phys_data_ipmi[n].items()))
        phys_data_bbb[n] = collections.OrderedDict(
                sorted(phys_data_bbb[n].items()))
        phys_data_occ[n] = collections.OrderedDict(
                sorted(phys_data_occ_fixed[n].items()))

    phys_data = {}
    for n in nodes:
        print("\t\t >> Merging IPMI,BBB,OCC node %s" % n)
        phys_data[n] = {}
        for k in phys_data_ipmi[n].keys():
            phys_data[n][k] = phys_data_ipmi[n][k]
            if k in phys_data_bbb[n]:
                phys_data[n][k].update(phys_data_bbb[n][k])
            if k in phys_data_occ[n]:
                phys_data[n][k].update(phys_data_occ[n][k])
    return phys_data

'''
Plot errors generated with DL models (prediction errors..)
- in this case the model was created and trained with a data set without
idle periods. To create a nice plot we have to reinsert them
'''
def plot_errors_DL_fixIdle(stats_res, series_labels, timestamps, 
        sampling_time, gaps, idle_periods, ts_noIdle, ts_idle, 
        test_idx, freqGov, title):
    time_idxs = []
    errors = []
    errors_stat = []

    for i in range(len(ts_noIdle)):
        errors.append(0)
    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
                errors[i] += stats_res["ABS_ERRORS"][j][i]
    for ce in errors:
        errors_stat.append(ce / len(stats_res["ABS_ERRORS"]))

    # we assign an error equal to 0 to idle periods
    errors_withIdle = []
    errors_stat_withIdle = []
    k = 0
    for i in range(len(timestamps)):
        if timestamps[i] in ts_noIdle:
            errors_withIdle.append(errors[k])
            errors_stat_withIdle.append(errors_stat[k])
            k += 1
        else: 
            errors_withIdle.append(0)
            errors_stat_withIdle.append(0)
    n_idle = 0
    for i in range(len(errors_stat_withIdle)):
        if errors_stat_withIdle[i] == 0:
            n_idle += 1

    fig = plt.figure()
    plt.plot(timestamps, errors_stat_withIdle, linewidth=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = fig.add_subplot(111)
    add_freq_govs_to_plot(freqGov, ax)
    plt.title(title)
    plt.show()

'''
Check if a data frame has already been prepared
- no need to load it and prepare it twice
'''
def check_df(data_dir, start_time, end_time, node, remove_idle, unscaled=False):
    df_file = data_dir + node + '_'
    df_file += str(start_time).replace('-','').replace(
            ':','').replace(' ','_') + '_'
    df_file += str(end_time).replace('-','').replace(':','').replace(' ','_') 
    if remove_idle:
        df_file += '_preparedDF_noIdle'
    else:
        df_file += '_preparedDF_withIdle'

    if unscaled:
        df_file += '_unscaled'
    df_file += '.pickle'
    if os.path.isfile(df_file):
        return True, df_file
    else:
        return False, df_file

'''
Prepare data frame.
- drop unused columns
- scale values
- (if needed) remove idle periods from data frame
IN: data frame
IN: idle periods need to be removed or not (True or False)
OUT: the prepared data frame
OUT: list of timestamps 
OUT: list of idle periods
OUT: list telling if each timestamp is active (1) or not (0)
OUT: lists with timestamps of idle & non-idle periods
    - populated only if idle periods are removed, empty lists otherwise
OUT: scaler used to scale
'''
def prepare_dataframe(df, remove_idle, unscaled=False):
    # drop columns that have all the same values
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df = drop_stuff(df, cols_to_drop)

    continuous_features = df.columns 
    continuous_features = [c for c in continuous_features 
            if c not in ['time_stamp', 'node_id']]
    categorical_features = []

    timestamps = df['time_stamp']
    idle_periods = find_idle_periods(df, timestamps, False)
    active_idle = count_idle_periods(df, timestamps, False)

    if remove_idle:
        # add active/idle to data frame as new column
        df['active'] = active_idle
        df_noIdle = df[(df['active'] == 1)]
        df_idle = df[~(df['active'] == 1)]
        # remove idle periods from data frame
        df = df_noIdle
        ts_noIdle = df_noIdle['time_stamp']
        ts_idle = df_idle['time_stamp']
        if len(ts_noIdle) == 0:
            return (-1, [], -1, [], [], [], -1)
        if unscaled:
            df = preprocess_noScaling(df, categorical_features, 
                    continuous_features)
            scaler = None
        else:
            df, scaler = preprocess(df, categorical_features, 
                    continuous_features)
        columns_2_drop = ['time_stamp','active']
        df = drop_stuff(df, columns_2_drop)

        return (df, timestamps, idle_periods, active_idle, 
                ts_noIdle, ts_idle, scaler)
    else:
        if unscaled:
            df = preprocess_noScaling(df, categorical_features, 
                    continuous_features)
            scaler = None
        else:
            df, scaler = preprocess(df, categorical_features, 
                    continuous_features)
        columns_2_drop = ['time_stamp']
        df = drop_stuff(df, columns_2_drop)

        return (df, timestamps, idle_periods, active_idle, 
                [], [], scaler)

'''
Split data set randomly
- returns distinct test set with and w/o anomalies
'''
def split_dataset(df, ts_noIdle, freqGov):
    if freqGov == 'performance':
        freqGov_periods = _anomaly_periods
    elif freqGov == 'powersave':
        freqGov_periods = _anomaly_periods
    else:
        freqGov_periods = []

    anomalies  = []
    anomalies_1type  = []
    anomalies_2type  = []
    all_idxs = []

    test_train_or_else_idxs = []
    test_train_or_else_tss = []

    for idx in range(len(ts_noIdle)):
        all_idxs.append(idx)
    for idx in range(len(ts_noIdle)):
        ts = ts_noIdle[idx]
        for st, et in freqGov_periods[:-1]:
            if st <= ts < et:
                anomalies.append(idx)
                anomalies_1type.append(idx)
                break
        (st, et) = freqGov_periods[-1]
        if st <= ts < et:
            anomalies.append(idx)
            anomalies_2type.append(idx)

    no_anomalies = list(set(all_idxs) - set(anomalies))
    df_anomalies = df[anomalies]
    df_noAnomalies = df[no_anomalies]

    msk = np.random.rand(len(df_noAnomalies)) < 0.7
    train = df_noAnomalies[msk]
    test = df_noAnomalies[~msk]

    msk_idxs = {}
    for i in range(len(msk)):
        msk_idxs[no_anomalies[i]] = msk[i]

    for idx in range(len(ts_noIdle)):
        if idx in no_anomalies:
            if msk_idxs[idx] == True:
                test_train_or_else_idxs.append(0)
            else:
                test_train_or_else_idxs.append(1)
        elif idx in anomalies:
            if idx in anomalies_1type:
                test_train_or_else_idxs.append(2)
            else:
                test_train_or_else_idxs.append(3)
        else:
            test_train_or_else_idxs.append(-1)
    test_noAnomalies = test
    test = np.concatenate((test, df_anomalies), axis=0)

    return (train, test, test_train_or_else_idxs, df_noAnomalies, 
            df_anomalies, test_noAnomalies, df_anomalies)

'''
Create labels
'''
def get_labels(df, ts_noIdle, freqGov):
    if freqGov == 'performance':
        freqGov_periods = _anomaly_periods
    elif freqGov == 'powersave':
        freqGov_periods = _anomaly_periods
    else:
        freqGov_periods = []

    labels_binaryClass = [0] * len(ts_noIdle)
    labels_multiClass = [0] * len(ts_noIdle)

    n_anomalies = 0
    n_anomalies_1 = 0
    n_anomalies_2 = 0
    for idx in range(len(ts_noIdle)):
        ts = ts_noIdle[idx]
        for st, et in freqGov_periods[:-1]:
            if st <= ts < et:
                labels_binaryClass[idx] = 1
                labels_multiClass[idx] = 1
                n_anomalies += 1
                n_anomalies_1 += 1
                break
        (st, et) = freqGov_periods[-1]
        if st <= ts < et:
            labels_binaryClass[idx] = 1
            labels_multiClass[idx] = 2
            n_anomalies += 1
            n_anomalies_2 += 1

    return labels_binaryClass, labels_multiClass

'''
Analyse reconstruction errors distributions
    - extend previous function in order to classify data points in
    anomalous or normal classes
    - compute some statistics
    - explore varying thresholds to classify data point
'''
def error_distribution_2_class_varyThreshold(actual_normal, pred_normal, 
        actual_anomal, pred_anomal, node, actual_normal_all, pred_normal_all,
        debug=True):
    msk = np.random.rand(len(actual_anomal)) < 0.7
    validation_set_actual_A = actual_anomal[msk]
    test_set_actual_A = actual_anomal[~msk]
    validation_set_pred_A = actual_anomal[msk]
    test_set_pred_A = actual_anomal[~msk]

    actual_anomal_redux = actual_anomal[~msk]
    actual_anomal = actual_anomal_redux

    nn_samples, nn_series = actual_normal.shape
    errors_normal = [0] * nn_samples
    abs_errors_normal = {}
    squared_errors_normal = {}

    for j in range(nn_series):
        abs_errors_normal[j] = []
        squared_errors_normal[j] = []
    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors_normal[j].append(
                    abs(pred_normal[i][j] - actual_normal[i][j]))
            squared_errors_normal[j].append((
                pred_normal[i][j] - actual_normal[i][j])*
                (pred_normal[i][j] - actual_normal[i][j]))

    na_samples, na_series = actual_anomal.shape
    errors_anomal = [0] * na_samples
    abs_errors_anomal = {}
    squared_errors_anomal = {}

    for j in range(na_series):
        abs_errors_anomal[j] = []
        squared_errors_anomal[j] = []
    for i in range(na_samples):
        for j in range(na_series):
            abs_errors_anomal[j].append(
                    abs(pred_anomal[i][j] - actual_anomal[i][j]))
            squared_errors_anomal[j].append((
                pred_anomal[i][j] - actual_anomal[i][j])*
                (pred_anomal[i][j] - actual_anomal[i][j]))

    nn_all_samples, nn_all_series = actual_normal_all.shape
    errors_normal_all = [0] * nn_all_samples
    abs_errors_normal_all = {}
    squared_errors_normal_all = {}

    for j in range(nn_all_series):
        abs_errors_normal_all[j] = []
        squared_errors_normal_all[j] = []
    for i in range(nn_all_samples):
        for j in range(nn_all_series):
            abs_errors_normal_all[j].append(
                    abs(pred_normal_all[i][j] - actual_normal_all[i][j]))
            squared_errors_normal_all[j].append((
                pred_normal_all[i][j] - actual_normal_all[i][j])*
                (pred_normal_all[i][j] - actual_normal_all[i][j]))

    # max abs error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]
    for j in abs_errors_normal_all.keys():
        for i in range(len(abs_errors_normal_all[j])):
            if errors_normal_all[i] < abs_errors_normal_all[j][i]:
                errors_normal_all[i] = abs_errors_normal_all[j][i]
    for j in abs_errors_anomal.keys():
        for i in range(len(abs_errors_anomal[j])):
            if errors_anomal[i] < abs_errors_anomal[j][i]:
                errors_anomal[i] = abs_errors_anomal[j][i]

    n_perc_min = 70
    n_perc_max = 99
    classes_normal = [0] * nn_samples
    classes_anomal = [1] * na_samples
    errors = errors_normal + errors_anomal
    classes = classes_normal + classes_anomal

    best_threshold = n_perc_max
    fscore_A_best = 0
    fscore_N_best = 0
    fscore_W_best = 0
    fps = []
    fns = []
    tps = []
    tns = []
    n_percs = []
    precs = []
    recalls = []
    fscores = []
    for n_perc in range(n_perc_min, n_perc_max+2):
        error_threshold = np.percentile(np.asarray(errors_normal_all), n_perc)
        if debug:
            print("Try with percentile: %s (threshold: %s)" % (
                n_perc, error_threshold))

        predictions = []
        for e in errors:
            if e > error_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=0)
        precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=1)
        precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
                classes, predictions, average='weighted')

        tn, fp, fn, tp = confusion_matrix(classes, predictions).ravel()
        fscores.append(fscore_W)
        precs.append(precision_W)
        recalls.append(recall_W)
        n_percs.append(n_perc)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        tns.append(tn)
 
        if fscore_W > fscore_W_best:
            precision_W_best = precision_W
            precision_N_best = precision_N
            precision_A_best = precision_A
            recall_W_best = recall_W
            recall_N_best = recall_N
            recall_A_best = recall_A
            fscore_W_best = fscore_W
            fscore_N_best = fscore_N
            fscore_A_best = fscore_A
            best_threshold = n_perc

    if debug:
        print("Node {} - Best result obtained with threshold {}".format(node, 
            best_threshold))
        print("Normal: precision %f, recall %f, F-score %f" % (precision_N_best, 
                recall_N_best, fscore_N_best))
        print("Anomaly: precision %f, recall %f, F-score %f" %(precision_A_best, 
                recall_A_best, fscore_A_best))
        print("All Classes Weighted: precision %f, recall %f, F-score %f" % (
                precision_W_best, recall_W_best, fscore_W_best))

        mrkrsize = 5
        fig = plt.figure()
        plt.plot(n_percs, fscores, c='b', label='F-Score', marker='o', 
                linewidth=2, markersize=mrkrsize)
        plt.plot(n_percs, recalls, c='g', label='Recall', marker='x', 
                linewidth=2, markersize=mrkrsize)
        plt.plot(n_percs, precs, c='r', label='Precision', marker='D',
                linewidth=2, markersize=mrkrsize)
        plt.axvline(x=best_threshold, c='grey',linestyle='--')
        plt.xlabel("N-th percentile")
        plt.ylabel("Detection Accuracy")
        plt.legend()
        plt.show()

    return (best_threshold, precision_N_best, recall_N_best, fscore_N_best,
            precision_A_best, recall_A_best, fscore_A_best,
            precision_W_best, recall_W_best, fscore_W_best)

'''
Returns the error threshold computed with thanks to the errors distributions
'''
def compute_error_threshold(n_perc, actual_normal, pred_normal):
    nn_samples, nn_series = actual_normal.shape
    errors_normal = [0] * nn_samples
    abs_errors_normal = {}

    for j in range(nn_series):
        abs_errors_normal[j] = []
    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors_normal[j].append(
                    abs(pred_normal[i][j] - actual_normal[i][j]))
    # max abs error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]

    error_threshold = np.percentile(np.asarray(errors_normal), n_perc)
    return error_threshold

'''
Plot error distributions of one node
'''
def plot_error_distribution_singleNode(node, actual_N, pred_N, 
        actual_A, pred_A, actual_N_all, pred_N_all):

    fig = plt.figure()
    color_idx = 0
    actual_normal = actual_N[node]
    pred_normal = pred_N[node]
    actual_anomal = actual_A[node]
    pred_anomal = pred_A[node]
    actual_normal_all = actual_N_all[node]
    pred_normal_all = pred_N_all[node]

    msk = np.random.rand(len(actual_anomal)) < 0.7
    validation_set_actual_A = actual_anomal[msk]
    test_set_actual_A = actual_anomal[~msk]
    validation_set_pred_A = actual_anomal[msk]
    test_set_pred_A = actual_anomal[~msk]

    actual_anomal_redux = actual_anomal[~msk]
    actual_anomal = actual_anomal_redux

    nn_samples, nn_series = actual_normal.shape
    errors_normal = [0] * nn_samples
    abs_errors_normal = {}
    squared_errors_normal = {}

    for j in range(nn_series):
        abs_errors_normal[j] = []
        squared_errors_normal[j] = []

    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors_normal[j].append(
                    abs(pred_normal[i][j] - actual_normal[i][j]))
            squared_errors_normal[j].append((
                pred_normal[i][j] - actual_normal[i][j])*
                (pred_normal[i][j] - actual_normal[i][j]))

    na_samples, na_series = actual_anomal.shape
    errors_anomal = [0] * na_samples
    abs_errors_anomal = {}
    squared_errors_anomal = {}

    for j in range(na_series):
        abs_errors_anomal[j] = []
        squared_errors_anomal[j] = []
    for i in range(na_samples):
        for j in range(na_series):
            abs_errors_anomal[j].append(
                    abs(pred_anomal[i][j] - actual_anomal[i][j]))
            squared_errors_anomal[j].append((
                pred_anomal[i][j] - actual_anomal[i][j])*
                (pred_anomal[i][j] - actual_anomal[i][j]))

    nn_all_samples, nn_all_series = actual_normal_all.shape
    errors_normal_all = [0] * nn_all_samples
    abs_errors_normal_all = {}
    squared_errors_normal_all = {}

    for j in range(nn_all_series):
        abs_errors_normal_all[j] = []
        squared_errors_normal_all[j] = []
    for i in range(nn_all_samples):
        for j in range(nn_all_series):
            abs_errors_normal_all[j].append(
                    abs(pred_normal_all[i][j] - actual_normal_all[i][j]))
            squared_errors_normal_all[j].append((
                pred_normal_all[i][j] - actual_normal_all[i][j])*
                (pred_normal_all[i][j] - actual_normal_all[i][j]))

    # max abs error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]
    for j in abs_errors_normal_all.keys():
        for i in range(len(abs_errors_normal_all[j])):
            if errors_normal_all[i] < abs_errors_normal_all[j][i]:
                errors_normal_all[i] = abs_errors_normal_all[j][i]
    for j in abs_errors_anomal.keys():
        for i in range(len(abs_errors_anomal[j])):
            if errors_anomal[i] < abs_errors_anomal[j][i]:
                errors_anomal[i] = abs_errors_anomal[j][i]

    plt.hist(errors_normal, bins=80, color='b', alpha=.8,
            label='Node {} - Normal'.format(node))

    plt.hist(errors_anomal, bins=50, color='r', alpha=.8,
            label='Node {} - Anomaly'.format(node))
    color_idx += 1

    plt.ylabel('# data points')
    plt.xlabel('Error')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(prop={'size': 10})
    plt.show()

