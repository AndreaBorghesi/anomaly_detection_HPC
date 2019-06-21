'''
Misc util function used by several other scripts

Author: Andrea Borghesi, andrea.borghesi3@unibo.it,
    University of Bologna
Date: 20170928
'''

import os
import sys
import math
from decimal import *
from collections import deque
import collections
import operator
import json
import pickle
import datetime
import argparse
import configparser
from cassandra.cluster import Cluster
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from cassandra.query import dict_factory, SimpleStatement
from cassandra.auth import PlainTextAuthProvider
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.cluster import v_measure_score, homogeneity_score
from sklearn.metrics.cluster import completeness_score, adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn import mixture
import pandas as pd
#from sklearn_pandas import DataFrameMapper
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import time
import re
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np
import ast
import signal, psutil
import scipy.stats as stats
from keras.models import load_model
import itertools as it
from sklearn import metrics
from sklearn.manifold import SpectralEmbedding, Isomap, MDS, TSNE
from sklearn.manifold import LocallyLinearEmbedding

_ncores = 16

_threshold_feat = 0.15

'''
Horrible temporary solution.
In some nodes of DAVIDE I've changed the frequency governor a few times, to 
create anomalies.
In this lists I stored the phases
- this should _not_ be embedded in the code

Nodes that can be always trusted for POWERSAVE:
    - davide[4-5, 16-19] 
Nodes that can be trusted for POWERSAVE after ~2018-04-13 11:46
    - davide[42-45, 16-19] 

Nodes that can be always trusted for PERFORMANCE
    - davide[24-31]

Nodes that can be always trusted for ONDEMAND
    - davide[10-13]

The dates are converted to UTC

Temporary solutions: for those nodes that had different anomalies (_freqGov == 
power_perf or _freqGov == perf_power) the last period represent the period with
the different anomaly  --- yes quite terrible solution and code
'''
_powersave_periods = [
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

_performance_periods = _powersave_periods

_ondemand_periods = [
        (datetime.datetime.strptime("2018-04-12 11:50:00","%Y-%m-%d %H:%M:%S"),
            datetime.datetime.strptime("2018-04-13 10:41:00","%Y-%m-%d %H:%M:%S")
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
            )
        ]

class Box:
    pass

__m = Box()  

__m.duration_threshold = 120
__m.local_home_dir = '/home/b0rgh/'
__m.my_home_dir = '/media/b0rgh/Elements/'
__m.raw_data_dir = __m.my_home_dir + 'rawData_logs_eurora/'
__m.offline_jobs_dir = __m.raw_data_dir + 'davide/aggregate_job_info/'
__m.offline_phys_dir = __m.raw_data_dir + 'davide/physical_measures/'
__m.plot_dir = __m.raw_data_dir + 'davide/plots/predictions/'

__m.offline_jobs_file_data = __m.offline_jobs_dir + 'offline_jobs_data.pickle'
__m.offline_jobs_file_info = __m.offline_jobs_dir + 'offline_jobs_info.pickle'
__m.offline_phys_file = __m.offline_phys_dir + 'phys_measures_'

__m.node_whitelist = __m.local_home_dir + 'data_collection_analysis/data_analysis/davide'
__m.node_whitelist += '/nodes_whitelist'

__m.color_maps = {2 : {0: 'r', 1: 'b'}, 
        3: {0: 'r', 1: 'b', 2: 'gold'},
        4: {0: 'r', 1: 'b', 2: 'gold', 3: 'k'},
        5: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'k'},
        6: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'g', 5: 'k'},
        7: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'g', 5: 'c', 6: 'k'},
        8: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'g', 5: 'c', 
            6: 'orange', 7: 'k'},
        9: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'sienna', 5: 'c', 
            6: 'orange', 7: 'm', 8: 'k'},
        10: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'sienna', 5: 'c', 
            6: 'orange', 7: 'm', 8: 'grey', 9: 'k'},
        11: {0: 'r', 1: 'b', 2: 'gold', 3: 'g', 4: 'sienna', 5: 'c', 
            6: 'orange', 7: 'm', 8: 'grey', 9: 'pink', 10: 'k'}

        }

__m.epoch = datetime.datetime.utcfromtimestamp(0)

JOBS_NSAMPLES = 100

def unix_time_millis(dt):
    return long((dt - epoch).total_seconds() * 1000.0)
def millis_unix_time(millis):
    seconds = millis / 1000
    return epoch + datetime.timedelta(seconds=seconds)

def read_node_whitelist():
    with open(node_whitelist, 'r') as nf:
        nodes = nf.readlines()
    nodes = [n.strip() for n in nodes]
    return nodes

'''
Some jobs stored in Cassandra have wrong dates, i.e. start times later
than end times...
This function tries to fix these errors (when possible)
'''
def fix_jobs_times(offline_jobs_info):
    strange_times = 0
    end_time_fake = 0
    jobs_info_mod = {}
    for job_id, vals in offline_jobs_info.iteritems():
        fake_end = datetime.datetime.strptime("2000-06-21 09:00:00",
                "%Y-%m-%d %H:%M:%S")
        if vals['end_time'] <= fake_end:
            end_time_fake += 1
        # in this case (usually, hopefully always) the start time has been
        # stored "one day in advance", so we just need to subtract 24 hours 
        # and we're OK
        elif vals['start_time'] > vals['end_time']:
            strange_times += 1
            vals['start_time'] -= datetime.timedelta(hours=24)
        jobs_info_mod[job_id] = vals
    return jobs_info_mod

'''
Some jobs are 'strange', i.e. they have very low duration
This function remove strange jobs from the data set
'''
def filter_jobs(jobs_info, jobs_data):
    job_ids_to_remove = []
    for jid in jobs_info.keys():
        # consider only jobs with a significant duration
        duration = (jobs_info[jid]['end_time'] -
                jobs_info[jid]['start_time']).seconds
        if duration < duration_threshold:
            job_ids_to_remove.append(jid)
    # not really filtering: if we've had some problem in the stable values
    # we use the non-stable ones (in order not decrease data size too much)
    for jid in jobs_data.keys():
        if 'cpu_tot_avg_power_stable' not in jobs_data[jid]:
            jobs_data[jid]['cpu_tot_avg_power_stable'] = jobs_data[jid][
                    "job_tot_avg_power"]
        if 'ipmi_mean_pow_stable' not in jobs_data[jid]:
            jobs_data[jid]['ipmi_mean_pow_stable'] = jobs_data[jid][
                    "ipmi_job_avg_power"]

        if math.isnan(jobs_data[jid]['cpu_tot_avg_power_stable']):
            jobs_data[jid]['cpu_tot_avg_power_stable'] = jobs_data[jid][
                    "job_tot_avg_power"]
        if math.isnan(jobs_data[jid]['ipmi_mean_pow_stable']):
            jobs_data[jid]['ipmi_mean_pow_stable'] = jobs_data[jid][
                    "ipmi_job_avg_power"]
    for jid in job_ids_to_remove:
        if jid in jobs_info:
            del jobs_info[jid]
        if jid in jobs_data:
            del jobs_data[jid]
    return jobs_info, jobs_data

'''
Compute the time sampling events during a job life time, excluding
transitory phases (begin and end)
'''
def compute_times_stable(start_time, end_time, powers):
    times = []
    powers_stable = []
    time = start_time
    tstart = unix_time_millis(start_time)
    tstop = unix_time_millis(end_time)
    sampling_time = (tstop - tstart) / (1000 * JOBS_NSAMPLES)
    new_start_time = -1
    new_end_time = start_time
    transitory_duration_asTime = datetime.timedelta(seconds=transitory_duration)
    if sampling_time == 0:
        sampling_time = 1
    for p in powers:
        if(start_time + transitory_duration_asTime <= time 
                <= end_time - transitory_duration_asTime):
            times.append(time)
            powers_stable.append(float(p))
            if new_start_time == -1:
                new_start_time = time
            if new_start_time != -1 and new_end_time < time:
                new_end_time = time
        time += datetime.timedelta(seconds=sampling_time) 
    return times, powers_stable, new_start_time, new_end_time

'''
Compute mean metric value excluding the transitory initial/final phase
'''
def compute_mean_stable(job_info, job_data, metric, start_time, end_time):
    used_nodes = job_info['vnode_list'].split(',')

    vals_all_nodes = job_data[metric].split('#')
    if vals_all_nodes[len(vals_all_nodes)-1] == ' ':
        vals_all_nodes = vals_all_nodes[:-1]

    if len(vals_all_nodes) != len(used_nodes):
        return -1, -1, -1, -1, -1 

    mean_stable_sum = 0
    mean_stable = {}
    cpu_powers_allNodes = []
    new_start_time = -1
    new_end_time = -1
    job_node_powers = ""
    for i in range(len(used_nodes)):
        vals = vals_all_nodes[i].split(',')
        times, vals_stable, nst, net = compute_times_stable(
                start_time, end_time, vals)
        if nst != -1 and new_start_time != -1:
            new_start_time = nst
        if net != -1 and new_end_time != -1:
            new_end_time = net
        mean_val = Decimal(np.nanmean(np.asarray(vals_stable)))
        cpu_powers_allNodes_str = ""
        for v in vals_stable:
            cpu_powers_allNodes_str += str(v) + ','
        cpu_powers_allNodes_str = cpu_powers_allNodes_str[:-1]
        cpu_powers_allNodes.append(cpu_powers_allNodes_str)
        mean_stable[used_nodes[i]] = mean_val
        mean_stable_sum += mean_val
        job_node_powers += str(mean_val) + ','
    job_node_powers = job_node_powers[:-1]

    return mean_stable_sum, job_node_powers, millis_unix_time(new_start_time), \
            millis_unix_time(new_end_time), cpu_powers_allNodes

'''
Compute mean stable power consumption (excluding transitory phases) for
each job in the data set and return the updated data set. Also need to
update pickle file
'''
def add_stable_means_to_data_set(jobs_info, jobs_data):
    data_set_mod = {}
    for job_id, vals in jobs_data.iteritems():
        data_set_mod[job_id] = vals
        if job_id not in jobs_info:
            continue
        start_time = jobs_info[job_id]['start_time']
        end_time = jobs_info[job_id]['end_time']
        #used_nodes = jobs_info[job_id]['used_nodes'].split(',')
        used_nodes = jobs_info[job_id]['vnode_list'].split(',')
        used_cores = jobs_info[job_id]['used_cores']

        (mean_cpu_pow_stable, cpu_job_node_powers_stable, start_time_stable, 
                end_time_stable, cpu_powers_allNode_stable
                ) = compute_mean_stable(
                jobs_info[job_id], jobs_data[job_id], 'job_node_avg_powerlist', 
                start_time, end_time)
        (ipmi_mean_pow_stable, ipmi_job_node_powers_stable, start_time_stable, 
                end_time_stable, ipmi_powers_allNodes_stable
                ) = compute_mean_stable(
                jobs_info[job_id], jobs_data[job_id], 'ipmi_job_powers', 
                start_time, end_time)

        if math.isnan(mean_cpu_pow_stable):
            mean_cpu_pow_stable = jobs_data[job_id]['job_tot_avg_power']
        if math.isnan(ipmi_mean_pow_stable):
            ipmi_mean_pow_stable = jobs_data[job_id]['ipmi_job_avg_power']

        data_set_mod[job_id]['cpu_tot_avg_power_stable'] = mean_cpu_pow_stable
        data_set_mod[job_id]['cpu_job_node_powers_stable'
                ] = cpu_job_node_powers_stable
        data_set_mod[job_id]['ipmi_mean_pow_stable'] = ipmi_mean_pow_stable
        data_set_mod[job_id]['ipmi_job_node_powers_stable'
                ] = ipmi_job_node_powers_stable

    # update pickle file (for future usages)
    with open(offline_jobs_file_data, 'wb') as handle:
        pickle.dump(data_set_mod, handle, 
                protocol=pickle.HIGHEST_PROTOCOL)
    return data_set_mod

'''
Dump DL model results in pickle file
'''
def dump_stats_results(stats_res, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(stats_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Load DL model results stored as pickle file
'''
def  load_stats_results(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None

'''
Load information about jobs (stuff from galileo_jobs* tables) and 
aggregated data (mean power, etc. - jobs_measure_aggregate table)

For quicker testing I cache the data in cassandra into a pickle file
offline_jobs_data contains aggregated information, i.e. job id and 
mean power
offline_jobs_info contains information about job request and life
time, i.e. start/end time
'''
def load_jobs(offline_file, info_data, CASSANDRA_IP, CASSANDRA_PORT,
        CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_JOB_KEYSPACE, table):
    if os.path.isfile(offline_file):
        # the offline data set has already been loaded
        with open(offline_file, 'rb') as handle:
            return pickle.load(handle)
    else:
        ''' 
        Connect to Cassandra 
        '''
        auth_provider = PlainTextAuthProvider(
                username=CASSANDRA_USR, password=CASSANDRA_PWD)
        cluster = Cluster(contact_points=(CASSANDRA_IP,),
            auth_provider=auth_provider)

        session = cluster.connect(CASSANDRA_JOB_KEYSPACE)
        session.row_factory = dict_factory

        #Load offline data set
        query = "select * from %s;" % (table)
        result = session.execute(query, timeout=None)
        offline_data_info = {}
        for r in result:
            offline_data_info[r['job_id']] = r
        with open(offline_file, 'wb') as handle:
            pickle.dump(offline_data_info, handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

        cluster.shutdown()
        return offline_data_info

'''
Same purpose of load_jobs, to be used if the internet connection is
slow and too much data has to be retrieved
'''
def load_jobs_lowNet(offline_file, info_data, CASSANDRA_IP, CASSANDRA_PORT,
        CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_JOB_KEYSPACE, table):
    query = "select * from %s;" % (table)

    if os.path.isfile(offline_file):
        # the offline data set has already been loaded
        with open(offline_file, 'rb') as handle:
            return pickle.load(handle)
    else:
        ''' 
        Connect to Cassandra 
        '''
        auth_provider = PlainTextAuthProvider(
                username=CASSANDRA_USR, password=CASSANDRA_PWD)
        cluster = Cluster(contact_points=(CASSANDRA_IP,),
            auth_provider=auth_provider)
        session = cluster.connect(CASSANDRA_JOB_KEYSPACE)
        session.row_factory = dict_factory

        #Load offline data set
        stmt = SimpleStatement(query, fetch_size=1000)
        result = session.execute(stmt, timeout=None)

        offline_data_info = {}
        for r in result:
            offline_data_info[r['job_id']] = r

        with open(offline_file, 'wb') as handle:
            pickle.dump(offline_data_info, handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

        cluster.shutdown()
        return offline_data_info

'''
Load information coming from physical sensors
'''
def load_phys(node, test_start_time, test_end_time, offline_file, CASSANDRA_IP, 
        CASSANDRA_PORT, CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_JOB_KEYSPACE, 
        table):
    if os.path.isfile(offline_file):
        # the offline data set has already been loaded
        with open(offline_file, 'rb') as handle:
            return pickle.load(handle)
    else:
        ''' 
        Connect to Cassandra 
        '''
        auth_provider = PlainTextAuthProvider(
                username=CASSANDRA_USR, password=CASSANDRA_PWD)
        cluster = Cluster(contact_points=(CASSANDRA_IP,),
            auth_provider=auth_provider, connect_timeout=10)
        session = cluster.connect(CASSANDRA_JOB_KEYSPACE)
        session.row_factory = dict_factory

        #Loar# offline data set
        query = "select * from %s where node_id = '%s' and time_stamp <= '%s' \
                and time_stamp >= '%s';" % (table, node, test_end_time,
                        test_start_time)
        #result = session.execute(query, timeout=None)

        stmt = SimpleStatement(query, fetch_size=1000)
        result = session.execute(stmt, timeout=None)

        offline_phys = {}
        for r in result:
            offline_phys[r['time_stamp']] = r

        with open(offline_file, 'wb') as handle:
            pickle.dump(offline_phys, handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)

        cluster.shutdown()
        return offline_phys


'''
Compute prediction accuracy
'''
def eval_predictor(model, scaler, scaler_target, test_data, test_target, plot):
    predictions = {}
    for job_id, features in test_data.iterrows():
        if scaler != -1:
            scaled_features = scaler.transform(features)
            pred_pow = model.predict(scaled_features)
        else:
            # new sklearn version
            pred_pow = model.predict([features])
            # old sklearn versione
            #pred_pow = model.predict(features)
        predictions[job_id] = pred_pow[0]

    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    test_powers = []
    pred_list = []

    for job_id, pred in predictions.iteritems():
        if scaler != -1:
            test_powers.append(scaler_target.transform(test_target[job_id]))
        else:
            test_powers.append(test_target[job_id])
        pred_list.append(pred)

        abs_errors.append(abs(Decimal(pred) - test_target[job_id]))
        squared_errors.append((Decimal(pred) - test_target[job_id])*
            (Decimal(pred) - test_target[job_id]))
        if test_target[job_id] != 0:
            p_abs_errors.append((abs(Decimal(pred) - test_target[job_id]))
                * 100 / test_target[job_id])
        sp_abs_errors.append((abs(Decimal(pred) - test_target[job_id])) * 100 / 
            (test_target[job_id] + Decimal(pred)))

    mean_test_power = np.mean(np.asarray(test_powers))

    MAE = Decimal(np.mean(np.asarray(abs_errors)))
    MAPE = Decimal(np.mean(np.asarray(p_abs_errors)))
    SMAPE = Decimal(np.mean(np.asarray(sp_abs_errors)))
    MSE = Decimal(np.mean(np.asarray(squared_errors)))
    RMSE = Decimal(math.sqrt(MSE))
    if len(test_powers) > 1 and (max(test_powers) != min(test_powers)):
        NRMSE = RMSE / (max(test_powers) - min(test_powers))
    else:
        NRMSE = RMSE / mean_test_power
    CVRMSE = RMSE / mean_test_power

    R2 = r2_score(test_powers, pred_list)
    SK_MAE = mean_absolute_error(test_powers, pred_list)
    MedAE = median_absolute_error(test_powers, pred_list)
    SK_MSE = mean_squared_error(test_powers, pred_list)
    EV = explained_variance_score(test_powers, pred_list)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["NRMSE"] = NRMSE
    stats_res["CVRMSE"] = CVRMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV

    if plot:
        plot_errors(abs_errors, p_abs_errors, sp_abs_errors, 
                squared_errors, test_powers, pred_list)

    return stats_res

'''
Compute prediction accuracy
'''
def eval_predictor_2(test_predictions, plot, target_label):
    predictions = {}
    test_target = {}
    for p in test_predictions:
        # some predictions might be negative....
        predictions[p['job_id']] = abs(p['prediction'])
        test_target[p['job_id']] = p[target_label]

    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    test_powers = []
    pred_list = []
    for job_id, pred in predictions.iteritems():
        pred_list.append(pred)
        test_powers.append(test_target[job_id])

        abs_errors.append(abs(Decimal(pred) - test_target[job_id]))
        squared_errors.append((Decimal(pred) - test_target[job_id])*
            (Decimal(pred) - test_target[job_id]))
        if test_target[job_id] != 0:
            p_abs_errors.append((abs(Decimal(pred) - test_target[job_id]))
                * 100 / test_target[job_id])
        sp_abs_errors.append((abs(Decimal(pred) - test_target[job_id])) * 100 / 
            (test_target[job_id] + Decimal(pred)))

    mean_test_power = Decimal(np.mean(np.asarray(test_powers)))

    MAE = Decimal(np.mean(np.asarray(abs_errors)))
    MAPE = Decimal(np.mean(np.asarray(p_abs_errors)))
    SMAPE = Decimal(np.mean(np.asarray(sp_abs_errors)))
    MSE = Decimal(np.mean(np.asarray(squared_errors)))
    RMSE = Decimal(math.sqrt(MSE))
    if len(test_powers) > 1 and (max(test_powers) != min(test_powers)):
        NRMSE = RMSE / (max(test_powers) - min(test_powers))
    else:
        NRMSE = RMSE / mean_test_power
    CVRMSE = RMSE / mean_test_power

    R2 = r2_score(test_powers, pred_list)
    SK_MAE = mean_absolute_error(test_powers, pred_list)
    MedAE = median_absolute_error(test_powers, pred_list)
    SK_MSE = mean_squared_error(test_powers, pred_list)
    EV = explained_variance_score(test_powers, pred_list)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["NRMSE"] = NRMSE
    stats_res["CVRMSE"] = CVRMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV

    if plot:
        plot_errors(abs_errors, p_abs_errors, sp_abs_errors, 
                squared_errors, test_powers, pred_list)
    return stats_res

'''
Create train set from time-series
See: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
Author: Dr. Jason Brownlee

    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

'''
Compute prediction accuracy
'''
def eval_predictor_3(X, Y):
    if len(X) != len(Y):
        return None

    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    test_powers = []
    pred_list = []
    i = 0
    for x in X:
        y = Y[i]
        abs_errors.append(abs(y - x))
        squared_errors.append(abs(y - x)*abs(y - x))
        if x != 0:
            p_abs_errors.append(abs(y -x) * 100 / x)
        sp_abs_errors.append(abs(y - x) * 100 / (x + y))
        i += 1

    #mean_test_power = np.mean(np.asarray(test_powers))
    mean_test_power = np.mean(np.asarray(X))

    MAE = np.mean(np.asarray(abs_errors)) * 100
    MAPE = np.mean(np.asarray(p_abs_errors)) 
    SMAPE = np.mean(np.asarray(sp_abs_errors)) 
    MSE = np.mean(np.asarray(squared_errors)) * 100
    RMSE = math.sqrt(MSE)

    #if len(test_powers) > 1 and (max(test_powers) != min(test_powers)):
    #    NRMSE = RMSE / (max(test_powers) - min(test_powers))
    #else:
    #    NRMSE = RMSE / mean_test_power
    if len(X) > 1 and (max(X) != min(X)):
        NRMSE = RMSE / (max(X) - min(X))
    else:
        NRMSE = RMSE / mean_test_power
    CVRMSE = RMSE / mean_test_power

    #R2 = r2_score(test_powers, pred_list)
    #SK_MAE = mean_absolute_error(test_powers, pred_list)
    #MedAE = median_absolute_error(test_powers, pred_list)
    #SK_MSE = mean_squared_error(test_powers, pred_list)
    #EV = explained_variance_score(test_powers, pred_list)
    R2 = r2_score(X, Y)
    SK_MAE = mean_absolute_error(X, Y)
    MedAE = median_absolute_error(X, Y)
    SK_MSE = mean_squared_error(X, Y)
    EV = explained_variance_score(X, Y)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["NRMSE"] = NRMSE
    stats_res["CVRMSE"] = CVRMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV

    return stats_res

'''
Drop unused features and rows with NaN
'''
def drop_stuff(df, features_to_be_dropped):
    for fd in features_to_be_dropped:
        if fd in df:
            del df[fd]
    #new_df = df.dropna(axis=0, how='any')
    #new_df = new_df.dropna()
    new_df = df.dropna(axis=0, how='all')
    new_df = new_df.dropna(axis=1, how='all')
    #new_df = df.dropna()
    new_df = new_df.fillna(0)
    #new_df = new_df.dropna(0)
    #new_df = df
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
    #scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    for c in categorical_features:
        df = encode_category(df, c)

    return df, scaler

'''
Plot some information about prediction errors
'''
def plot_errors(abs_errors, p_abs_errors, sp_abs_errors, 
        squared_errors, test_powers, pred_list):
    fig = plt.figure()
    plt.hist(np.asarray(sp_abs_errors, dtype='float'), bins=500)
    plt.ylabel("# Jobs")
    plt.xlabel("sp_abs_errors")
    #filename = plot_dir + "hist_error_job_powers/"
    #filename += "hist_jobs_power_IPMI_" + metric + ".png"
    #plt.savefig(filename)
    plt.show()

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

    #cumul_errors = []
    #for i in range(len(stats_res["ABS_ERRORS"][0])):
    #    cumul_errors.append(0)

    #for j in stats_res["ABS_ERRORS"].keys():
    #    for i in range(len(stats_res["ABS_ERRORS"][j])):
    #        cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

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

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

'''
Evaluate prediction
- directly compute rolling averages
'''
def evaluate_predictions_rolling_avg(predicted, actual, gaps, N):
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

    abs_errors_rollingMeans = {}
    p_abs_errors_rollingMeans = {}
    sp_abs_errors_rollingMeans = {}
    squared_errors_rollingMeans = {}
    for j in range(nb_series):
        abs_errors_rollingMeans[j] = running_mean(np.asarray(
            abs_errors[j]),N).tolist()
        p_abs_errors_rollingMeans[j] = running_mean(np.asarray(
            p_abs_errors[j]),N).tolist()
        sp_abs_errors_rollingMeans[j] = running_mean(np.asarray(
            sp_abs_errors[j]),N).tolist()
        squared_errors_rollingMeans[j] = running_mean(np.asarray(
            squared_errors[j]),N).tolist()

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors_rollingMeans
    stats_res["P_ABS_ERRORS"] = p_abs_errors_rollingMeans
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors_rollingMeans
    stats_res["SQUARED_ERRORS"] = squared_errors_rollingMeans

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
Evaluate prediction with multiple model.
For each point in the test set select the model that has the smaller error
'''
def evaluate_predictions_multi(predicted, actual, gaps):
    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    nb_samples, nb_series= actual.shape

    abs_errors = {}
    p_abs_errors = {}
    sp_abs_errors = {}
    squared_errors = {}
    predictions = {}
    best_model_perFeature = {}
    MAE = {}
    MAPE = {}
    SMAPE = {}
    MSE = {}
    RMSE = {}
    R2 = {}

    for j in range(nb_series):
        abs_errors[j] = []
        p_abs_errors[j] = []
        sp_abs_errors[j] = []
        squared_errors[j] = []
        predictions[j] = []
        best_model_perFeature[j] = []

    best_model_perSample = []

    for i in range(nb_samples):
        ae_model = {}
        for j in range(nb_series):
            min_ae = sys.maxint * 1.0 
            min_sqe = sys.maxint * 1.0
            min_pae = sys.maxint * 1.0 
            min_spae = sys.maxint * 1.0 
            pred = 0
            for model in predicted.keys():
                ae = abs(predicted[model][i][j] - actual[i][j])
                sqe = ((predicted[model][i][j] - actual[i][j]) *
                    (predicted[model][i][j] - actual[i][j]))
                if actual[i][j] != 0:
                    pae = ((abs(predicted[model][i][j] - actual[i][j]))* 
                            100 / actual[i][j])
                spae = ((abs(predicted[model][i][j] - actual[i][j])) * 100 / 
                    (predicted[model][i][j] + actual[i][j]))
                if ae < min_ae:
                    min_ae = ae
                    min_sqe = sqe
                    min_pae = pae
                    min_spae = spae
                    pred = predicted[model][i][j]
                    chosen_model = model

                if model not in ae_model:
                    ae_model[model] = ae
                else:
                    ae_model[model] += ae

            abs_errors[j].append(min_ae)
            squared_errors[j].append(min_sqe)
            p_abs_errors[j].append(min_pae)
            sp_abs_errors[j].append(min_spae)
            predictions[j].append(pred)
            best_model_perFeature[j].append(chosen_model)
        
        for m in predicted.keys():
            ae_model[m] /= nb_series
        min_ae_allFeatures = sys.maxint * 1.0
        for m in predicted.keys():
            if ae_model[m] <= min_ae_allFeatures:
                min_ae_allFeatures = ae_model[m]
                best_model_allFeatures = m
        best_model_perSample.append(best_model_allFeatures)

    actual = actual.transpose()

    for j in range(nb_series):
        MAE[j] = Decimal(np.mean(np.asarray(abs_errors[j])))
        MAPE[j] = Decimal(np.mean(np.asarray(p_abs_errors[j])))
        SMAPE[j] = Decimal(np.nanmean(np.asarray(sp_abs_errors[j])))
        MSE[j] = Decimal(np.mean(np.asarray(squared_errors[j])))
        RMSE[j] = Decimal(math.sqrt(MSE[j]))
        R2[j] = r2_score(actual[j], predictions[j])

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
    stats_res["BEST_MODEL"] = best_model_perSample

    cumul_errors = []
    for i in range(len(stats_res["ABS_ERRORS"][0])):
        cumul_errors.append(0)

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
    #plt.hist(np.asarray(sp_abs_errors, dtype='float'), bins=500)
    #plt.ylabel("# Jobs")
    #plt.xlabel("sp_abs_errors")
    #filename = plot_dir + "hist_error_job_powers/"
    #filename += "hist_jobs_power_IPMI_" + metric + ".png"
    #plt.savefig(filename)
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
        #plt.plot(timestamps[1:], cumul_errors_norm)
        if len(timestamps) > len(cumul_errors_norm):
            dif = len(timestamps) - len(cumul_errors_norm)
            timestamps = timestamps[dif:]
            #plt.plot(timestamps[dif:], cumul_errors_norm)
        else:
            dif = len(cumul_errors_norm) - len(timestamps)
            cumul_errors_norm = cumul_errors_norm[dif:]
            #plt.plot(timestamps, cumul_errors_norm[dif:])
    #else:
    #    plt.plot(timestamps, cumul_errors_norm)

    if "BEST_MODEL" not in stats_res:
        plt.plot(timestamps, cumul_errors_norm)

        if test_idx != -1:
            test_timestamp = ts_noIdle[test_idx]
            plt.axvline(x=test_timestamp, linewidth=2, 
                    linestyle='--', color='k')

        ax = fig.add_subplot(111)
        for st, et in gaps:
            #dif = (et - st).total_seconds()
            start = mdates.date2num(st)
            end = mdates.date2num(et)
            dif = end - start
            #rect = patches.Rectangle((start, 0), dif, 1, color='grey')
            rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='/', 
                    color='grey')
            ax.add_patch(rect)
        for st, et in idle_periods:
            #dif = (et - st).total_seconds()
            start = mdates.date2num(st)
            end = mdates.date2num(et)
            dif = end - start
            #rect = patches.Rectangle((start, 0), dif, 1, color='yellow')
            rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='//', 
                    color='yellow')
            ax.add_patch(rect)

        add_freq_govs_to_plot(freqGov, ax)

        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    else:
        ax = fig.add_subplot(111)
        colors = [int(i % len(set(stats_res["BEST_MODEL"]))) 
                for i in stats_res["BEST_MODEL"]]
        timestamps_mod = [pd.to_datetime(t) for t in timestamps]
        plt.scatter(timestamps_mod, cumul_errors_norm, c=colors)

    plt.title(title)
    plt.show()

'''
Plot errors generated with DL models (prediction errors..)
- plot results of different models
'''
def plot_errors_DL_multi(stats_res_multi, series_labels, 
        timestamps, sampling_time, gaps, idle_periods, node=""):
    
    cumul_errors_multi = {}
    cumul_errors_norm_multi = {}
    for model, stats_res in stats_res_multi.iteritems():
        cumul_errors = []
        for i in range(len(stats_res["ABS_ERRORS"][0])):
            cumul_errors.append(0)

        for j in stats_res["ABS_ERRORS"].keys():
            for i in range(len(stats_res["ABS_ERRORS"][j])):
                cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

        # 'normalize' cumulated errors
        cumul_errors_norm = []
        for ce in cumul_errors:
            cumul_errors_norm.append(ce / len(stats_res["ABS_ERRORS"]))

        cumul_errors_multi[model] = cumul_errors
        cumul_errors_norm_multi[model] = cumul_errors_norm

    #gaps = find_gaps_timeseries(timestamps, sampling_time)

    fig = plt.figure()
    for model in cumul_errors_norm_multi.keys():
        if len(timestamps) != len(cumul_errors_norm_multi[model]):
            dif = len(timestamps) - len(cumul_errors_norm_multi[model])
            plt.plot(timestamps[dif:], 
                    cumul_errors_norm_multi[model],label=model)
        else:
            plt.plot(timestamps, cumul_errors_norm_multi[model], label=model)
    ax = fig.add_subplot(111)
    for st, et in gaps:
        #dif = (et - st).total_seconds()
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        #rect = patches.Rectangle((start, 0), dif, 1, color='grey')
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='/', 
                color='grey')
        ax.add_patch(rect)
    for st, et in idle_periods:
        #dif = (et - st).total_seconds()
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, 0), dif, 1, color='yellow')
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='//', 
                color='yellow')
        ax.add_patch(rect)

    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    legend = ax.legend(loc='upper center')
    plt.title("Cumulative Errors %s" % node)
    plt.xlabel("Time")
    plt.ylabel("Normalized Error")
    plt.show()

'''
Plot the contributions of each feature to the cumulative error
'''
def plot_error_contributions_DL(stats_res, series_labels, timestamps, 
        sampling_time, gaps, idle_periods, samples_idleness=[], 
        ts_noIdle=[], ts_idle=[]):
    fig = plt.figure()

    if len(ts_noIdle) > 0:
        plot_error_contributions_DL_fixIdle(stats_res, series_labels, 
                timestamps, gaps, idle_periods, ts_noIdle, ts_idle)
        return

    cm = plt.cm.get_cmap('coolwarm')
    ax1 = fig.add_subplot(111)
    vmin = 0
    vmax = len(stats_res["ABS_ERRORS"]) + 1
    for j in stats_res["ABS_ERRORS"]:
        z = stats_res["ABS_ERRORS"][j]
        y = [j] * len(z)
        #x = range(len(z))
        x = [pd.to_datetime(d) for d in timestamps[1:]]
        #x = timestamps[1:]
        ax1.scatter(x, y, c=z, vmin=0, vmax=1, s=10, cmap=cm, 
                marker='s', label=series_labels[j])
    #plt.legend(loc='upper left')
    plt.show()

'''
Plot the contributions of each feature to the cumulative error
    - in this case the model was created and trained with a data set without
    idle periods. To create a nice plot we have to reinsert them
'''
def plot_error_contributions_DL_fixIdle(stats_res, series_labels, 
        timestamps, gaps, idle_periods, ts_noIdle, ts_idle):
    fig = plt.figure()

    cm = plt.cm.get_cmap('coolwarm')
    ax1 = fig.add_subplot(111)
    vmin = 0
    vmax = len(stats_res["ABS_ERRORS"]) + 1
    for j in stats_res["ABS_ERRORS"]:
        z = []
        k = 0
        for i in range(len(timestamps)):
            if timestamps[i] in ts_noIdle:
                z.append(stats_res["ABS_ERRORS"][j][k])
                k += 1
            else:
                z.append(0)
        y = [j] * len(z)
        x = [pd.to_datetime(d) for d in timestamps]
        ax1.scatter(x, y, c=z, vmin=0, vmax=1, s=10, cmap=cm, 
                marker='s', label=series_labels[j])
    plt.show()

'''
Plot gaps (grey) and idle periods (yellow)
'''
def plot_gaps_idle_periods_infig(stats_res, series_labels, timestamps, 
        sampling_time, gaps, idle_periods, fig, axarr, y_min, y_max):
    for st, et in gaps:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, 0), dif, 1, color='grey')
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='/', 
                color='grey')
        axarr[0].add_patch(rect)
    for st, et in idle_periods:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, 0), dif, 1, color='yellow')
        rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='//', 
                color='yellow')
        axarr[0].add_patch(rect)
    #for st, et in gaps:
    #    start = mdates.date2num(st)
    #    end = mdates.date2num(et)
    #    dif = end - start
    #    rect = patches.Rectangle((start, 0), dif, y_max, color='grey')
    #    rect = patches.Rectangle((start, 0), dif, y_max, fill=False, hatch='/', 
    #            color='grey')
    #    axarr[1].add_patch(rect)
    #for st, et in idle_periods:
    #    start = mdates.date2num(st)
    #    end = mdates.date2num(et)
    #    dif = end - start
    #    rect = patches.Rectangle((start, 0), dif, y_max, color='yellow')
    #    rect = patches.Rectangle((start, 0), dif, y_max, fill=False, hatch='//', 
    #            color='yellow')
    #    axarr[1].add_patch(rect)

'''
Plot individual feature contributions
'''
def plot_contributions_infig(stats_res, series_labels, timestamps,
        sampling_time, fig, axarr, y_min, y_max, ts_noIdle=[], ts_idle=[]):

    if len(ts_noIdle) > 0:
        n_feature_shown = plot_contributions_infig_fixIdle(stats_res, 
                series_labels, timestamps, fig, axarr, y_min, y_max, 
                ts_noIdle, ts_idle)
        return n_feature_shown

    n_feature_shown = 0
    if len(stats_res["ABS_ERRORS"]) < 100:
        threshold = 0.0
    else:
        threshold = _threshold_feat
    cm = plt.cm.get_cmap('coolwarm')
    for j in stats_res["ABS_ERRORS"]:
        if stats_res["MAE"][j] > threshold:
            n_feature_shown += 1
            z = stats_res["ABS_ERRORS"][j]
            y = [n_feature_shown] * len(z)
            x = [pd.to_datetime(d) for d in timestamps]
            x_idx = range(len(z))
            axarr[1].scatter(x, y, c=z, vmin=0, vmax=1, s=10, cmap=cm, 
                    marker='s', label=series_labels[j])
            margin = datetime.timedelta(seconds=sampling_time*5)
            axarr[1].text(x[-1] + margin, y[j], series_labels[j], fontsize=8)
        #axarr[1].set_yticklabels(series_labels_ext)
        #axarr[1].set_yticklabels(series_labels_ext, minor=True)
        #axarr[1].set_yticks(series_labels_ext)
    return n_feature_shown

'''
Plot individual feature contributions
    - in this case the model was created and trained with a data set without
    idle periods. To create a nice plot we have to reinsert them
'''
def plot_contributions_infig_fixIdle(stats_res, series_labels, timestamps,
        fig, axarr, y_min, y_max, ts_noIdle=[], ts_idle=[]):
    n_feature_shown = 0
    if len(stats_res["ABS_ERRORS"]) < 100:
        threshold = 0.0
    else:
        threshold = _threshold_feat
    cm = plt.cm.get_cmap('coolwarm')
    for j in stats_res["ABS_ERRORS"]:
        if stats_res["MAE"][j] > threshold:
            n_feature_shown += 1
            z = []
            k = 0
            for i in range(len(timestamps)):
                if timestamps[i] in ts_noIdle:
                    z.append(stats_res["ABS_ERRORS"][j][k])
                    k += 1
                else:
                    z.append(0)
            y = [n_feature_shown] * len(z)
            x = [pd.to_datetime(d) for d in timestamps]
            x_idx = range(len(z))
            axarr[1].scatter(x, y, c=z, vmin=0, vmax=1, s=10, cmap=cm, 
                    marker='s', label=series_labels[j])
            margin = datetime.timedelta(seconds=300*5)
            axarr[1].text(x[-1] + margin, y[j], series_labels[j], fontsize=8)
    return n_feature_shown

'''
Plot both cumulative errors and individual contributions
'''
def plot_errors_and_contributions_DL(stats_res, series_labels, timestamps, 
        sampling_time, gaps, idle_periods, ts_noIdle=[], ts_idle=[],
        test_idx=-1, freqGov='conservative', node="", title=""):
    if len(ts_noIdle) > 0:
        plot_errors_and_contributions_DL_fixIdle(stats_res, series_labels, 
                timestamps, sampling_time, gaps, idle_periods, 
                ts_noIdle, ts_idle, test_idx, freqGov)
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
    if len(timestamps) != len(cumul_errors_norm):
        #timestamps = timestamps[1:]
        timestamps = timestamps[(len(timestamps) - 
            len(cumul_errors_norm)):]

    fig, axarr = plt.subplots(2, sharex=True)

    if node != "":
        axarr[0].set_title(node)
    if title != "":
        axarr[0].set_title(title)
    axarr[1].set_title("Cumulative Error & Contributions")
    axarr[0].set_ylabel('Cumulative Errors')
    y_min = 0
    y_max = len(stats_res["ABS_ERRORS"]) + 1

    if "BEST_MODEL" not in stats_res:
        axarr[0].plot(timestamps, cumul_errors_norm)
        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        axarr[0].xaxis.set_major_locator(locator)
        axarr[0].xaxis.set_major_formatter(formatter)

        y_max_mod = plot_contributions_infig(stats_res, series_labels, 
                timestamps, sampling_time, fig, axarr, y_min, y_max, [], [])

        plot_gaps_idle_periods_infig(stats_res, series_labels, timestamps, 
            sampling_time, gaps, idle_periods, fig, axarr, y_min, y_max_mod)

        add_freq_govs_to_plot(freqGov, axarr[0])
    else:
        colors = [int(i % len(set(stats_res["BEST_MODEL"]))) 
                for i in stats_res["BEST_MODEL"]]
        timestamps_mod = [pd.to_datetime(t) for t in timestamps]
        axarr[0].scatter(timestamps_mod, cumul_errors_norm, c=colors)

    axarr[1].set_xlabel('Time')
    axarr[1].set_ylabel('Features Contributions')
    series_labels_ext = [0.0] 
    for l in series_labels:
        series_labels_ext.append(l)
    start, end = axarr[1].get_ylim()

    plt.show()

'''
Plot both cumulative errors and individual contributions
    - in this case the model was created and trained with a data set without
    idle periods. To create a nice plot we have to reinsert them
'''
def plot_errors_and_contributions_DL_fixIdle(stats_res, series_labels, 
        timestamps, sampling_time, gaps, idle_periods, 
        ts_noIdle, ts_idle, test_idx, freqGov, node="", title=""):

    cumul_errors = []
    cumul_errors_norm = []
    time_idxs = []
    
    for i in range(len(ts_noIdle)):
        cumul_errors.append(0)

    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
                cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

    # 'normalize' cumulated errors
    for ce in cumul_errors:
        cumul_errors_norm.append(ce / len(stats_res["ABS_ERRORS"]))

    # we assign an error equal to 0 to idle periods
    cumul_errors_withIdle = []
    cumul_errors_norm_withIdle = []
    k = 0
    for i in range(len(timestamps)):
        if timestamps[i] in ts_noIdle:
            cumul_errors_withIdle.append(cumul_errors[k])
            cumul_errors_norm_withIdle.append(cumul_errors_norm[k])
            k += 1
        else: 
            cumul_errors_withIdle.append('Nan')
            cumul_errors_norm_withIdle.append('Nan')

    n_idle = 0
    for i in range(len(cumul_errors_norm_withIdle)):
        if cumul_errors_norm_withIdle[i] == 0:
            n_idle += 1

    fig, axarr = plt.subplots(2, sharex=True)

    if node != "":
        axarr[0].set_title(node)
    if title != "":
        axarr[0].set_title(title)
    axarr[1].set_title("Cumulative Error & Contributions")
    axarr[0].set_ylabel('Cumulative Errors')
    y_min = 0
    y_max = len(stats_res["ABS_ERRORS"]) + 1

    if "BEST_MODEL" not in stats_res:
        axarr[0].plot(timestamps, cumul_errors_norm_withIdle)
        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        axarr[0].xaxis.set_major_locator(locator)
        axarr[0].xaxis.set_major_formatter(formatter)

        y_max_mod = plot_contributions_infig(stats_res, series_labels, 
                timestamps, sampling_time, fig, axarr, y_min, y_max, 
                ts_noIdle, ts_idle)

        plot_gaps_idle_periods_infig(stats_res, series_labels, timestamps, 
            sampling_time, gaps, idle_periods, fig, axarr, y_min, y_max_mod)

        add_freq_govs_to_plot(freqGov, axarr[0])
    else:
        colors = [int(i % len(set(stats_res["BEST_MODEL"]))) 
                for i in stats_res["BEST_MODEL"]]
        timestamps_mod = [pd.to_datetime(t) for t in timestamps]
        axarr[0].scatter(timestamps_mod, cumul_errors_norm, c=colors)

    axarr[1].set_xlabel('Time')
    axarr[1].set_ylabel('Features Contributions')
    series_labels_ext = [0.0] 
    for l in series_labels:
        series_labels_ext.append(l)
    start, end = axarr[1].get_ylim()

    plt.show()



'''
Plot both cumulative errors and individual contributions
    - multiple AE results
'''
def plot_errors_and_contributions_DL_multi(stats_res_multi, series_labels, 
        timestamps, sampling_time, gaps, idle_periods, node="", title=""):
    for ae, stats_res in stats_res_multi.iteritems():
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
        if len(timestamps) != len(cumul_errors_norm):
            timestamps = timestamps[1:]

        fig, axarr = plt.subplots(2, sharex=True)

        axarr[0].plot(timestamps, cumul_errors_norm)

        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        axarr[0].xaxis.set_major_locator(locator)
        axarr[0].xaxis.set_major_formatter(formatter)
        if node != "":
            axarr[0].set_title(node)
        if title != "":
            axarr[0].set_title(title)
        axarr[0].set_title("AE" + str(ae))
        axarr[1].set_title("Cumulative Error & Contributions")
        axarr[0].set_ylabel('Cumulative Errors')

        y_min = 0
        y_max = len(stats_res["ABS_ERRORS"]) + 1

        y_max_mod =plot_contributions_infig(stats_res,series_labels,timestamps,
            sampling_time, fig, axarr, y_min, y_max)

        plot_gaps_idle_periods_infig(stats_res, series_labels, timestamps, 
            sampling_time, gaps, idle_periods, fig, axarr, y_min, y_max_mod)
 
        axarr[1].set_xlabel('Time')
        axarr[1].set_ylabel('Features Contributions')
        series_labels_ext = [0.0] 
        for l in series_labels:
            series_labels_ext.append(l)
        start, end = axarr[1].get_ylim()

        #axarr[0].set_xlim([
        #    datetime.datetime.strptime("2017-06-14", "%Y-%m-%d"),
        #    datetime.datetime.strptime("2017-06-22", "%Y-%m-%d")])
        #    #datetime.datetime.strptime("2017-06-27", "%Y-%m-%d"),
        #    #datetime.datetime.strptime("2017-07-01", "%Y-%m-%d")])
        #axarr[1].set_xlim([
        #    datetime.datetime.strptime("2017-06-14", "%Y-%m-%d"),
        #    datetime.datetime.strptime("2017-06-22", "%Y-%m-%d")])
        #    #datetime.datetime.strptime("2017-06-27", "%Y-%m-%d"),
        #    #datetime.datetime.strptime("2017-07-01", "%Y-%m-%d")])

    plt.show()

'''
Plot both cumulative errors and individual contributions
'''
def plot_errors_and_contributions_and_other_DL(stats_res, df, 
        series_labels, timestamps, sampling_time, gaps, idle_periods, 
        node="", title=""):
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
    if len(timestamps) != len(cumul_errors_norm):
        timestamps = timestamps[1:]

    load_cols = []
    temp_cols = []
    for i in range(16):
        load_cols.append('load_core_' + str(i))
        temp_cols.append('temp_' + str(i))

    df['loads_means'] = df[load_cols].mean(axis=1)
    df['temps_means'] = df[temp_cols].mean(axis=1)
    loads_means = df['loads_means']
    temps_means = df['temps_means']

    if len(timestamps) < len(loads_means):
        loads_means = loads_means[1:]
    if len(timestamps) < len(temps_means):
        temps_means = temps_means[1:]

    fig, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(timestamps, cumul_errors_norm, label='errors')
    axarr[1].plot(timestamps, loads_means, 'r-', alpha=.7, label='avg load')
    axarr[1].plot(timestamps, temps_means, 'g-', alpha=.7,label='avg temp')

    core_id = 0
    load_sel = 'load_core_' + str(core_id)
    temp_sel = 'temp_' + str(core_id)
    loads = df[load_sel]
    temps = df[temp_sel]
    if len(timestamps) < len(loads):
        loads = loads[1:]
    if len(timestamps) < len(temps):
        temps = temps[1:]
    #axarr[1].plot(timestamps, loads, 'r-', alpha=.7, label=load_sel)
    #axarr[1].plot(timestamps, temps, 'g-', alpha=.7, label=temp_sel)

    axarr[0].legend()
    axarr[1].legend()
    #axarr[2].legend()

    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    axarr[0].xaxis.set_major_locator(locator)
    axarr[0].xaxis.set_major_formatter(formatter)
    if node != "":
        axarr[0].set_title(node)
    if title != "":
        axarr[0].set_title(title)
    axarr[0].set_ylabel('Cumulative Errors')

    plt.show()


'''
[s0, s1, s2, s3, s5,  ..]
s -> (s0, s1), (s2, s3), (s4, s5), ...
'''
def pairwise_A(iterable):
    a = iter(iterable)
    return zip(a, a)

'''
[s0, s1, s2, s3, s5,  ..]
s -> (s0, s1), (s1, s2), (s2, s3), ...
'''
def pairwise_B(iterable):
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
    for ts1, ts2 in pairwise_B(timestamps):
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
Given data (predicted, decoded by AE or else) that contains temporal info 
only in form of index in a list, return the list of associated timestamps
    - this function can be used only if we are sure that the values are
    continuous and consecutive (definitely not always our case..)
'''
def from_idx_2_timestamp(start_time, end_time, sampling_time, data):
    timestamps = []
    current_time = start_time
    for d in data:
        current_time += datetime.timedelta(seconds=sampling_time)
        timestamps.append(current_time)
    return timestamps
 
'''
Visualize model training history
'''
def visualize_train_history(history, metrics):
    # list all data in history
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.title('model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

'''
Print results of regressor evaluation
'''
def print_result(stats_res, series_labels, avg_only):
    MAEs = []
    MSEs = []
    RMSEs = []
    MAPEs = []
    SMAPEs = []
    R2s = []
    for j in range(len(series_labels)):
        if not avg_only:
            print("Series: %s" % series_labels[j])
            print("\tMAE: %0.3f" % stats_res["MAE"][j])
            print("\tMSE: %0.3f" % stats_res["MSE"][j])
            print("\tRMSE: %0.3f" % stats_res["RMSE"][j])
            print("\tMAPE: %0.3f" % stats_res["MAPE"][j])
            print("\tSMAPE: %0.3f" % stats_res["SMAPE"][j])
            print("\tR2: %0.3f" % stats_res["R2"][j])
        MAEs.append(stats_res["MAE"][j])
        MSEs.append(stats_res["MSE"][j])
        RMSEs.append(stats_res["RMSE"][j])
        MAPEs.append(stats_res["MAPE"][j])
        SMAPEs.append(stats_res["SMAPE"][j])
        R2s.append(stats_res["R2"][j])
    print("\nAverage Results:")
    print("\tAvg. MAE: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in MAEs])))
    print("\tAvg. MSE: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in MSEs])))
    print("\tAvg. RMSE: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in RMSEs])))
    print("\tAvg. MAPE: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in MAPEs])))
    print("\tAvg. SMAPE: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in SMAPEs])))
    print("\tAvg. R2: %0.3f" % np.nanmean(np.asarray(
        [float(x) for x in R2s])))

'''
Multiple AEs in cascade - each one focusing on the part misrepresented
by the previous one. Evaluate them in conjunction (for each point in the test
set choose the best AE)
'''
def compute_errors_and_plot_multi_modes(AEs, x_test, series_labels, 
        timestamps, _phys_sampling_time, gaps, idle_periods):
    decoded_features = {}
    stats_res = {}
    for ae in AEs.keys():
        decoded_features[ae] = AEs[ae].predict(x_test)

    stats_res["MIXED"] = evaluate_predictions_multi(
            decoded_features, x_test, gaps)

    plot_errors_DL(stats_res["MIXED"], series_labels, 
            timestamps, _phys_sampling_time, gaps,  idle_periods)

    return stats_res["MIXED"]

'''
Combine information on AE prediction (input recreation..) results and
clustering results based on hidden layer 
'''
def combine_predictions_and_cluster(decoded_features, x_train, gaps,
        stats_res, cluster_predictions, timestamps, title,
        active_idle, ts_noIdle, ts_idle, freqGov):

    if len(ts_noIdle) > 0:
        combine_predictions_and_cluster_fixIdle(decoded_features, 
                x_train, gaps, stats_res, cluster_predictions, timestamps, 
                title, ts_noIdle, ts_idle, freqGov)
        return

    idxs = []
    for i in range(len(cluster_predictions)):
        idxs.append(i)

    fig = plt.figure()
    colors = [int(i % len(set(cluster_predictions))) 
            for i in cluster_predictions]

    timestamps_mod = [pd.to_datetime(t) for t in timestamps]
    plt.scatter(idxs, stats_res["CUMUL_ABS_ERRORS_NORM"], c=colors)
    plt.title(title + "(# Clusters: " + str(len(set(colors))) + ")")
    plt.show()

'''
Combine information on AE prediction (input recreation..) results and
clustering results based on hidden layer 
    - in this case the model was created and trained with a data set without
    idle periods. To create a nice plot we have to reinsert them
'''
def combine_predictions_and_cluster_fixIdle(decoded_features, x_train, 
        gaps, stats_res, cluster_predictions, timestamps, title, 
        ts_noIdle, ts_idle, freqGov):
    idxs = []
    cluster_predictions_withIdle = []
    cumul_errors_norm_withIdle = []
    k = 0

    if len(ts_noIdle) > len(cluster_predictions):
        ts_noIdle = ts_noIdle[:-1]

    for i in range(len(timestamps)):
        idxs.append(i)
        if timestamps[i] in ts_noIdle:
            cluster_predictions_withIdle.append(cluster_predictions[k])
            cumul_errors_norm_withIdle.append(
                    stats_res["CUMUL_ABS_ERRORS_NORM"][k])
            k += 1
        else:
            cluster_predictions_withIdle.append(len(set(cluster_predictions)))
            cumul_errors_norm_withIdle.append(0)

    fig = plt.figure()
    colors = [__m.color_maps[len(set(cluster_predictions_withIdle))][
        int(i % len(set(cluster_predictions_withIdle)))] 
            for i in cluster_predictions_withIdle]

    #colors = [int(i % len(set(cluster_predictions_withIdle))) 
    #        for i in cluster_predictions_withIdle]
    #colors = [int(i % len(set(cluster_predictions))) 
    #        for i in cluster_predictions]

    #markers = [int(i % len(set(cluster_predictions_withIdle))) 
    #        for i in cluster_predictions_withIdle]

    timestamps_mod = [pd.to_datetime(t) for t in timestamps]

    plt.scatter(timestamps_mod, cumul_errors_norm_withIdle, c=colors)
    plt.title(title + "(# Clusters: " + str(len(set(colors)) - 1) + ")")
    ax = fig.add_subplot(111)
    add_freq_govs_to_plot(freqGov, ax, True)

    plt.show()

'''
Combine information on AE prediction (input recreation..) results and
clustering results based on hidden layer 
    - in this case the model was created and trained with a data set without
    idle periods. To create a nice plot we have to reinsert them
    - the AE was trained splitting the data set in train & test sets
    - in this plot the different frequency governor are highlighted too
'''
def combine_predictions_and_cluster_split(decoded_features, data, 
        gaps, stats_res, cluster_predictions, timestamps, title, 
        ts_noIdle, ts_idle, freqGov):
    idxs = []
    cluster_predictions_withIdle = []
    cumul_errors_norm_withIdle = []
    k = 0
    for i in range(len(timestamps)):
        idxs.append(i)
        if timestamps[i] in ts_noIdle:
            cluster_predictions_withIdle.append(cluster_predictions[k])
            cumul_errors_norm_withIdle.append(
                    stats_res["CUMUL_ABS_ERRORS_NORM"][k])
            k += 1
        else:
            cluster_predictions_withIdle.append(len(set(cluster_predictions)))
            cumul_errors_norm_withIdle.append(0)

    fig = plt.figure()
    colors = [__m.color_maps[len(set(cluster_predictions_withIdle))][
        int(i % len(set(cluster_predictions_withIdle)))] 
            for i in cluster_predictions_withIdle]

    timestamps_mod = [pd.to_datetime(t) for t in timestamps]

    #plt.scatter(idxs, cumul_errors_norm_withIdle, c=colors)
    plt.scatter(timestamps_mod, cumul_errors_norm_withIdle, c=colors)
    ax = fig.add_subplot(111)
    add_freq_govs_to_plot(freqGov, ax, True)
    plt.title(title + "(# Clusters: " + str(len(set(colors)) - 1) + ")")
    plt.show()

'''
Add frequency governor types to plot
    - only highlight the time periods where the freq governor is not default
'''
def add_freq_govs_to_plot(freqGov, ax, cluster=False):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
        first_color = 'blue'
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
        first_color = 'red'
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
    else:
        freqGov_periods = []

    if freqGov == 'perf_power':
        second_color = 'red'
        two_anomalies = True
    elif freqGov == 'power_perf':
        second_color = 'blue'
        two_anomalies = True
    else:
        two_anomalies = False

    if not two_anomalies:
        freq_govs_plot_singleAnomaly(ax, freqGov_periods, first_color, cluster)
    else:
        freq_govs_plot_twoAnomaly(ax, freqGov_periods, first_color, 
                second_color, cluster)
        
    #for st, et in freqGov_periods:
    #    start = mdates.date2num(st)
    #    end = mdates.date2num(et)
    #    dif = end - start
    #    if not cluster:
    #        rect = patches.Rectangle((start, -0.002), dif, 0.002, fill=True, 
    #                color=first_color)
    #    else:
    #        rect = patches.Rectangle((start, -0.005), dif, 0.004, fill=True, 
    #                color='darkviolet')
    #    ax.add_patch(rect)

'''
Add frequency governor types to plot
- second version (used for journal paper)
'''
def add_freq_govs_to_plot2(freqGov, ax, hmin, hmax):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
        first_color = 'darkturquoise'
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
        first_color = 'c'
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
    else:
        freqGov_periods = []

    if freqGov == 'perf_power':
        second_color = 'c'
        two_anomalies = True
    elif freqGov == 'power_perf':
        second_color = 'darkturquoise'
        two_anomalies = True
    else:
        two_anomalies = False

    for st, et in freqGov_periods[:-1]:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, hmin), dif, hmax, 
                fill=False, hatch='/', color=first_color)
        ax.add_patch(rect)

    (st, et) = freqGov_periods[-1]
    start = mdates.date2num(st)
    end = mdates.date2num(et)
    dif = end - start
    rect = patches.Rectangle((start, hmin), dif, hmax, 
            fill=False, hatch='/', color=second_color)
    ax.add_patch(rect)

def freq_govs_plot_singleAnomaly(ax, freqGov_periods, color, cluster):
    for st, et in freqGov_periods:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        if not cluster:
            rect = patches.Rectangle((start, -0.005), dif, 0.05, fill=True, 
                    color=color)
        else:
            rect = patches.Rectangle((start, -0.005), dif, 0.008, fill=True, 
                    color='darkviolet')
        ax.add_patch(rect)

'''
We assume that the second anomaly appears only in the last period..
'''
def freq_govs_plot_twoAnomaly(ax, freqGov_periods, first_color, 
        second_color, cluster):
    for st, et in freqGov_periods[:-1]:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        if not cluster:
            rect = patches.Rectangle((start, -0.002), dif, 0.012, fill=True, 
                    #color=first_color)
                    color=second_color)
        else:
            rect = patches.Rectangle((start, -0.005), dif, 0.008, fill=True, 
                    color='darkviolet')
        ax.add_patch(rect)

    (st, et) = freqGov_periods[-1]
    start = mdates.date2num(st)
    end = mdates.date2num(et)
    dif = end - start
    if not cluster:
        rect = patches.Rectangle((start, -0.002), dif, 0.012, fill=True, 
                #color=second_color)
                color=first_color)
    else:
        rect = patches.Rectangle((start, -0.005), dif, 0.008, fill=True, 
                color='violet')
    ax.add_patch(rect)

'''
We assume that the second anomaly appears only in the last period..
- second versione 
'''
def freq_govs_plot_twoAnomaly2(ax, freqGov_periods, first_color, 
        second_color):
    for st, et in freqGov_periods[:-1]:
        start = mdates.date2num(st)
        end = mdates.date2num(et)
        dif = end - start
        rect = patches.Rectangle((start, -0.045), dif, 0.02, fill=True, 
                color=first_color)
        ax.add_patch(rect)

    (st, et) = freqGov_periods[-1]
    start = mdates.date2num(st)
    end = mdates.date2num(et)
    dif = end - start
    rect = patches.Rectangle((start, -0.045), dif, 0.02, fill=True, 
            color=second_color)
    ax.add_patch(rect)

'''
Add frequency governor types to plot
    - only highlight the time periods where the freq governor is not default
    - work with indexes and not timestamps
'''
def add_freq_govs_to_plot_idx(freqGov, ax, ts_noIdle, n_hidden):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
        first_color = 'blue'
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
        first_color = 'red'
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
    else:
        freqGov_periods = []

    if freqGov == 'perf_power':
        second_color = 'red'
        two_anomalies = True
    elif freqGov == 'power_perf':
        second_color = 'blue'
        two_anomalies = True
    else:
        two_anomalies = False

    idxs = []
    for st, et in freqGov_periods:
        prev_t = ts_noIdle[0]
        for i in range(len(ts_noIdle[1:])):
            if prev_t <= st < ts_noIdle[i]:
                st_idx = i
                break
            prev_t = ts_noIdle[i]
        prev_t = ts_noIdle[0]
        for i in range(len(ts_noIdle[1:])):
            if prev_t <= et < ts_noIdle[i]:
                et_idx = i
                break
            prev_t = ts_noIdle[i]
        idxs.append((st_idx, et_idx))

        dif = et_idx - st_idx
        #rect = patches.Rectangle((st_idx, 0), dif, 5, fill=True, 
        #        color='red')
        rect = patches.Rectangle((st_idx, 3), dif, n_hidden-4, linewidth=2, 
                edgecolor='k', facecolor='none')

        ax.add_patch(rect)

'''
Get the job nodes as list
    - limitation: we expect all node names to have the same prefix
'''
def nodes_to_list(nodes):
    node_list = []
    # davide44
    if '[' not in nodes and  ']' not in nodes:
        node_list.append(nodes)

    #davide[31-43]
    elif ',' not in nodes:
        chars = re.findall('[a-zA-Z]+', nodes)
        if len(chars) < 1:
            return [-1]
        node_name_str = str(chars[0])
        digits = re.findall('\d+', nodes)
        d_max = -sys.maxint - 1
        d_min = sys.maxint
        for d in digits:
            d_int = int(d)
            if d_int < d_min:
                d_min = d_int
            if d_int > d_max:
                d_max = d_int
        for i in range(d_min, d_max+1):
            node_name = node_name_str + str(i)
            node_list.append(node_name)

    # davide[2-6,16-22,31-44]
    else:
        chars = re.findall('[a-zA-Z]+', nodes)
        if len(chars) < 1:
            return [-1]
        node_name_str = str(chars[0])

        sub_str = nodes.split(']')[0].split('[')[1]
        pieces = sub_str.split(',')

        for piece in pieces:
            if '-' not in piece:
                digit = re.findall('\d+', piece)
                node_list.append(node_name_str + str(digit[0]))
            else:
                digits = re.findall('\d+', piece)
                d_max = -sys.maxint - 1
                d_min = sys.maxint
                for d in digits:
                    d_int = int(d)
                    if d_int < d_min:
                        d_min = d_int
                    if d_int > d_max:
                        d_max = d_int
                for i in range(d_min, d_max+1):
                    node_name = node_name_str + str(i)
                    node_list.append(node_name)
    return node_list

'''
Retrieve jobs info from Cassandra or local host for a given node
'''
def retrieve_jobs_info_node(start_time, end_time, node):
    config = ConfigParser.RawConfigParser()
    config.read('../config.conf')

    # Davide
    DAVIDE_USR = config.get('Davide','DAVIDE_USR')
    DAVIDE_PWD = config.get('Davide','DAVIDE_PWD')
    DAVIDE_IP = config.get('Davide','DAVIDE_IP')

    # Cassandra
    CASSANDRA_USR = config.get('Cassandra','CASSANDRA_USR')
    CASSANDRA_PWD = config.get('Cassandra','CASSANDRA_PWD')
    CASSANDRA_IP = config.get('Cassandra','CASSANDRA_IP')
    CASSANDRA_IP_TUNNEL = config.get('Cassandra','CASSANDRA_IP_TUNNEL')
    CASSANDRA_PORT = config.get('Cassandra','CASSANDRA_PORT')
    CASSANDRA_JOB_KEYSPACE = config.get('Cassandra','CASSANDRA_JOB_KEYSPACE')
    CASSANDRA_PHYS_KEYSPACE = config.get('Cassandra','CASSANDRA_PHYS_KEYSPACE')

    # Tables
    JOBS_TABLE = config.get('Tables','JOBS_TABLE')
    JOBS_AGGR = config.get('Tables','JOBS_AGGR')
    PHYS_AGGR_TABLE_ipmi = config.get('Tables','PHYS_AGGR_TABLE_ipmi')
    PHYS_AGGR_TABLE_bbb = config.get('Tables','PHYS_AGGR_TABLE_bbb')
    PHYS_AGGR_TABLE_occ = config.get('Tables','PHYS_AGGR_TABLE_occ')

    jobs_info = job = load_jobs(__m.offline_jobs_file_info, 'INFO',
            CASSANDRA_IP, CASSANDRA_PORT, CASSANDRA_USR, CASSANDRA_PWD,
            CASSANDRA_JOB_KEYSPACE, JOBS_TABLE)

    jobs = {}
    for job_id in jobs_info:
        job = jobs_info[job_id]
        st = job['start_time']
        et = job['end_time']
        nodes = nodes_to_list(job['nodes'])
        if et > start_time and st < end_time and node in nodes:
            jobs[job_id] = job
    return jobs

'''
Retrieve physical data from Cassandra or local host
'''
def retrieve_data(start_time, end_time, nodes):
    config = configparser.RawConfigParser()
    config.read('../config.conf')

    # Davide
    DAVIDE_USR = config.get('Davide','DAVIDE_USR')
    DAVIDE_PWD = config.get('Davide','DAVIDE_PWD')
    DAVIDE_IP = config.get('Davide','DAVIDE_IP')

    # Cassandra
    CASSANDRA_USR = config.get('Cassandra','CASSANDRA_USR')
    CASSANDRA_PWD = config.get('Cassandra','CASSANDRA_PWD')
    CASSANDRA_IP = config.get('Cassandra','CASSANDRA_IP')
    CASSANDRA_IP_TUNNEL = config.get('Cassandra','CASSANDRA_IP_TUNNEL')
    CASSANDRA_PORT = config.get('Cassandra','CASSANDRA_PORT')
    CASSANDRA_JOB_KEYSPACE = config.get('Cassandra','CASSANDRA_JOB_KEYSPACE')
    CASSANDRA_PHYS_KEYSPACE = config.get('Cassandra','CASSANDRA_PHYS_KEYSPACE')

    # Tables
    JOBS_TABLE = config.get('Tables','JOBS_TABLE')
    JOBS_AGGR = config.get('Tables','JOBS_AGGR')
    PHYS_AGGR_TABLE_ipmi = config.get('Tables','PHYS_AGGR_TABLE_ipmi')
    PHYS_AGGR_TABLE_bbb = config.get('Tables','PHYS_AGGR_TABLE_bbb')
    PHYS_AGGR_TABLE_occ = config.get('Tables','PHYS_AGGR_TABLE_occ')

    phys_data = {}
    phys_data_ipmi = {}
    phys_data_bbb = {}
    phys_data_occ = {}

    #for node in read_node_whitelist():
    #    print("Reading node %s" % node)
    for node in nodes:
        print("\tLoading node %s data" % node)

        # this time period is stored in another folder
        # 20180301_130001_20180322_100001
        dt_str = str(start_time).replace('-','').replace(
                ':','').replace(' ','_') + '_'
        if "20180301_130001_" in dt_str:
            offline_phys_dir = "/home/b0rgh/data_collection_analysis/"
            offline_phys_dir += "public_standalone_projects/"
            offline_phys_dir += "anomaly_detection_davide/data/phys_info/"

        # IPMI
        offline_phys_file = __m.offline_phys_dir + node + '_ipmi_'
        offline_phys_file += str(start_time).replace('-','').replace(
                ':','').replace(' ','_') + '_'
        offline_phys_file += str(end_time).replace('-','').replace(
                ':','').replace(' ','_') + '.pickle'
        phys_data_ipmi[node] = load_phys(node, start_time,
                end_time, offline_phys_file, CASSANDRA_IP, CASSANDRA_PORT,
                CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_PHYS_KEYSPACE, 
                PHYS_AGGR_TABLE_ipmi)
        # OCC
        offline_phys_file = __m.offline_phys_dir + node + '_occ_'
        offline_phys_file += str(start_time).replace('-','').replace(
                ':','').replace(' ','_') + '_'
        offline_phys_file += str(end_time).replace('-','').replace(
                ':','').replace(' ','_') + '.pickle'
        phys_data_occ[node] = load_phys(node, start_time,
                end_time, offline_phys_file, CASSANDRA_IP, CASSANDRA_PORT,
                CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_PHYS_KEYSPACE, 
                PHYS_AGGR_TABLE_occ)
        # BBB
        offline_phys_file = __m.offline_phys_dir + node + '_bbb_'
        offline_phys_file += str(start_time).replace('-','').replace(
                ':','').replace(' ','_') + '_'
        offline_phys_file += str(end_time).replace('-','').replace(
                ':','').replace(' ','_') + '.pickle'
        phys_data_bbb[node] = load_phys(node, start_time,
                end_time, offline_phys_file, CASSANDRA_IP, CASSANDRA_PORT,
                CASSANDRA_USR, CASSANDRA_PWD, CASSANDRA_PHYS_KEYSPACE, 
                PHYS_AGGR_TABLE_bbb)

    return phys_data_ipmi, phys_data_bbb, phys_data_occ

'''
Fix wrong values, sort, merge
'''
def create_df(phys_data_ipmi, phys_data_bbb, phys_data_occ):
    print('\t >> Creating df')
    # OCC measurements appear to have a 'wrong' timestamp, one second later
    # than the real value -- fix this
    #nodes = list(phys_data_ipmi.keys())
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

    # 'normalize' cumulated errors (aka average error)
    for i in range(len(ts_noIdle)):
        errors.append(0)
    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
                errors[i] += stats_res["ABS_ERRORS"][j][i]
    for ce in errors:
        errors_stat.append(ce / len(stats_res["ABS_ERRORS"]))

    # standard deviation
    #errors_avg = []
    #errors_temp = []
    #for i in range(len(ts_noIdle)):
    #    errors.append(0)
    #    errors_temp.append(0)
    #for j in stats_res["ABS_ERRORS"].keys():
    #    for i in range(len(stats_res["ABS_ERRORS"][j])):
    #        errors[i] += stats_res["ABS_ERRORS"][j][i]
    #for e in errors:
    #    errors_avg.append(e / len(stats_res["ABS_ERRORS"]))
    #for j in stats_res["ABS_ERRORS"].keys():
    #    for i in range(len(stats_res["ABS_ERRORS"][j])):
    #        errors_temp[i] += ((
    #                stats_res["ABS_ERRORS"][j][i] - errors_avg[i]) ** 2)
    #for e in errors_temp:
    #    errors_stat.append(math.sqrt(e / (len(stats_res["ABS_ERRORS"]) - 1)))

    # max 
    #for i in range(len(ts_noIdle)):
    #    errors.append(0)
    #for j in stats_res["ABS_ERRORS"].keys():
    #    for i in range(len(stats_res["ABS_ERRORS"][j])):
    #        if errors[i] < stats_res["ABS_ERRORS"][j][i]:
    #            errors[i] = stats_res["ABS_ERRORS"][j][i]
    #errors_stat = errors

    # n percentiile
    #n_perc = 95
    #for i in range(len(ts_noIdle)):
    #    errors.append([])
    #for j in stats_res["ABS_ERRORS"].keys():
    #    for i in range(len(stats_res["ABS_ERRORS"][j])):
    #            errors[i].append(stats_res["ABS_ERRORS"][j][i])
    #for e in errors:
    #    errors_stat.append(np.percentile(np.asarray(e), n_perc))

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
            errors_withIdle.append('Nan')
            errors_stat_withIdle.append('Nan')

    n_idle = 0
    for i in range(len(errors_stat_withIdle)):
        if errors_stat_withIdle[i] == 0:
            n_idle += 1

    fig = plt.figure()

    if "BEST_MODEL" not in stats_res:
        plt.plot(timestamps, errors_stat_withIdle, linewidth=2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        #if test_idx != -1:
        #    test_timestamp = ts_noIdle[test_idx]
        #    plt.axvline(x=test_timestamp, linewidth=2, 
        #            linestyle='--', color='k')

        ax = fig.add_subplot(111)
        #for st, et in gaps:
        #    start = mdates.date2num(st)
        #    end = mdates.date2num(et)
        #    dif = end - start
        #    rect = patches.Rectangle((start, 0), dif, 1, color='grey')
        #    rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='/', 
        #            color='grey')
        #    ax.add_patch(rect)
        #for st, et in idle_periods:
        #    start = mdates.date2num(st)
        #    end = mdates.date2num(et)
        #    dif = end - start
        #    rect = patches.Rectangle((start, 0), dif, 1, color='yellow')
        #    rect = patches.Rectangle((start, 0), dif, 1, fill=False, hatch='//', 
        #            color='yellow')
        #    ax.add_patch(rect)

        add_freq_govs_to_plot(freqGov, ax)

        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    else:
        ax = fig.add_subplot(111)
        colors = [int(i % len(set(stats_res["BEST_MODEL"]))) 
                for i in stats_res["BEST_MODEL"]]
        timestamps_mod = [pd.to_datetime(t) for t in timestamps]
        plt.scatter(timestamps_mod, errors_stat, c=colors)

    plt.title(title)
    plt.show()

'''
Check if a data frame has already been prepared
    - no need to load it and prepare it twice
'''
def check_df(start_time, end_time, node, remove_idle, unscaled=False):
    offline_df_file = __m.offline_phys_dir + node + '_'
    offline_df_file += str(start_time).replace('-','').replace(
            ':','').replace(' ','_') + '_'
    offline_df_file += str(end_time).replace('-','').replace(
            ':','').replace(' ','_') 
    if remove_idle:
        offline_df_file += '_preparedDF_noIdle'
    else:
        offline_df_file += '_preparedDF_withIdle'

    if unscaled:
        offline_df_file += '_unscaled'

    offline_df_file += '.pickle'

    if os.path.isfile(offline_df_file):
        # the offline data frame has already been prepared
        return True, offline_df_file
    else:
        return False, offline_df_file

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
Check if a timestamp is in a idle period
- idle periods are expressed as a list composed by start/end time
'''
def is_timestamp_in_idle_period(idle_periods, timestamp):
    for st, et in idle_periods:
        if st <= timestamp < et:
            return True
    return False

'''
Check if a timestamp is in a frequency governor modified period
'''
def is_timestamp_in_freqGov_mod_period(freqGov, timestamp):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
    else:
        freqGov_periods = []

    for st, et in freqGov_periods:
        if st <= timestamp < et:
            return True
    return False

'''
There's a discrepancy between the time format of aggregated data
and time format used by SLURM, Graphana and jobs info table.
Probably aggregated data timestamps were converted to UTC while the rest
has not. This is not a problem if we use only aggregated data but becomes
a problem if we want to cross data source.
This function fix this problem.
!!!
This is a terrible fix (for example you should manually change the code
depending if DST is on or not) ----> EXERCISE EXTREME CAUTIONS!!!!
ALWAYS DOUBLE CHECK THE RESULTS WITH THE AID OF OTHER TOOLS (GRAPHANA,
SLURM RECORDS, JOBS INFO TABLE DATA, AGGREGATED TABLES DATA)
!!!
TODO: fix the fixing function (ahem..)
'''
def fix_time(timestamp):
    DST = True
    if DST:
        time_delta = 7200  # in seconds
    else:
        # I'm not sure of this value.. 
        # the function was tested only during DST.. good luck future myself
        time_delta = 3600  # in seconds
    
    tdelta = datetime.timedelta(seconds=time_delta)
    return timestamp - tdelta

'''
Group jobs depending on the input clusters
- Works only on single nodes (specified as input)
'''
def group_jobs_cluster(decoded_features, x_train, gaps, stats_res, 
        cluster_predictions, timestamps, title, active_idle, 
        ts_noIdle, ts_idle, node, test_start_time, test_end_time):

    # we assume that timestamsp have already been sorted
    if test_start_time != -1 and test_end_time != -1:
        jobs = retrieve_jobs_info_node(test_start_time, 
                test_end_time, node)
    else:
        jobs = retrieve_jobs_info_node(timestamps[0], 
                timestamps[len(timestamps)-1], node)

    for job in jobs.keys():
        jobs[job]['start_time'] = fix_time(jobs[job]['start_time'])
        jobs[job]['end_time'] = fix_time(jobs[job]['end_time'])

    k = 0
    for i in range(len(timestamps)):
        # discard cluster of idle periods (we know there are no jobs..)
        if not timestamps[i] in ts_noIdle:
            continue
        k += 1

    job_clusters = {}
    for job_id in jobs.keys(): 
        st = jobs[job_id]['start_time']
        et = jobs[job_id]['end_time']
        if (et - st).seconds < __m.duration_threshold:
            continue

        k = 0
        prev_ts = __m.epoch
        for i in range(len(timestamps)):
            if timestamps[i] > et and prev_ts > et:
                break
            # discard cluster of idle periods (we know there are no jobs..)
            if not timestamps[i] in ts_noIdle:
                #prev_ts = timestamps[i]
                continue
            cluster = cluster_predictions[k]
            k += 1

            if(st <= timestamps[i] <= et or (
                st > prev_ts and et < timestamps[i])):
                if job_id not in job_clusters:
                    job_clusters[job_id] = [cluster]
                else:
                    if cluster not in job_clusters[job_id]:
                        job_clusters[job_id].append(cluster)

            prev_ts = timestamps[i]

    cluster_jobs = {}
    for cluster in set(cluster_predictions):
        cluster_jobs[cluster] = []
    for job_id, clusters in job_clusters.iteritems():
        for cluster in clusters:
            if job_id not in cluster_jobs[cluster]:
                cluster_jobs[cluster].append(job_id)

    for cluster, jobs in cluster_jobs.iteritems():
        print("Cluster %s" % cluster)
        print("Jobs: ")
        print("\t %s" % jobs)
        print("=========================================================")

'''
Split data set randomly
- need to carefully avoid to put anomalies in data set
- this is the original version (better to use only with single anomaly)
'''
def split_dataset_old(df, ts_noIdle, freqGov):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
    else:
        freqGov_periods = []

    anomalies  = []
    all_idxs = []
    for idx in range(len(ts_noIdle)):
        all_idxs.append(idx)
    for idx in range(len(ts_noIdle)):
        ts = ts_noIdle[idx]
        for st, et in freqGov_periods:
            if st <= ts < et:
                anomalies.append(idx)
                break
    no_anomalies = list(set(all_idxs) - set(anomalies))
    df_anomalies = df[anomalies]
    df_noAnomalies = df[no_anomalies]

    msk = np.random.rand(len(df_noAnomalies)) < 0.8
    train = df_noAnomalies[msk]
    test = df_noAnomalies[~msk]

    test = np.concatenate((test, df_anomalies), axis=0)
    return train, test

'''
Split data set randomly
- new version, also returns a list with info about anomalies
'''
def split_dataset(df, ts_noIdle, freqGov):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
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

    msk = np.random.rand(len(df_noAnomalies)) < 0.8
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

    test = np.concatenate((test, df_anomalies), axis=0)

    return train, test, test_train_or_else_idxs, df_noAnomalies, df_anomalies

'''
Split data set randomly
- new version, also returns a list with info about anomalies
- returns distinct test set with and w/o anomalies
'''
def split_dataset_2(df, ts_noIdle, freqGov):
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
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
    if freqGov == 'performance' or freqGov == 'perf_power':
        freqGov_periods = _performance_periods
    elif freqGov == 'powersave' or freqGov == 'power_perf':
        freqGov_periods = _powersave_periods
    elif freqGov == 'ondemand':
        freqGov_periods = _ondemand_periods
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
Examine output of hidden layer
'''
def look_at_hidden_layer(output, freqGov, ts_noIdle):
    n_out, out_len = output.shape

    fig, ax = plt.subplots()
    im = ax.imshow(output.reshape(out_len, n_out), 
            #cmap=plt.get_cmap('coolwarm'), vmin=0, vmax=1)
            cmap=plt.get_cmap('coolwarm'))
    fig.colorbar(im)
    add_freq_govs_to_plot_idx(freqGov, ax, ts_noIdle, out_len)

    plt.show()
    sys.exit()

'''
Analyse reconstruction errors distributions
'''
def error_distribution(actual_normal, pred_normal, 
        actual_anomal, pred_anomal, node):
    print("# Normal actual: %s" % len(actual_normal))
    print("# Normal pred: %s" % len(pred_normal))
    print("# Anomaly actual: %s" % len(actual_anomal))
    print("# Anomaly pred: %s" % len(pred_anomal))

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

    # 'normalized' cumulated errors (aka average error)
    #for j in abs_errors_normal.keys():
    #    for i in range(len(abs_errors_normal[j])):
    #        errors_normal[i] += abs_errors_normal[j][i]
    #errors_normal[:] = [x / len(errors_normal) for x in errors_normal]

    # max error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]

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

    # 'normalized' cumulated errors (aka average error)
    #for j in abs_errors_anomal.keys():
    #    for i in range(len(abs_errors_anomal[j])):
    #        errors_anomal[i] += abs_errors_anomal[j][i]
    #errors_anomal[:] = [x / len(errors_anomal) for x in errors_anomal]

    # max error 
    for j in abs_errors_anomal.keys():
        for i in range(len(abs_errors_anomal[j])):
            if errors_anomal[i] < abs_errors_anomal[j][i]:
                errors_anomal[i] = abs_errors_anomal[j][i]


    anomal_chi2, anomal_pVal = stats.normaltest(errors_anomal)
    print("Normal test (anomaly data set): chi-squared %f; p-value %f" % (
            anomal_chi2, anomal_pVal))

    fig = plt.figure()
    plt.hist(errors_normal, bins=80)
    plt.ylabel('# data points')
    plt.xlabel('Error')
    #plt.title(node + ' - Normal Behaviour')

    fig = plt.figure()
    plt.hist(errors_anomal, bins=50)
    plt.ylabel('# data points')
    plt.xlabel('Error')
    #plt.title(node + ' - Anomalies')

    plt.show()

'''
This function computes precision, recall, F-Score in the 2 classes case:
    anomaly or normal behaviour.
    - works for the output generated by the autoencoder (unsupervised)
The parameter class target specifies if we are computing the stats for the 
normal or anomaly class
'''
def fscore(errors, error_threshold, classes, class_target):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i = 0
    print("\tfscore, class target %s, error threshold %s" % (
            class_target, error_threshold))
    if class_target == 0:            # fscore for normal classes
        for e in errors:
            if e <= error_threshold: # we predict normal class
                if classes[i] == 0:  # normal actual
                    tp += 1
                else:                # anomaly actual (false positive)
                    fp += 1
            else:                    # we predict anomaly
                if classes[i] == 0:  # normal actual (false negative)
                    fn += 1
                else:                # anomaly actual
                    tn += 1
            i += 1

    else:                            # fscore for anomaly classes
        for e in errors:
            if e > error_threshold:  # we predict anomaly 
                if classes[i] == 1:  # anomaly actual
                    tp += 1
                else:                # normal actual (false positive)
                    fp += 1
            else:                    # we predict normal
                if classes[i] == 1:  # anomaly actual (false negative)
                    fn += 1
                else:                # normal actual
                    tn += 1
            i += 1

    print("\ttp %s" % tp)
    print("\tfp %s" % fp)
    print("\ttn %s" % tn)
    print("\tfn %s" % fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    fscore = stats.hmean([precision, recall])
    return precision, recall, fscore

'''
Analyse reconstruction errors distributions
    - extend previous function in order to classify data points in
    anomalous or normal classes
    - compute some statistics
'''
def error_distribution_2_class(actual_normal, pred_normal, 
        actual_anomal, pred_anomal, node, actual_normal_all, pred_normal_all):
    print("# Normal actual: %s" % len(actual_normal))
    print("# Normal pred: %s" % len(pred_normal))
    print("# Normal actual (all): %s" % len(actual_normal_all))
    print("# Normal pred (all): %s" % len(pred_normal_all))
    print("# Anomaly actual: %s" % len(actual_anomal))
    print("# Anomaly pred: %s" % len(pred_anomal))

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
    print("Max Abs Error")
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

    # average abs error
    #print("Average Abs Error")
    #for j in abs_errors_normal.keys():
    #    for i in range(len(abs_errors_normal[j])):
    #        errors_normal[i] += abs_errors_normal[j][i]
    #errors_normal[:] = [x / len(errors_normal) for x in errors_normal]
    #for j in abs_errors_normal_all.keys():
    #    for i in range(len(abs_errors_normal_all[j])):
    #        errors_normal_all[i] += abs_errors_normal_all[j][i]
    #errors_normal_all[:] = [
    #        x / len(errors_normal_all) for x in errors_normal_all]
    #for j in abs_errors_anomal.keys():
    #    for i in range(len(abs_errors_anomal[j])):
    #        errors_anomal[i] += abs_errors_anomal[j][i]
    #errors_anomal[:] = [x / len(errors_anomal) for x in errors_anomal]

    # sum abs error
    #print("Sum Abs Error")
    #for j in abs_errors_normal.keys():
    #    for i in range(len(abs_errors_normal[j])):
    #        errors_normal[i] += abs_errors_normal[j][i]
    #for j in abs_errors_normal_all.terkeys():
    #    for i in range(len(abs_errors_normal_all[j])):
    #        errors_normal_all[i] += abs_errors_normal_all[j][i]
    #for j in abs_errors_anomal.keys():
    #    for i in range(len(abs_errors_anomal[j])):
    #        errors_anomal[i] += abs_errors_anomal[j][i]

    # max squared error 
    #print("Max Squared Error")
    #for j in squared_errors_normal.keys():
    #    for i in range(len(squared_errors_normal[j])):
    #        if errors_normal[i] < squared_errors_normal[j][i]:
    #            errors_normal[i] = squared_errors_normal[j][i]
    #for j in squared_errors_normal_all.keys():
    #    for i in range(len(squared_errors_normal_all[j])):
    #        if errors_normal_all[i] < squared_errors_normal_all[j][i]:
    #            errors_normal_all[i] = squared_errors_normal_all[j][i]
    #for j in squared_errors_anomal.keys():
    #    for i in range(len(squared_errors_anomal[j])):
    #        if errors_anomal[i] < squared_errors_anomal[j][i]:
    #            errors_anomal[i] = squared_errors_anomal[j][i]

    # average squared error
    #print("Average Squared Error")
    #for j in squared_errors_normal.keys():
    #    for i in range(len(squared_errors_normal[j])):
    #        errors_normal[i] += squared_errors_normal[j][i]
    #errors_normal[:] = [x / len(errors_normal) for x in errors_normal]
    #for j in squared_errors_normal_all.keys():
    #    for i in range(len(squared_errors_normal_all[j])):
    #        errors_normal_all[i] += squared_errors_normal_all[j][i]
    #errors_normal_all[:] = [
    #        x / len(errors_normal_all) for x in errors_normal_all]
    #for j in squared_errors_anomal.keys():
    #    for i in range(len(squared_errors_anomal[j])):
    #        errors_anomal[i] += squared_errors_anomal[j][i]
    #errors_anomal[:] = [x / len(errors_anomal) for x in errors_anomal]

    # sum squared error
    #print("Sum Squared Error")
    #for j in squared_errors_normal.keys():
    #    for i in range(len(squared_errors_normal[j])):
    #        errors_normal[i] += squared_errors_normal[j][i]
    #for j in squared_errors_normal_all.keys():
    #    for i in range(len(squared_errors_normal_all[j])):
    #        errors_normal_all[i] += squared_errors_normal_all[j][i]
    #for j in squared_errors_anomal.keys():
    #    for i in range(len(squared_errors_anomal[j])):
    #        errors_anomal[i] += squared_errors_anomal[j][i]

    n_perc = 95
    error_threshold = np.percentile(np.asarray(errors_normal_all), n_perc)
    print("Percentile: %s (threshold: %s)" % (n_perc, error_threshold))

    #fig = plt.figure()
    #plt.hist(errors_normal, bins=80)
    #plt.axvline(x=error_threshold,c='r')
    #plt.ylabel('# data points')
    #plt.xlabel('Error')
    #plt.title(node + ' - Normal Behaviour')

    #fig = plt.figure()
    #plt.hist(errors_anomal, bins=50)
    #plt.axvline(x=error_threshold,c='r')
    #plt.ylabel('# data points')
    #plt.xlabel('Error')
    #plt.title(node + ' - Anomalies')

    fig = plt.figure()
    plt.hist(errors_normal, bins=80, color='b', label='Normal')
    plt.hist(errors_anomal, bins=50, color='r', label='Anomaly')
    plt.ylabel('# data points')
    plt.xlabel('Error')
    plt.legend()
    plt.show()
    sys.exit()

    classes_normal = [0] * nn_samples
    classes_anomal = [1] * na_samples
    errors = errors_normal + errors_anomal
    classes = classes_normal + classes_anomal

    precision_N, recall_N, fscore_N = fscore(
            errors, error_threshold, classes, 0)
    precision_A, recall_A, fscore_A = fscore(
            errors, error_threshold, classes, 1)

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

    print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
            recall_N, fscore_N))
    print("Anomaly: precision %f, recall %f, F-score %f" % (precision_A, 
            recall_A, fscore_A))

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
    #for n_perc in range(97, 98):
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

        #print("\tNormal: precision %f, recall %f, F-score %f" % (precision_N, 
        #        recall_N, fscore_N))
        #print("\tAnomaly: precision %f, recall %f, F-score %f" %(precision_A, 
        #        recall_A, fscore_A))
        #print("\tAll Classes Weighted: precision %f, recall %f, F-score %f" % (
        #        precision_W, recall_W, fscore_W))
        #print("\ttn {}, fp {}, fn {}, tp {}".format(tn, fp, fn, tp))
        
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
        plt.savefig("{}_nodeSpecificModel_accuracy_VS_nperc.png".format(node))

        #plt.show()

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
Given an input set for an AE and its output set (supposedly with same size),
compute the abs error for each example; the max error over all features
- also return a list saying if the error is greater than threshold
- if 'skip_idle' == True we don't raise alarms for idle periods
'''
def get_max_abs_recon_error_AE(inset, outset, error_threshold, active_idle,
        skip_idle=True):

    inset = inset.values

    n_samples, n_series = inset.shape
    max_errors = [0] * n_samples
    abs_errors = {}

    for j in range(n_series):
        abs_errors[j] = []
    for i in range(n_samples):
        for j in range(n_series):
            abs_errors[j].append(abs(outset[i][j] - inset[i][j]))
 
    # max abs error 
    for j in abs_errors.keys():
        for i in range(len(abs_errors[j])):
            if max_errors[i] < abs_errors[j][i]:
                max_errors[i] = abs_errors[j][i]

    alarms_raised = [0] * len(max_errors)
    for i in range(len(max_errors)):
        if max_errors[i] > error_threshold:
            if skip_idle and active_idle[i] == 0:
                alarms_raised[i] = 0
            else:
                alarms_raised[i] = 1

    return max_errors, alarms_raised

'''
Given an input set for an AE and its output set (supposedly with same size),
compute the abs error for each example; the max error over all features
- also return a list saying if the error is greater than threshold
- if 'skip_idle' == True we don't raise alarms for idle periods
'''
def get_max_abs_recon_error_AE_2(inset, outset, error_threshold, active_idle,
        skip_idle=True):
    n_samples, n_series = inset.shape
    max_errors = [0] * n_samples
    abs_errors = {}

    for j in range(n_series):
        abs_errors[j] = []
    for i in range(n_samples):
        for j in range(n_series):
            abs_errors[j].append(abs(outset[i][j] - inset[i][j]))
 
    # max abs error 
    for j in abs_errors.keys():
        for i in range(len(abs_errors[j])):
            if max_errors[i] < abs_errors[j][i]:
                max_errors[i] = abs_errors[j][i]

    alarms_raised = [0] * len(max_errors)
    for i in range(len(max_errors)):
        if max_errors[i] > error_threshold:
            #if skip_idle and active_idle[i] == 0:
            if active_idle[i] == 0:
                alarms_raised[i] = 0
            else:
                #continue
                alarms_raised[i] = 1

    return max_errors, alarms_raised


'''
To be used only with VAE
    - plot latent variable layer
    - different color for healthy data points and anomalies
'''
def plot_latent_var(z, n_z, idx_normal_anomalies, plotname='', epoch='',
        show=False):
    if n_z != 3 and n_z != 2:
        return
    fig = plt.figure()
    side = 2

    if n_z == 2:     # 2d plot
        plt.scatter(z[:, 0], z[:, 1], c=idx_normal_anomalies, alpha=0.5)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
    else:            # 3d plot
        ax = fig.add_subplot(111, projection='3d') 
        p = ax.scatter(z[:, 0], z[:, 1], z[:, 2], 
                c=idx_normal_anomalies)
        fig.colorbar(p)
        ax.set_xlabel('z[0]')
        ax.set_ylabel("z[1]")
        ax.set_zlabel("z[2]")

    if epoch != '':
        plt.title('Epoch: %s' % epoch)

    if show:
        plt.show()
    else:
        plt.savefig(plotname)

    plt.close(fig)

'''
To be used only with VAE
    - plot latent variable layer
    - different color for healthy data points and anomalies
    - also plot the predicted clusters (one marker per cluster)
'''
def plot_latent_var_withCluster(z, n_z, idx_normal_anomalies, plotname='',
        centroids=[], pred_labels=[], plot=True):
    if n_z != 3 and n_z != 2:
        return
   
    if n_z ==2 and len(pred_labels) == len(z):
        fig = plt.figure()
        z_0 = [[] for i in range(len(set(pred_labels)))]
        z_1 = [[] for i in range(len(set(pred_labels)))]
        col = [[] for i in range(len(set(pred_labels)))]
        for i in range(len(pred_labels)):
            z_0[pred_labels[i]].append(z[:, 0][i])
            z_1[pred_labels[i]].append(z[:, 1][i])
            if idx_normal_anomalies[i] == 0:
                col[pred_labels[i]].append('r')
            elif idx_normal_anomalies[i] == 1:
                col[pred_labels[i]].append('b')
            elif idx_normal_anomalies[i] == 2:
                col[pred_labels[i]].append('c')
        m = ['^','x','o']

        for i in range(len(z_0)):
            plt.scatter(z_0[i], z_1[i], c=col[i], marker=m[i], alpha=0.5)

        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

        if len(centroids) > 0:
            plt.scatter(centroids[:,0], centroids[:,1], marker='+', 
                    color='k', s=500)

        if(plot):
            plt.show()
        plt.savefig(plotname)
        plt.close(fig)
    
'''
Try to obtain clusters from latent variable (VAE) layer
'''
def cluster_latent_var(n_cluster, latent_repr, labels_true):

    ###########################################################################
    # KMeans
    #kmeans = KMeans(n_clusters=n_cluster)
    #kmeans.fit(latent_repr)
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    #labels = kmeans.predict(latent_repr)
    ###########################################################################

    ###########################################################################
    # DBSCAN
    db = DBSCAN(eps=0.2, min_samples=20).fit(latent_repr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    centroids = []
    ###########################################################################

    ###########################################################################
    # Spectral Clustering
    #spectral = SpectralClustering(
    #        n_clusters=n_cluster, eigen_solver='arpack',
    #        affinity="nearest_neighbors")
    #spectral.fit(latent_repr)
    #labels = spectral.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    #centroids = []
    ###########################################################################

    ###########################################################################
    # Gaussian Mixture
    #gmm = mixture.GaussianMixture(
    #        n_components=n_cluster, covariance_type='full')
    #gmm.fit(latent_repr)
    #centroids = []
    #labels = gmm.predict(latent_repr)
    #n_clusters_ = n_cluster
    ############################################################################

    # compute & print results
    hs = homogeneity_score(labels_true, labels)
    cs = completeness_score(labels_true, labels)
    vms = v_measure_score(labels_true, labels)
    ars = adjusted_rand_score(labels_true, labels)
    amis = adjusted_mutual_info_score(labels_true, labels) 
    if n_clusters_ > 1:
        ss = silhouette_score(latent_repr, labels)
    else:
        ss = -1000
    avg_s = (hs + cs + vms + ars + amis + ss) / float(6)
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels)
    purity_score = np.sum(np.amax(contingency_matrix, axis=0
        )) / np.sum(contingency_matrix) 
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % hs)
    print("Completeness: %0.3f" % cs)
    print("V-measure: %0.3f" % vms)
    print("Adjusted Rand Index: %0.3f" % ars)
    print("Adjusted Mutual Information: %0.3f" % amis)
    print("Silhouette Coefficient: %0.3f" % ss)
    print("Purity Score: %0.3f" % purity_score)
    print("\tAverage of all scores (mmmhh..): %0.3f" % avg_s)
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    
    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
            labels_true, labels, labels=[0])
    precision_A1, recall_A1, fscore_A1, xyz = precision_recall_fscore_support(
            labels_true, labels, labels=[1])
    precision_A2, recall_A2, fscore_A2, xyz = precision_recall_fscore_support(
            labels_true, labels, labels=[2])
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            labels_true, labels, average='weighted')

    print("Cluster evaluation: Multi-classes")
    print("Normal: precision %f, recall %f, F-score %f" % (precision_N, 
            recall_N, fscore_N))
    print("Anomaly 1: precision %f, recall %f, F-score %f" % (precision_A1, 
            recall_A1, fscore_A1))
    print("Anomaly 2: precision %f, recall %f, F-score %f" % (precision_A2, 
            recall_A2, fscore_A2))
    print("All Classes Weighted: precision %f, recall %f, F-score %f" % (
            precision_W, recall_W, fscore_W))

    return centroids, labels

'''
To be used only with VAE
    - plot latent variable layer
    - different color for healthy data points and anomalies
    - different shape for different jobs (currently it only QE, CPU/MEM stress 
    and others are considered
'''
def plot_latent_var2(z, n_z, idx_normal_anomalies, dataPoint_jobs):
    if n_z != 3 and n_z != 2:
        return

    z_0 = [[] for i in range(len(set(dataPoint_jobs)))]
    z_1 = [[] for i in range(len(set(dataPoint_jobs)))]
    z_2 = [[] for i in range(len(set(dataPoint_jobs)))]
    c = [[] for i in range(len(set(dataPoint_jobs)))]
    for i in range(len(dataPoint_jobs)):
        z_0[dataPoint_jobs[i]].append(z[:, 0][i])
        z_1[dataPoint_jobs[i]].append(z[:, 1][i])
        if n_z == 3:
            z_2[dataPoint_jobs[i]].append(z[:, 2][i])
        c[dataPoint_jobs[i]].append(idx_normal_anomalies[i])
    m = ['^','x','o']

    fig = plt.figure()
    if n_z == 2:     # 2d plot
        for i in range(len(z_0)):
            plt.scatter(z_0[i], z_1[i], c=c[i], marker=m[i])
        #plt.scatter(z[:, 0], z[:, 1], c=idx_normal_anomalies, 
        #        marker=dataPoint_jobs)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
    else:            # 3d plot
        ax = fig.add_subplot(111, projection='3d') 
        for i in range(len(z_0)):
            p = ax.scatter(z_0[i], z_1[i], z_2[i], 
                    c=c[i], marker=m[i])
        #p = ax.scatter(z[:, 0], z[:, 1], z[:, 2], 
        #        c=idx_normal_anomalies, marker=dataPoint_jobs)
        fig.colorbar(p)
        ax.set_xlabel('z[0]')
        ax.set_ylabel("z[1]")
        ax.set_zlabel("z[2]")

    plt.show()
    return fig

'''
Find the job active in the input data point
'''
def mark_dataPoints_jobs(start_time, end_time, node, timestamps):
    dataPoint_jobs = []
    jobs = retrieve_jobs_info_node(start_time, end_time, node)
    for t in timestamps:
        found = False
        for jid, j in jobs.iteritems():
            if j['start_time'] <= t < j['end_time']:
                found = True
                if 'qe' in j['job_name']:
                    dataPoint_jobs.append(0)
                elif 'CPU_stress' in j['job_name']:
                    dataPoint_jobs.append(1)
                #elif 'MEM_stress' in j['job_name']:
                #    dataPoint_jobs.append(2)
                else:
                    dataPoint_jobs.append(2)
                break
        if not found:
            dataPoint_jobs.append(2)
    return dataPoint_jobs

def plot_cluster_bars(z_mean, _n_z, idx_normal_anomalies, pred_labels):
    n_clusters = set(pred_labels)
    n_anomalies = set(idx_normal_anomalies)
    n_point_in_cluster = {}
    n_actual_anoms_in_cluster = {}
    for l in pred_labels:
        if l in n_point_in_cluster:
            n_point_in_cluster[l] += 1
        else:
            n_point_in_cluster[l] = 1
    n_points_list = []
    x_pos = []
    n = 0
    for k in n_point_in_cluster.keys():
        n_actual_anoms_in_cluster[k] = {}
        n_points_list.append(n_point_in_cluster[k])
        x_pos.append(n)
        n += 1
    for i in range(len(idx_normal_anomalies)):
        cluster = pred_labels[i]
        if idx_normal_anomalies[i] in n_actual_anoms_in_cluster[cluster]:
            n_actual_anoms_in_cluster[cluster][idx_normal_anomalies[i]] += 1
        else:
            n_actual_anoms_in_cluster[cluster][idx_normal_anomalies[i]] = 1

    n_anom_lists = {}
    for a in list(set(idx_normal_anomalies)):
        for k in n_actual_anoms_in_cluster.keys():
            if a in n_actual_anoms_in_cluster[k]:
                n = n_actual_anoms_in_cluster[k][a]
            else:
                n = 0
            if a in n_anom_lists:
                n_anom_lists[a].append(n)
            else:
                n_anom_lists[a] = [n]

    plt.figure()
    x_pos = np.arange(len(n_point_in_cluster))
    plt.bar(x_pos, n_points_list, align='center')
    for k, v in n_anom_lists.items():
        if k != 0:
            plt.bar(x_pos, v, align='center')
    plt.xticks(x_pos, list(n_point_in_cluster.keys()))
    plt.ylabel('# Points')
    plt.xlabel('Cluster')
    plt.show()

'''
Analyse features importance
    - Random Forest
'''
def eval_feature_importance_RF(model, X, feature_names):
    #print("Features importance: %s " % model.feature_importances_)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
            axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        if importances[indices[f]] > 0:
            print("%d. feature %d/%s (%f)" % (f + 1, indices[f], 
                feature_names[f], importances[indices[f]]))
    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(X.shape[1]), importances[indices],
    #               color="r", yerr=std[indices], align="center")
    #plt.xticks(range(X.shape[1]), indices)
    #plt.xlim([-1, X.shape[1]])
    #plt.show()

'''
Analyse features importance
    - Decision Tree
'''
def eval_feature_importance_DT(model, X, feature_names):
    #print("Features importance: %s " % model.feature_importances_)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        if importances[indices[f]] > 0:
            print("%d. feature %d/%s (%f)" % (f + 1, indices[f], 
                feature_names[f], importances[indices[f]]))
    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(X.shape[1]), importances[indices],
    #               color="r", align="center")
    #plt.xticks(range(X.shape[1]), indices)
    #plt.xlim([-1, X.shape[1]])
    #plt.show()

'''
Plot any data in 2D, using manifold learning
'''
def plot_2d_proj(x, classes):
    n_neighbors = 100
    n_comp = 2
    x_se = SpectralEmbedding(n_components=n_comp).fit_transform(x)
    x_im = Isomap(n_neighbors, n_comp).fit_transform(x)
    x_mds = MDS(n_comp, max_iter=100, n_init=1).fit_transform(x)
    x_tsne = TSNE(n_components=n_comp, init='pca', random_state=0
            ).fit_transform(x)
    x_lle = LocallyLinearEmbedding(n_neighbors, n_comp, method='modified'
            ).fit_transform(x)

    plt.figure()
    plt.scatter(*zip(*x_se), c=classes)
    plt.title('Spectral Embedding')

    plt.figure()
    plt.scatter(*zip(*x_im), c=classes)
    plt.title('Isomap')

    plt.figure()
    plt.scatter(*zip(*x_mds), c=classes)
    plt.title('MDS')

    plt.figure()
    plt.scatter(*zip(*x_tsne), c=classes)
    plt.title('TSNE')

    plt.figure()
    plt.scatter(*zip(*x_lle), c=classes)
    plt.title('Locally Linear Embedding')

    plt.show()

'''
Plot error distributions of multiple nodes
'''
def plot_error_distribution_singleNode(node, actual_N, pred_N, 
        actual_A, pred_A, actual_N_all, pred_N_all):

    fig = plt.figure()
    color_idx = 0
    print('Single Node: Computing error distribution, node {}'.format(node))
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
    plot_dir = '/home/b0rgh/various_plots_logs/davide/plots/anomaly_detection/'
    plot_dir += 'split_train_test/rndmSplit/multiple_anomalies/'
    plot_dir += 'error_distribution/eng_apps_AI_paper'
    plot_name = '{}_normalVSanormal_errorDistrib_{}.png'.format(plot_dir, node)
    plt.savefig(plot_name)


'''
Plot error distributions of multiple nodes
'''
def plot_error_distribution_multiNode(nodes, actual_N, pred_N, 
        actual_A, pred_A, actual_N_all, pred_N_all):

    fig = plt.figure()
    color_idx = 0
    #colors_N_more = pl.cm.Blues(np.linspace(0,1,len(nodes)+2))
    #colors_A_more = pl.cm.Reds(np.linspace(0,1,len(nodes)+2))
    colors_N_more = pl.cm.autumn(np.linspace(0,1,len(nodes)+2))
    colors_A_more = pl.cm.winter(np.linspace(0,1,len(nodes)+2))
    #colors_N_more = pl.cm.cool(np.linspace(0,1,len(nodes)+2))
    #colors_A_more = pl.cm.hot(np.linspace(0,1,len(nodes)+2))
    #colors_N_more = pl.cm.summer(np.linspace(0,1,len(nodes)+2))
    #colors_A_more = pl.cm.copper(np.linspace(0,1,len(nodes)+2))

    colors_N = colors_N_more[1:-1]
    colors_A = colors_A_more[1:-1]
    for node in nodes:
        print('Computing error distribution, node {}'.format(node))
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

        plt.hist(errors_normal, bins=80, color=colors_N[color_idx],
                label='Node {} - Normal'.format(node), alpha=.8)

        plt.hist(errors_anomal, bins=50, color=colors_A[color_idx], 
                label='Node {} - Anomaly'.format(node), alpha=.8)
        color_idx += 1

    plt.ylabel('# data points')
    plt.xlabel('Error')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(prop={'size': 10})
    plt.show()

'''
Plot alarms triggered by AE
- multiple nodes
'''
def plot_alarms_multi_node(errors, alarms_raised, timestamps, freqGov):
    fig = plt.figure()
    cm = plt.cm.get_cmap('bwr')
    node_idx = 0
    #for node in alarms_raised.keys():
    #    z = alarms_raised[node]
    #    y = [node_idx] * len(z)

    #    m = ['o', 'x']
    #    #x = range(len(z))
    #    #x = [pd.to_datetime(d) for d in timestamps[1:]]
    #    #x = timestamps[1:]
    #    #x = timestamps[node]
    #    x = [pd.to_datetime(d) for d in timestamps[node]]
    #    #plt.scatter(x, y, c=z, vmin=0, vmax=1, s=10, cmap=cm, 
    #    #        marker='s', label=node)
    #    print(node)
    #    print(len(x))
    #    print(len(y))
    #    plt.scatter(x, y, c=z, s=10, cmap=cm, marker=m, label=node)

    #    node_idx += 1

    ax = fig.add_subplot(111)
    for node in alarms_raised.keys():
        z = alarms_raised[node]
        y = [node_idx] * len(z)

        xA = []
        xN = []
        for i in range(len(alarms_raised[node])):
            if alarms_raised[node][i] == 1:
                xA.append(pd.to_datetime(timestamps[node][i]))
            else:
                xN.append(pd.to_datetime(timestamps[node][i]))
        yN = [node_idx] * len(xN)
        yA = [node_idx] * len(xA) 
        xs = [xN, xA]
        ys = [yN, yA]
        cs = ['b', 'r']
        ms = ['o', 'x']
        ss = [10, 10]
        for i in range(len(xs)):
            ax.scatter(xs[i], ys[i], c=cs[i], marker=ms[i], label=node, s=ss[i])
        margin = datetime.timedelta(seconds=60)
        ax.text(xs[0][-1] + margin, node_idx - 0.05, node, fontsize=10)
        node_idx += 1

    add_freq_govs_to_plot2(freqGov, ax, 0, node_idx)

    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
 
    plt.ylabel('Nodes')
    plt.xlabel('Time (Idx)')
    plt.show()



