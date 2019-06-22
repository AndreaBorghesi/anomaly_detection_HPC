# Semi-supervised Autoencoder-based anomaly detection on HPC Systems
This repository contains the set of script capable to replicate parts of the
work described in:
1) "Anomaly Detection using Autoencoders in High Performance Computing Systems",
Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca
Benini, IAAI19 (proceedings in process) -- https://arxiv.org/abs/1902.08447
2) "Online Anomaly Detection in HPC Systems", Andrea Borghesi, Antonio Libri,
Luca Benini, Andrea Bartolini, AICAS19 (proceedings in process) --
https://arxiv.org/abs/1811.05269

Fine-grained data was collected on the D.A.V.I.D.E. HPC system (developed in
joint collaboration by CINECA & E4 & University of Bologna, Bologna, Italy); the
collected data refers to the period March-May 2018.
The data was collected using the EXAMON framework and consists in a set of
measurements coming from a variety of sources, ranging from physical sensors
(such as a temperatures, voltages, power, etc.) to performance counters (IPS,
CPI, core load, etc).
For additional details on the supercomputer see Ahmad et al. "Design of an
energy aware petaflops class high performance cluster based on power
architecture", IPDPSW 2017. 
For a detailed description of the collected data see Bartolini et al. "The
DAVIDE big-data-powered fine-grain power and performance monitoring support",
International Conference on Computing Frontiers, 2018.

The data set has been used to train a autoencoder-based model to automatically
detect anomalies in a semi-supervised fashion, on a real HPC system.

## Building
Requires python > 3.6
Python modules required:
* Tensorflow 1.x
    - instructions at https://www.tensorflow.org/install
    - e.g. pip install tensorflow (CPU only)
* keras 
    - instructions at https://keras.io/
* numpy
* scikit-learn
* pandas

## Usage
* Get source code (git clone/download) 
    - destination folder: <anomaly_detection_dir>
    - cd <anomaly_detection_dir>

* Download the training and the validation set. The data sets correspond on the
  measurements collected on a subset of computing nodes of D.A.V.I.D.E.
supercomputer (period 2018/03/03-2018/05/31). 
    - The data can be found here: https://zenodo.org/record/3251873
    - the nodes available are the following (the following node identifiers are
      those recognized by the anomaly detection script); in the following list,
the node identifiers are separated by commas
    - davide16, davide17, davide18, davide19, davide26, davide27, davide28,
      davide29, davide30, davide31, davide32, davide33, davide34, davide42,
davide45

* Copy HPC data to correct folder
    - cp <data_download_dir>*.pickle <anomaly_detection_dir>/data

* Run script and perform anomaly detection on a particular node (among those
  available -- check in the downloaded data)
    - python3 detect_anomalies <node> <mode>
    - <node> specifies the node to perform the anomaly detection
    - <mode> specifies the approach to be used; allowed values: 
        - 0 -- autoencoder-based semi-supervised approach
    - e.g. python3 detect_anomalies davide18 0



