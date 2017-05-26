# Copyright (c) 2017 Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


import os

import csv
import numpy as np
import scipy.io as sio

from sklearn.covariance import GraphLassoCV
import nilearn
from nilearn import connectome


# Output path
save_path = '/vol/dhcp-hcp-data/ABIDE'

# Number of subjects
num_subjects = 1000

# Selected pipeline
pipeline = 'cpac'

# Files to fetch
derivatives = ['rois_ho']

# Get the root folder
root_folder = os.path.join(save_path, 'ABIDE_pcp/cpac/filt_noglobal')


def get_ids(num_subjects=None, short=True):
    """
        num_subjects   : number of subject IDs to get
        short          : True of False, specifies whether to get short or long subject IDs

    return:
        subject_IDs    : list of subject IDs (length num_subjects)
    """

    if short:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'subject_IDs.txt'), dtype=int)
        subject_IDs = subject_IDs.astype(str)
    else:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'full_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


def fetch_filenames(subject_list, file_type):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_aal': '_rois_aal.1D',
                   'rois_cc200': '_rois_cc200.1D',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Load subject ID lists
    subject_IDs = get_ids(short=True)
    subject_IDs = subject_IDs.tolist()
    full_IDs = get_ids(short=False)

    # Fill list with requested file paths
    for s in subject_list:
        try:
            if file_type in filemapping:
                idx = subject_IDs.index(s)
                pattern = full_IDs[idx] + filemapping[file_type]
            else:
                pattern = s + file_type

            filenames.append(os.path.join(root_folder, s, pattern))
        except ValueError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


def fetch_subject_files(subjectID):
    """
        subjectID : short subject ID for which list of available files are fetched

    returns:

        onlyfiles : list of absolute paths for available subject files
    """

    # Load subject ID lists
    subject_IDs = get_ids(short=True)
    subject_IDs = subject_IDs.tolist()
    full_IDs = get_ids(short=False)

    try:
        idx = subject_IDs.index(subjectID)
        subject_folder = os.path.join(root_folder, subjectID)
        onlyfiles = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder)
                     if os.path.isfile(os.path.join(subject_folder, f))]
    except ValueError:
        onlyfiles = []

    return onlyfiles


def fetch_conn_matrices(subject_list, atlas_name, kind):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
        kind         : the kind of correlation used to estimate the matrices, i.e.

    returns:
        connectivity : list of square connectivity matrices, one for each subject in subject_list
    """

    conn_files = fetch_filenames(subject_list,
                                 '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')

    conn_matrices = []

    for fl in conn_files:
        print("Reading connectivity file %s" % fl)
        try:
            mat = sio.loadmat(fl)['connectivity']
            conn_matrices.append(mat)
        except IOError:
            print("File %s does not exist" % fl)

    return conn_matrices


def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        ts           : list of timeseries arrays, each of shape (timepoints x regions)
    """

    ts_files = fetch_filenames(subject_list, 'rois_' + atlas_name)

    ts = []

    for fl in ts_files:
        print("Reading timeseries file %s" % fl)
        ts.append(np.loadtxt(fl, skiprows=0))

    return ts


def norm_timeseries(ts_list):
    """
        ts_list    : list of timeseries arrays, each of shape (timepoints x regions)

    returns:
        norm_ts    : list of normalised timeseries arrays, same shape as ts_list
    """

    norm_ts = []

    for ts in ts_list:
        norm_ts.append(nilearn.signal.clean(ts, detrend=False))

    return norm_ts


def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=root_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject short ID
        atlas_name   : name of the atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind == 'lasso':
        # Graph Lasso estimator
        covariance_estimator = GraphLassoCV(verbose=1)
        covariance_estimator.fit(timeseries)
        connectivity = covariance_estimator.covariance_
        print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


def group_connectivity(timeseries, subject_list, atlas_name, kind, save=True, save_path=root_folder):
    """
        timeseries   : list of timeseries tables for subjects (timepoints x regions)
        subject_list : the subject short IDs list
        atlas_name   : name of the atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind == 'lasso':
        # Graph Lasso estimator
        covariance_estimator = GraphLassoCV(verbose=1)
        connectivity_matrices = []

        for i, ts in enumerate(timeseries):
            covariance_estimator.fit(ts)
            connectivity = covariance_estimator.covariance_
            connectivity_matrices.append(connectivity)
            print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity_matrices = conn_measure.fit_transform(timeseries)

    if save:
        for i, subject in enumerate(subject_list):
            subject_file = os.path.join(save_path, subject_list[i],
                                        subject_list[i] + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity_matrices[i]})
            print("Saving connectivity matrix to %s" % subject_file)

    return connectivity_matrices


def get_subject_label(subject_list, label_name):
    """
        subject_list : the subject short IDs list
        label_name   : name of the label to be retrieved

    returns:
        label        : dictionary of subject labels
    """

    label = {}

    with open(os.path.join(save_path, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['subject'] in subject_list:
                label[row['subject']] = row[label_name]

    return label


def load_all_networks(subject_list, kind, atlas_name="aal"):
    """
        subject_list : the subject short IDs list
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the atlas used

    returns:
        all_networks : list of connectivity matrices (regions x regions)
    """

    all_networks = []

    for subject in subject_list:
        fl = os.path.join(root_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)['connectivity']

        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)

        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    return all_networks


def get_net_vectors(subject_list, kind, atlas_name="aal"):
    """
        subject_list : the subject short IDs list
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the atlas used

    returns:
        matrix       : matrix of connectivity vectors (num_subjects x num_connections)
    """

    # This is an alternative implementation
    networks = load_all_networks(subject_list, kind, atlas_name=atlas_name)
    # Get Fisher transformed matrices
    norm_networks = [np.arctanh(mat) for mat in networks]
    # Get upper diagonal indices
    idx = np.triu_indices_from(norm_networks[0], 1)
    # Get vectorised matrices
    vec_networks = [mat[idx] for mat in norm_networks]
    # Each subject should be a row of the matrix
    matrix = np.vstack(vec_networks)

    return matrix


def get_atlas_coords(atlas_name='ho'):
    """
        atlas_name   : name of the atlas used

    returns:
        matrix       : matrix of roi 3D coordinates in MNI space (num_rois x 3)
    """

    coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
    coords = np.loadtxt(coords_file, delimiter=',')

    if atlas_name == 'ho':
        coords = np.delete(coords, 82, axis=0)

    return coords