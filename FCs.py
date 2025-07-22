import os
import csv
import numpy as np
import scipy.io as sio
import sklearn
from sklearn.covariance import GraphicalLassoCV
import nilearn
from nilearn import connectome

# Output path
save_path = '/home/celery/Documents/Research/dataset/Outputs'

# Number of subjects
num_subjects = 1000

# Selected pipeline
pipeline = 'cpac'

# Files to fetch
derivatives = ['rois_ho']

# Get the root folder
root_folder = '/home/celery/Documents/Research/dataset/Outputs/cpac/filt_global/rois_ho'

def get_ids(num_subjects=None, short=True):
    """
        num_subjects   : number of subject IDs to get
        short          : True of False, specifies whether to get short or long subject IDs (Eg: 51431 or NYU_0051431_session_1_rest_1)

    return:
        subject_IDs    : list of subject IDs (length num_subjects)
    """

    if short:
        subject_IDs = np.loadtxt('/home/celery/Documents/Research/dataset/valid_subject_ids.txt', dtype=int)
        subject_IDs = subject_IDs.astype(str)
    else:
        subject_IDs = np.loadtxt('/home/celery/Documents/Research/dataset/full_subject_ids.txt', dtype=str)

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


            filenames.append(os.path.join(root_folder, pattern))
        except ValueError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames

def find_subject_file(subject_id, folder_path, suffix='_rois_ho.1D'):
    """
    Search for a file in folder_path that matches the given subject_id.

    subject_id   : short subject ID like '50004'
    folder_path  : the path where .1D files are stored (e.g., rois_ho folder)
    suffix       : file suffix to look for (default is '_rois_ho.1D')

    Returns:
        file_path  : full path to the matching file, or 'N/A' if not found
    """
    for fname in os.listdir(folder_path):
        if subject_id in fname and fname.endswith(suffix):
            return os.path.join(folder_path, fname)
    return 'N/A'

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
        covariance_estimator = GraphicalLassoCV(verbose=1)
        covariance_estimator.fit(timeseries)
        connectivity = covariance_estimator.covariance_
        print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    # defining custom save path
    save_path_mat = '/home/celery/Documents/Research/dataset/Outputs/cpac/filt_global/mat/'

    if save:
        subject_file = os.path.join(save_path_mat,
                            subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')

        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity

def group_connectivity(timeseries, subject_list, atlas_name, kind, save=True, save_path=root_folder): #batch version of the function above
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
        covariance_estimator = GraphicalLassoCV(verbose=1)
        connectivity_matrices = []

        for i, ts in enumerate(timeseries):
            covariance_estimator.fit(ts)
            connectivity = covariance_estimator.covariance_
            connectivity_matrices.append(connectivity)
            print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity_matrices = conn_measure.fit_transform(timeseries)
    # defining custom save path for .mat files
    save_path_mat = '/home/celery/Documents/Research/dataset/Outputs/cpac/filt_global/mat/'

    if save:
        for i, subject in enumerate(subject_list):
            subject_file = os.path.join(save_path_mat,
                                        subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity_matrices[i]})
            print("Saving connectivity matrix to %s" % subject_file)

    return connectivity_matrices

# Getting the file using group connectivity
subject_file = '/home/celery/Documents/Research/dataset/valid_subject_ids.txt'
subject_list = np.loadtxt(subject_file, dtype = str).tolist()

ts_list = get_timeseries(subject_list, 'ho')
norm_ts_list = norm_timeseries(ts_list)

group_connectivity(
    timeseries=ts_list,
    subject_list=subject_list,
    atlas_name='ho',
    kind='correlation',
    save=True,
    save_path= '/home/celery/Documents/Research/dataset/Outputs/cpac/filt_global/mat'
)
