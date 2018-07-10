import time

import h5py
import numpy as np
from sklearn.cluster import DBSCAN


DATA = 'data'
LABELS = 'labels'

MODE = 'mode'
SEED = 'seed'
REMOVED_LABELS = 'removed_labels'
EPSILON = 'epsilon'
MIN_POINTS = 'min_points'
TRAINING_FRACTION = 'training_fraction'
SELECTION = 'selection'
MEAN = 'mean'

REMOVED = 0
TEST = 1
TRAIN = 2

RANDOM = 'random'
SIZE = 'size'
STDDEV = 'stddev'


def generate_patches(in_file, out_file, mode, remove, train_fraction, seed, epsilon, min_points):
    """
    Sample the data from the source file and divide them into training and test data through a selection matrix
    Result is stored in the given output HDF5 file.

    :param in_file: 	   str containing the filename of the basic .hdf5 file
    :param out_file:       str containing the path where the hdf5-file will be saved
    :param mode:           str, deciding how to split training and test data, can be 'random', 'size' or 'stddev'
    :param remove:         list(int), list of class labels to remove from the ground-truth (e.g. background)
    :param train_fraction: positive float determining the fraction of the data used for training
    :param seed:           positive, int random seed for reproducibility of the random sampling
    :param epsilon:        positive, float, cluster sampling search radius
    :param min_points:     positive, int, cluster sampling density threshold
    """
    # read data
    start = time.time()
    print('\t\tReading data source... ', end='', flush=True)
    data, labels = read_datasets(in_file, DATA, LABELS)
    print('\t\t\t[Done, took: {:.2f}s]'.format(time.time() - start))

    # create train/test selection mask
    start = time.time()
    print('\t\tCreating selection mask... ', end='', flush=True)
    if mode == RANDOM:
        selection = create_random_selection(labels, remove, train_fraction, seed)
    else:
        selection = create_cluster_selection(mode, data, labels, remove, train_fraction, epsilon, min_points)
    print('\t\t[Done, took: {:.2f}s]'.format(time.time() - start))

    # determine normalization coefficients
    start = time.time()
    print('\t\tCalculate normalization coefficients...', end='', flush=True)
    mean = data[selection == TRAIN].reshape(-1, data.shape[-1]).mean(axis=0).reshape(1, 1, -1)
    stddev = data[selection == TRAIN].reshape(-1, data.shape[-1]).std(axis=0).reshape(1, 1, -1)
    print('\t[Done, took: {:.2f}s]'.format(time.time() - start))

    # store the selected data
    start = time.time()
    print('\t\tSaving data file ... ', end='', flush=True)
    safe_patches(out_file, mode, remove, train_fraction, seed, epsilon, min_points, selection, mean, stddev)
    print('\t\t\t[Done, took: {:.2f}s]'.format(time.time() - start))


def read_datasets(in_file, *args):
    """
    Load a list of datasets from an HDF5 file into memory.

    :param   in_file: str of the source HDF5 file
    :return: dataset: tuple(np.array) of memory-mapped datasets
    """
    with h5py.File(in_file) as handle:
        datasets = [np.array(handle[name]) for name in args]

    return tuple(datasets)


def create_random_selection(labels, remove, train_fraction, seed):
    """
    Create a random boolean selection matrix for the division of the data into test and train set based on a given
    fraction. The algorithms maintains the initial label distribution.

    :param labels:         np.array(width, height), data labels
    :param remove:         list(int), list of class labels to remove from the ground-truth (e.g. background)
    :param train_fraction: positive number in [0,1], implying the fraction of data points being in the training set
    :param seed:           the seed used for the random number generator
    :return: selection:	   np.array(width, height), selection matrix of the elements
    """
    # set the random seed
    np.random.seed(seed)

    # initialize helper variables
    flat_labels = labels.flatten()
    distinct_labels = np.unique(flat_labels)
    selection = np.full(flat_labels.shape, TEST, dtype=np.int8)

    # sample the labels by shuffling and index cut-off using label distribution
    for label in distinct_labels:
        indices = np.where(flat_labels == label)[0]

        # check whether we have to remove the respective label
        if label in remove:
            selection[indices] = REMOVED
            continue

        # shuffle the indices and select the train_fraction elements for training
        np.random.shuffle(indices)
        train_indices = indices[:int(np.round(train_fraction * indices.shape[0]))]
        selection[train_indices] = TRAIN

    return selection.reshape(labels.shape)


def create_cluster_selection(mode, data, labels, remove, train_fraction, epsilon, min_points):
    """
    Create a boolean selection mask (using cluster sampling) for the division of the data into training and test data.

    :param mode:           str, indicating the cluster selection mode, may be 'size' or 'stddev'
    :param data: 	       np.array(width, height, channels), raw hyper-spectral images
    :param labels:         np.array(width, height), data labels
    :param remove:         list(int), list of class labels to remove from the ground-truth (e.g. background)
    :param train_fraction: positive number in [0,1], implying the fraction of data points being in the training set
    :param epsilon:        positive float, DBSCAN spatial search radius
    :param min_points:     positive int, DBSCAN density threshold
    :return: selection:	   np.array(width, height), 'True' marks training, 'False' test samples.
    """
    # initialize helper variables
    distinct_labels = np.unique(labels)
    selection = np.full(labels.shape, TEST, dtype=np.int8)
    dbscan = DBSCAN(epsilon, min_points, algorithm='kd_tree')

    # for each possible label cluster the labels and assign them to either the or training set
    for label in distinct_labels:
        label_selection = labels == label

        # mark elements for removal
        if label in remove:
            selection[label_selection] = REMOVED
            continue

        # perform a DBSCAN clustering on the current label image coordinates
        indices = np.column_stack(np.where(label_selection))
        clusters = dbscan.fit(indices).labels_
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)

        # determine the sort criterion
        if mode == SIZE:
            ordering = np.lexsort((cluster_counts, unique_clusters >= 0))[::-1]
        elif mode == STDDEV:
            stddevs = []
            for cluster_label in unique_clusters:
                cluster_selection = indices[clusters == cluster_label]
                x = cluster_selection[:, 0]
                y = cluster_selection[:, 1]
                stddevs.append(data[x, y].std())
            ordering = np.lexsort((stddevs, unique_clusters >= 0))[::-1]
        else:
            raise ValueError('Unknown selection mode: {}'.format(mode))

        pick_order = unique_clusters[ordering]
        pick_counts = cluster_counts[ordering]

        # actually assign the training samples to be part of the selection
        assigned_elements = 0
        elements_to_assign = int(np.round(train_fraction * clusters.shape[0]))

        for i, cluster_label in enumerate(pick_order):
            pick_count = pick_counts[i]
            cluster_selection = clusters == cluster_label

            # all the elements of the found cluster can be assigned to the training data entirely
            if pick_count + assigned_elements <= elements_to_assign:
                assignment_indices = indices[cluster_selection]
                selection[assignment_indices[:, 0], assignment_indices[:, 1]] = TRAIN
                assigned_elements += pick_count
            # we have to split the cluster as it is too large
            else:
                reduced = indices[np.where(cluster_selection)[0][:elements_to_assign - assigned_elements]]
                selection[reduced[:, 0], reduced[:, 1]] = TRAIN
                break

    return selection


def safe_patches(out_file, mode, remove, train_fraction, seed, epsilon, min_points, selection, mean, stddev):
    """
    Store the generated patches in an HDF5 file.

    :param out_file:       str containing the path where the hdf5-file will be saved
    :param mode:           str, deciding how to split training and test data, can be 'random', 'size' or 'stddev'
    :param remove:         list(int), list of class labels to remove from the ground-truth (e.g. background)
    :param train_fraction: positive number in [0,1], implying the fraction of data points being in the training set
    :param train_fraction: positive float determining the fraction of the data used for training
    :param seed:           positive, odd int random seed for reproducibility of the random sampling
    :param epsilon:        positive, float, cluster sampling search radius
    :param min_points:     positive, int, cluster sampling density threshold
    :param selection:      np.array(width, height), boolean mask determining train (True) and test (False) set assignment
    :param mean:           np.array, mean of training data
    :param stddev:         np.array, standard deviation of training data
    """
    with h5py.File(out_file, 'w') as handle:
        handle[MODE] = mode
        handle[REMOVED_LABELS] = remove

        if mode == RANDOM:
            handle[SEED] = seed
        else:
            handle[EPSILON] = epsilon
            handle[MIN_POINTS] = min_points

        handle[TRAINING_FRACTION] = train_fraction
        handle[SELECTION] = selection
        handle[MEAN] = mean
        handle[STDDEV] = stddev
