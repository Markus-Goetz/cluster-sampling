#!/usr/bin/env python

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

import data


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        type=str,
        metavar='DATA_FILE',
        dest='data_file',
        help='HDF5 file containing the raw image data'
    )

    parser.add_argument(
        type=str,
        metavar='MASK_FILE',
        dest='mask_file',
        help='HDF5 containing the individual patches'
    )

    return parser.parse_args()


def plot(data_file, mask_file):
    # open the hdf5 files with the raw data and the mask
    data_handle = h5py.File(data_file, 'r')
    mask_handle = h5py.File(mask_file, 'r')

    # load the datasets into memory
    datasets = [
        (np.array(data_handle[data.DATA]).mean(axis=-1), 'viridis',),
        (np.array(data_handle[data.LABELS]), 'nipy_spectral',),
        (np.array(mask_handle[data.SELECTION]).astype(np.uint8), 'gist_gray',)
    ]

    # release the file handles
    mask_handle.close()
    data_handle.close()

    # plot the data
    is_landscape = datasets[0][0].shape[0] < datasets[0][0].shape[1]
    plots_x = 1 if is_landscape else len(datasets)
    plots_y = len(datasets) if is_landscape else 1

    figure = plt.figure(figsize=(16, 10))
    for i, item in enumerate(datasets):
        dataset, cmap = item
        figure.add_subplot(plots_y, plots_x, i + 1)
        plt.imshow(dataset, cmap=cmap)

    plt.show()


if __name__ == '__main__':
    arguments = parse_arguments()
    plot(arguments.data_file, arguments.mask_file)
