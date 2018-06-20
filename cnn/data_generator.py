#!/usr/bin/env python

import argparse
import os.path

import data


def generate_data_set(in_file, out_file, mode, remove, fractions, count, epsilon, min_points):
    for fraction in fractions:
        loop_count = count if mode == data.RANDOM else 1
        for i in range(loop_count):
            print('\tGenerating fraction: {}, {}/{}...'.format(fraction, i + 1, loop_count), flush=True)
            suffix = str(i) if mode == data.RANDOM else '{}_{}'.format(epsilon, min_points)
            data.generate_patches(in_file, out_file.format(fraction, suffix), mode, remove, fraction, i, epsilon, min_points)
        print('')


def positive_numeric(argument, kind):
    try:
        parsed = kind(argument)
        if not parsed > 0:
            raise ValueError()
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError('value must be positive numeric value')


def positive_int(argument):
    return positive_numeric(argument, int)


def positive_float(argument):
    return positive_numeric(argument, float)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--fractions',
        type=positive_float,
        default=[0.1, 0.3, 0.6, 0.9],
        nargs='+',
        help='train set fraction'
    )

    parser.add_argument(
        '-s', '--sampling',
        choices=[data.RANDOM, data.SIZE, data.STDDEV],
        default=[data.RANDOM, data.SIZE, data.STDDEV],
        nargs='+',
        help='sampling mode'
    )

    parser.add_argument(
        '-c', '--count',
        type=positive_int,
        default=5,
        nargs='?',
        help='number of different data sets to be generated'
    )

    parser.add_argument(
        '-e', '--epsilon',
        type=positive_float,
        default=1.5,
        nargs='?',
        help='cluster sampling search radius'
    )

    parser.add_argument(
        '-m', '--minpoints',
        type=positive_int,
        default=9,
        nargs='?',
        help='cluster sampling density threshold'
    )

    parser.add_argument(
        '-r', '--remove',
        type=int,
        default=[0],
        nargs='+',
        help='ids of class labels to remove'
    )

    parser.add_argument(
        type=str,
        metavar='FILE',
        dest='file',
        help='path to HDF5 file with the spectral image data'
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    path = os.path.splitext(arguments.file)

    for mode in arguments.sampling:
        print('___{}___'.format(mode))
        generate_data_set(
            arguments.file,
            '{}_{}_{{}}_{{}}{}'.format(path[0], mode, path[1]),
            mode,
            arguments.remove,
            arguments.fractions,
            arguments.count,
            arguments.epsilon,
            arguments.minpoints
        )
        print('')
