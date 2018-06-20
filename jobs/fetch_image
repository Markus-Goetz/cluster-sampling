#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import os.path
import subprocess
import sys


def fetch_image(arguments):
    prefix = arguments.image.split('/')[0]
    docker_tag = arguments.image.split(':')[-1].replace('.', '-')
    container_name = '{}-{}'.format(prefix, docker_tag)

    try:
        output = subprocess.check_output(['udocker', 'ps']).decode('utf-8')
    except subprocess.CalledProcessError:
        print('Could not obtain udocker processes')
        sys.exit(1)

    if container_name not in output:
        print('Pulling container', arguments.image)
        exit = subprocess.call(['udocker', 'pull', arguments.image])
        if exit != 0:
            print('Failed to pull container', arguments.image)
            sys.exit(exit)

        print('Creating container', container_name)
        exit = subprocess.call(['udocker', 'create', '--name={}'.format(container_name), arguments.image])
        if exit != 0:
            print('Failed to create container', container_name)
            sys.exit(exit)
    else:
        print('Container', container_name, 'already exists')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--docker',
        type=str,
        dest='docker',
        default=os.path.join(os.environ['PROJECT'], '.udocker'),
        help='path to your docker installation'
    )
    parser.add_argument(
        metavar='[DOCKER_IMAGE]',
        type=str,
        dest='image',
        default='tensorflow/tensorflow:1.8.0-gpu-py3',
        help='name of the docker image to pull'
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    fetch_image(arguments)
