"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse

from pathlib import Path

import multiprocessing as mp

import pprint
import yaml

from llip.training.distributed import init_distributed

from evals.scaffold import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def infer_app(fname):
    path = Path(fname)
    while path.name != '':
        if path.parent.name == 'app':
            return path.name
        path = path.parent
    return None


def process_main(rank, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    app = infer_app(fname)
    if app is None:
        logger.warn('Could not find app name')
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        params['app'] = app
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    from llip.configs import search_config
    config = search_config(params['mc_config_name'])
    params['mc_args'] = config

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    if num_gpus == 1:
        # We don't need to start multiple processes. Allows us to run ibpd.
        process_main(0, args.fname, num_gpus, args.devices)

    else:
        for rank in range(num_gpus):
            mp.Process(
                target=process_main,
                args=(rank, args.fname, num_gpus, args.devices)
            ).start()
