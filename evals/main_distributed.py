"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import logging
import os
import pprint
import sys
import time
import yaml
import time

import submitit

from evals.scaffold import main as eval_main
from llip.configs import search_config

from enum import Enum
from typing import Optional
from datetime import datetime



class ClusterType(Enum):
    AWS = "aws"
    FAIR = "fair"
    RSC = "rsc"


def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    if uname.sysname == "Linux":
        if uname.release.endswith("-aws"):
            # Linux kernel versions on AWS instances are of the form "5.4.0-1051-aws"
            return ClusterType.AWS
        elif uname.nodename.startswith("rsc"):
            # Linux kernel versions on RSC instances are standard ones but hostnames start with "rsc"
            return ClusterType.RSC

    return ClusterType.FAIR


def get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/path/to/logs/submitit/')
parser.add_argument(
    '--exclude', type=str,
    help='nodes to exclude from training',
    default=None)
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str, default=None,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:

    def __init__(self, args_eval=None, resume_preempt=None):
        self.eval_name = args_eval['eval_name']
        self.args_eval = args_eval
        self.resume_preempt = resume_preempt

    def __call__(self):
        eval_name = self.eval_name
        args_eval = self.args_eval
        resume_preempt = self.resume_preempt

        logger.info('loaded eval params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args_eval)

        eval_main(
            eval_name,
            args_eval=args_eval,
            resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_eval, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_evals_with_parsed_args(
    args_for_evals,
    submitit_folder,
    partition='learnlab,learnfair',
    account="robust",
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    delay_seconds=10,
    exclude_nodes=None
):
    # hack to avoid bug with slurm submitit looking for pickle in wrong file
    training_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    logger.info(f"parent training job id:{training_job_id}")
    if training_job_id is not None:
        # clear job id variable from the main job
        os.environ.pop("SLURM_ARRAY_JOB_ID")

    if not isinstance(args_for_evals, list):
        logger.info(f'Passed in eval-args of type {type(args_for_evals)}')
        args_for_evals = [args_for_evals]

    time.sleep(delay_seconds)
    logger.info('Launching evaluations in separate jobs...')

    now = datetime.now()
    # Format the datetime object into a string
    formatted_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    job_folder = os.path.join(submitit_folder, formatted_string)
    logger.info(f'saving evaluations to {job_folder}')

    executor = submitit.AutoExecutor(
        folder=job_folder,
        slurm_max_num_timeout=20)

    cluster_type = get_cluster_type()
    if cluster_type == ClusterType.AWS:
        executor.update_parameters(
            slurm_partition=partition,
            # slurm_mem_per_gpu='55G',
            timeout_min=timeout,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=12,
            slurm_account=account,
            slurm_srun_args=["--cpu-bind", "none"],
            gpus_per_node=tasks_per_node)
    elif cluster_type == ClusterType.FAIR:
        executor.update_parameters(
            slurm_partition=partition,
            slurm_account=account,
            timeout_min=timeout,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=10,
            slurm_constraint='volta32gb',
            slurm_srun_args=["--cpu-bind", "none"],
            gpus_per_node=tasks_per_node)

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    fb_trainer = Trainer(args_for_evals[0])
    job = executor.submit(fb_trainer,)
    correct_job_id = job.job_id
    logger.info(f'Launched eval job with id {job.job_id}')
    time.sleep(5)
    if training_job_id is not None:
        os.environ["SLURM_ARRAY_JOB_ID"] = training_job_id




def launch_evals():
    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    configs = []
    for f in config_fnames:
        with open(f, 'r') as y_file:
            config = yaml.load(y_file, Loader=yaml.FullLoader)
            config_model = search_config(config['mc_config_name'])
            config['mc_args'] = config_model

            configs += [config]
        
    print(f'Loaded {len(configs)} config files')
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_evals_with_parsed_args(
        args_for_evals=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        exclude_nodes=args.exclude)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    launch_evals()
