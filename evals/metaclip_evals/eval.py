"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint
import json
from functools import partial

import numpy as np

import torch
import torch.multiprocessing as mp

from llip.training.distributed import init_distributed
from llip.open_clip import tokenize
from llip.open_clip import create_model_and_transforms
from llip.open_clip import get_mean_std
from llip.training.slip_evaluate import slip_evaluate

logging.basicConfig()
logger = logging.getLogger('llip.eval')
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    mc_args = args_eval['mc_args']
    mc_args.rank = rank
    mc_args.world_size = world_size
    mc_args.output_dir = args_eval['output_dir']
    os.makedirs(mc_args.output_dir, exist_ok=True)
    checkpoint_path = args_eval['checkpoint_path']
    # Initialize model
    mean, std = get_mean_std(mc_args)
    model, _, preprocess_val = create_model_and_transforms(
        mc_args.model,
        mc_args.pretrained,
        precision=mc_args.precision,
        device=device,
        jit=mc_args.torchscript,
        force_quick_gelu=mc_args.force_quick_gelu,
        pretrained_image=mc_args.pretrained_image,
        mean=mean, std=std,
        inmem=hasattr(mc_args, 'inmem'),
        clip_model=mc_args.clip_model
    )
    model.eval()
    model.to(device)

    load_mc_ckpt(model, checkpoint_path)

    catalog_name = args_eval.get('catalog_name', 'dataset_catalog.json')
    mc_args.catalog_name = catalog_name
    slip_evaluate(mc_args, model, preprocess_val, tokenize)


def save_result(result, result_dir, filename, rank, world_size, remove_duplicate=""):

    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, rank)
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        logging.warning("rank %d starts merging results." % rank)
        result = []

        for rank in range(world_size):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)

    return final_result_file


def report_metrics(save_dir, result_file):

    results = json.load(open(result_file, 'r'))
    pred = [result['correct'] for result in results]

    accuracy = np.mean(pred)

    metrics = {"accuracy": accuracy}
    with open(os.path.join(save_dir, "evaluate.txt"), 'a') as f:
        f.write(json.dumps(metrics) + "\n")
    logging.info(metrics)
    return metrics


def load_mc_ckpt(
    model,
    checkpoint_path,
):

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('_orig_mod'):
                sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            try:
                # We want to fix pre-trained model to interface them to our arch
                if 'out_proj' in sd:
                    sd['pred.out_proj'] = sd['out_proj']
                    del sd['out_proj']
                if sd['visual.class_embedding'].dim() == 1:
                    sd['visual.class_embedding'] = sd['visual.class_embedding'][None]

                msg = model.load_state_dict(sd)
                logger.info(f'Loaded pretrained model with msg: {msg}')
            except Exception as e:
                logger.error(
                    f"Encountered exception when loading checkpoint {e}")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(
                f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    else:
        logging.warn("=> no checkpoint found at '{}'".format(
            checkpoint_path))
    return model
