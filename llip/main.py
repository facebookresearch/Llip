"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .training import train
from .training.train import train_one_epoch, evaluate
from .training.scheduler import cosine_lr
from .training.params import parse_args
from .training.logger import setup_logging
from .training.distributed import is_master, init_distributed_device, world_info_from_env
from .training.data import get_data
from .open_clip import create_model_and_transforms, trace_model, get_mean_std
from torch.cuda.amp import GradScaler
from torch import optim
import torch
import numpy as np
from datetime import datetime
import random
import os
import logging
import sys


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def save_checkpoint(model, optimizer, scaler, completed_epoch, args):
    checkpoint_dict = {
        "epoch": completed_epoch,
        "name": args.name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    if args.save_logs:
        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (
                completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path,
                             f"epoch_{completed_epoch}.pt"),
            )
        if args.save_most_recent:
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
            )


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args=None):
    if args is None:
        args = parse_args()

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    print("Starting script")
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    print(
        f"Local rank {args.local_rank}, rank {args.rank}, world_size {args.world_size}.")

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and args.resume is None and not hasattr(args, "eval"):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logging.info("Setting up distributed")
    device = init_distributed_device(args)
    logging.info("Done")

    if is_master(args):
        args.checkpoint_path = os.path.join(
            args.logs, args.name, "checkpoints")
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    mean, std = get_mean_std(args)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        mean=mean, std=std,
        inmem=hasattr(args, "inmem"),
        clip_model=args.clip_model,
    )
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        import time
        t0 = time.time()
        logging.info("Distributing model")
        model = torch.nn.parallel.DistributedDataParallel(
            model, static_graph=True)
        logging.info(f"Done {time.time() - t0}")

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        def exclude(
            n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

        def include(n, p): return not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if args.precision == "amp":
            scaler = GradScaler()
        else:
            scaler = None

    # optionally resume from a checkpoint
    start_epoch = 0
    start_epoch_step = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('_orig_mod'):
                    sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                if 'epoch_step' in checkpoint:  # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch_step = checkpoint["epoch_step"] + 1
                    logging.info(
                        f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch}, step {start_epoch_step})")
                else:
                    start_epoch_step = 0
                    logging.info(
                        f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    logging.info("Loading datasets")
    import time
    t0 = time.time()
    data = get_data(args, (preprocess_train, preprocess_val),
                    epoch=start_epoch)
    logging.info(f"Done loading data: {time.time() - t0}")
    assert len(data), 'At least one train or eval dataset must be specified.'

    if hasattr(args, "torchcompile") and args.torchcompile:
        logging.info('Compiling model...')
        try:
            model = torch.compile(model)
        except Exception:
            logging.warn("please use PyTorch 2.0")

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None

    # huxu: merge native/SLIP eval.
    if 'train' not in data or hasattr(args, "eval") and args.eval:
        # TODO: move to below first.
        from training.slip_evaluate import slip_evaluate
        from open_clip import tokenize
        # in case a downloaded model.
        os.makedirs(args.output_dir, exist_ok=True)
        slip_evaluate(args, model, preprocess_val, tokenize)
        evaluate(model, data, start_epoch, args, writer)
        return

    epoch_step = start_epoch_step

    logging.info("Start training")
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        if hasattr(args, "engine"):  # this is where training happens
            engine = args.engine
            module = train
            engine_cls = getattr(module, engine)
            # this is train_one_epoch_ex
            engine_cls(model, data, epoch, epoch_step, optimizer,
                       scaler, scheduler, args, writer)
        else:
            train_one_epoch(model, data, epoch, optimizer,
                            scaler, scheduler, args, writer)

        epoch_step = 0  # reset for next epoch.

        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            from llip.training.train import launch_online_evals
            launch_online_evals(args, epoch, args)
        save_checkpoint(model, optimizer, scaler, completed_epoch, args)

    if hasattr(args, "eval") and args.eval and any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        from training.slip_evaluate import slip_evaluate
        from open_clip import tokenize

        slip_evaluate(args, model, preprocess_val, tokenize)

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path,
             ignore=ignore_patterns('log', 'logs'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    import sys
    from .configs import search_config
    config = search_config(sys.argv[1])
    if len(sys.argv) == 3:
        config.resume = os.path.join(
            config.output_dir, "checkpoints", sys.argv[2])
    main(config)
