"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import logging
from functools import partial
import math
import os
import time
from contextlib import suppress
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from evals.main_distributed import launch_evals_with_parsed_args as launch_evals

from llip.open_clip import ClipLoss, get_mean_std
from .distributed import is_master
from .zero_shot import zero_shot_eval


def save_checkpoint(model, optimizer, scaler, epoch, i, args):
    checkpoint_dict = {
        "epoch": epoch,
        # inner loop saves step and args.resume in main.py will decide if a checkpoint is saved by innerloop or epoch loop (in main).
        "epoch_step": i,
        "name": args.name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    # Saving checkpoints. use eval_steps to save a checkpoint.
    if args.save_logs:  # master_only.
        # epoch saving is removed. only save `epoch_latest.pt`.
        if args.save_most_recent:
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def to_device(batch, device, args):
    images, texts = batch
    images = images.to(device=device, non_blocking=True)
    if hasattr(args, "inmem") and args.inmem:
        images = images.to(torch.float32).div_(255.)  # b, 3, 224, 224
        mean, std = get_mean_std(args)
        mean = torch.as_tensor(mean, device=images.device)[None, :, None, None]
        std = torch.as_tensor(std, device=images.device)[None, :, None, None]
        images.sub_(mean).div_(std)
    texts = texts.to(device=device, non_blocking=True)
    return images, texts


def get_autocast(precision):
    if precision == 'amp':
        if 'A100' in torch.cuda.get_device_name(0):
            autocast = partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
        else:
            autocast = torch.cuda.amp.autocast
    else:
        autocast = suppress
    return autocast


def train_one_epoch_ex(model, data, epoch, epoch_step, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    # autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    autocast = get_autocast(args.precision)

    model.train()

    from llip.open_clip import loss
    if hasattr(args, "loss"):
        loss_cls = getattr(loss, args.loss)
    else:
        loss_cls = getattr(loss, "ClipLoss")

    loss = loss_cls(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if hasattr(args, "one_iter") and args.one_iter is True:
        # hack for big dataset using one iterator to run across 400M epoch.
        if not hasattr(data['train'], "dataloader_iter"):
            print(
                f"running dataloader across epochs ({args.train_num_samples} examples per epoch).")
            data['train'].dataloader_iter = iter(dataloader)
        batch_iter = data['train'].dataloader_iter
    else:
        batch_iter = iter(dataloader)

    for i in range(num_batches_per_epoch):
        if i < epoch_step:  # skip to the right i when resuming happens.
            continue
        batch = next(batch_iter)
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts = to_device(batch, device, args)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            if 'CMPLoss' in args.loss or 'ITM' in args.loss:
                (K, V), (Q, zt), out_proj, logit_scale, logit_bias = model(
                    images, texts)
                if 'CMPLoss' in args.loss or args.loss == 'ITMFusionLoss':
                    if 'CMPLoss' in args.loss:
                        zt = F.normalize(zt, dim=-1)
                    total_loss = loss(K, Q, V, zt, out_proj, args.weight_scale,
                                      logit_scale, logit_bias)
                else:
                    total_loss = loss(K, Q, V, out_proj, args.weight_scale,
                                      logit_scale, logit_bias)
            else:
                image_features, text_features, logit_scale, logit_bias = model(
                    images, texts)

                if logit_bias is not None:
                    total_loss = loss(image_features, text_features,
                                      logit_scale, logit_bias)
                else:
                    total_loss = loss(
                        image_features, text_features, logit_scale)

        if torch.isfinite(total_loss).all():
            if scaler is not None:
                scaler.scale(total_loss).backward()

                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.norm_gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.norm_gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        else:
            logging.warn(f"Loss is {total_loss}, skip back prop.")
            import sys
            sys.exit(1)  # protect the checkpoint for debugging.

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            if logit_bias is not None:
                logit_bias_scalar = logit_bias.item()
            else:
                logit_bias_scalar = 0
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"Logit bias: {logit_bias_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if hasattr(args, "save_steps") and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, scaler, epoch, i, args)

        # TODO: copied from main.py, wrap as a function call.
        # TODO (huxu): put eval on master only?
        if hasattr(args, "eval_steps") and (step + 1) % args.eval_steps == 0:
            save_checkpoint(model, optimizer, scaler, epoch, i, args)
            launch_online_evals(args, epoch, args)
            model.train()  # evaluate won't turn model back to train."""
    # end for


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    # autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    autocast = get_autocast(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts = to_device(batch, device, args)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def launch_online_evals(args, epoch, cfg):
    if is_master(args):
        logging.info('Launching evals')
        args_eval = build_eval_args(
            args_pretrain=args,
            run_path=cfg.checkpoint_path, cfg=cfg, tag=f'epoch-{epoch}')
        if args_eval is not None:
            launch_evals(
                delay_seconds=5,
                args_for_evals=args_eval,
                submitit_folder=os.path.join(
                    cfg.output_dir, 'submitit-evals'),
                nodes=1,
                # none so this launched in general partition
                partition=None,
                account=None,
                # partition=None,
                # account="robust",
                tasks_per_node=1)


def build_eval_args(
    args_pretrain,
    run_path,
    cfg,
    tag=None
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.

    :param args_pretrain: (dict) parsing of the config file used for pretrain
    :param tag: (str) tag used to denote this set of evaluations (e.g., 'ep-{pretrain_epoch}')
    """
    # By convention, the pre-training config should specify any required evals
    # in the 'evals' key
    if not args_pretrain.evals:
        logging.info('No evaluations specified in config!')
        return

    args_eval = []
    for i, f in enumerate(args_pretrain.evals):
        with open(f, 'r') as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _args['output_dir'] = run_path  # TODO
            _args['mc_args'] = args_pretrain
            _args['checkpoint_path'] = os.path.join(
                run_path, 'epoch_latest.pt')
            args_eval += [_args]

    return args_eval

# huxu: used inside train_epoch.


def evaluate_ex(model, data, step, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # huxu: epoch = 0 as a trick to bypass checking.
    zero_shot_metrics = zero_shot_eval(model, data, 0, args)
    metrics.update(zero_shot_metrics)

    # autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    autocast = get_autocast(args.precision)
    # and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):  # huxu: val anytime called.
    if 'val' in data:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = to_device(batch, device, args)

                with autocast():
                    image_features, text_features, logit_scale = model(
                        images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Step: {step} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "step": step,
                 "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Step: {step} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val_step/{name}", val, step)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    # autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    autocast = get_autocast(args.precision)
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = to_device(batch, device, args)

                with autocast():
                    image_features, text_features, logit_scale = model(
                        images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch,
                 "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
