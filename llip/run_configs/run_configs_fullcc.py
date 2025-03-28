"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# usage:
# python src/training/main.py b32_fullcc
# torchrun --nproc_per_node=8 src/training/main.py b32_fullcc
# python submitit_openclip.py b32_fullcc

from llip.configs import Config
import os


def b32_fullcc_test(**overwrite):
    return Config(
        overwrite,
        one_iter=True,
        inmem=True,
        engine="train_one_epoch_ex",
        distributed=False,
        distributed_engine='ddp',
        eval_steps=5000,
        save_frequency=1,
        train_data="/path/to/data/{0..200000}.tar",
        workers=8,
        train_num_samples=400000000,
        batch_size=512,
        epochs=32,
        model="ViT-B-32-quickgelu",
        name="ViT-B-32",
        force_quick_gelu=True,
        warmup=2000,
        seed=0,
        local_loss=True,
        gather_with_grad=True,
        nodes=8, ngpus=8,
    )


def b32_fullcc(**overwrite):
    return Config(
        overwrite,
        one_iter=True,
        inmem=True,
        engine="train_one_epoch_ex",
        distributed_engine='ddp',
        eval_steps=5000,
        save_frequency=1,
        logs=f"/checkpoint/{os.environ['USER']}/log",
        train_data="/path/to/data/{0..200000}.tar",
        workers=8,
        train_num_samples=400000000,
        batch_size=512,
        epochs=32,
        model="ViT-B-32-quickgelu",
        name="ViT-B-32",
        force_quick_gelu=True,
        warmup=2000,
        seed=0,
        local_loss=True,
        gather_with_grad=True,
        nodes=8, ngpus=8,
    )


def b32_fullcc_16nodes(**overwrite):
    return Config(
        overwrite,
        one_iter=True,
        inmem=True,
        engine="train_one_epoch_ex",
        distributed_engine='ddp',
        eval_steps=5000,
        save_frequency=1,
        train_data="/path/to/data/{0..200000}.tar",
        logs=f"/checkpoint/{os.environ['USER']}/logs",
        workers=8,
        train_num_samples=400000000,
        batch_size=256,
        epochs=32,
        model="ViT-B-32-quickgelu",
        name="ViT-B-32",
        force_quick_gelu=True,
        warmup=2000,
        seed=0,
        local_loss=True,
        gather_with_grad=True,
        nodes=16, ngpus=8,
    )


def b16_fullcc():
    return b32_fullcc(
        model="ViT-B-16-quickgelu",
        name="ViT-B-16",
        grad_checkpointing=True,
    )


def l14_fullcc():
    return b32_fullcc(
        model="ViT-L-14-quickgelu",
        name="ViT-L-14",
        lr=0.0004,
        batch_size=256,
        grad_checkpointing=True,
        nodes=16, ngpus=8,
    )


def h14_fullcc():
    return b32_fullcc(
        model="ViT-H-14-quickgelu",
        name="ViT-H-14",
        lr=0.0004,
        batch_size=256,
        grad_checkpointing=True,
        nodes=16, ngpus=8,
    )


def G14_fullcc():
    return b32_fullcc(
        model="ViT-bigG-14-quickgelu",
        name="ViT-G-14",
        lr=0.0004,
        batch_size=256,
        grad_checkpointing=True,
        nodes=16, ngpus=8,
    )

if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
