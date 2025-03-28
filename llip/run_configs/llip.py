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


def llip_b32(**overwrite):
    return Config(
        overwrite,
        one_iter=True,
        inmem=True,
        loss="CMPLoss",
        weight_scale=5,
        engine="train_one_epoch_ex",
        distributed_engine='ddp',
        eval_steps=5000,
        beta2=0.95,
        save_frequency=1,
        train_data="/path/to/dataset/{0..200000}.tar",
        logs="/path/to/llip/",
        workers=8,
        train_num_samples=400000000,
        batch_size=256,
        epochs=32,
        model="ViT-B-32-quickgelu-siglip-ncls:64",
        name="ViT-B-32",
        force_quick_gelu=True,
        warmup=2000,
        seed=0,
        local_loss=True,
        gather_with_grad=True,
        clip_model="CMP",
        torchcompile=True,
        grad_checkpointing=False,
        nodes=16, ngpus=8,
        evals=['configs_evals/llip_online.yaml']
    )


def llip_b32_k64():
    return llip_b32(
        model="ViT-B-16-quickgelu-siglip-ncls:64",
        name="ViT-B-16",
    )


def llip_b32_k128():
    return llip_b32(
        model="ViT-B-32-quickgelu-siglip-ncls:128",
    )


def llip_b32_k32():
    return llip_b32(
        model="ViT-B-32-quickgelu-siglip-ncls:32",
    )


def llip_b16_k32():
    return llip_b32(
        weight_scale=5,
        model="ViT-B-16-quickgelu-siglip-ncls:32",
        torchcompile=True,
    )


def llip_l14_k32():
    return llip_b32(
        model="ViT-L-14-quickgelu-siglip-ncls:32",
        name="ViT-L-14",
        lr=0.0004,
        batch_size=256,
        nodes=16, ngpus=8,
        torchcompile=True,
        context_length=64,
        weight_scale=1,
        grad_checkpointing=False,
    )


def llip_l14_k64():
    return llip_b32(
        model="ViT-L-14-quickgelu-siglip-ncls:64",
        name="ViT-L-14",
        lr=0.0004,
        batch_size=256,
        nodes=16, ngpus=8,
        torchcompile=True,
        weight_scale=1,
        grad_checkpointing=False,
    )


def llip_h14_k64():
    return llip_b32(
        model="ViT-H-14-quickgelu-siglip-ncls:64-norm",
        name="ViT-H-14",
        lr=0.0004,
        batch_size=128,
        nodes=32, ngpus=8,
        torchcompile=False,
        weight_scale=5,
        grad_checkpointing=True,
    )


def llip_G14_k64():
    return llip_b32(
        model="ViT-bigG-14-quickgelu-siglip-ncls:64-norm",
        name="ViT-G-14",
        lr=0.0004,
        batch_size=128,
        nodes=32, ngpus=8,
        torchcompile=False,
        weight_scale=5,
        grad_checkpointing=True,
    )

def llip_b16_k32_12_node():
    """For launching on smaller cluster"""
    return llip_b32(
        weight_scale=5,
        model="ViT-B-16-quickgelu-siglip-ncls:32",
        torchcompile=False,
        nodes=12,
        batch_size=342,
        workers=6,
    )


def llip_b16_k32_2_node_debug():
    """For launching on smaller cluster"""
    return llip_b32(
        weight_scale=5,
        model="ViT-B-16-quickgelu-siglip-ncls:32",
        torchcompile=False,
        nodes=2,
        batch_size=342,
        eval_steps=100,
        train_num_samples=200000,
        workers=6,
    )


if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
