"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

import json
import os
from functools import partial


@torch.no_grad()
def slip_evaluate(args, model, val_transform, tokenizer):
    from llip.clipeval import datasets, eval_zeroshot

    print(args)
    catalog_name = getattr(args, 'catalog_name', 'dataset_catalog.json')
    catalog, all_templates, all_labels = eval_zeroshot.load_metadata(
        "llip/clipeval", catalog_name)

    context_length = getattr(args, 'context_length', 77)
    tokenizer = partial(tokenizer, context_length=context_length)

    if hasattr(model, "module"):
        model = model.module

    metrics = {}
    for d in catalog:
        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        templates = all_templates[d]
        labels = all_labels[d]

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//2, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False)

        is_cmp = args.clip_model == 'CMP' or args.clip_model == 'CLIPQuery'
        is_itm = args.loss == 'ITMLoss'
        metric = eval_zeroshot.evaluate(
            d, val_loader, templates, labels, model, tokenizer, is_cmp=is_cmp, is_itm=is_itm)
        metrics[d] = metric
        json_str = json.dumps({"task": d, "acc": metric})
        if args.rank == 0:
            print(json_str)
            with open(os.path.join(args.output_dir, "slip.txt"), mode="a+", encoding="utf-8") as f:
                f.write(json_str + "\n")
    if args.rank == 0:
        with open(os.path.join(args.output_dir, "mc_eval.txt"), mode="a") as f:
            f.write(json.dumps(metrics) + '\n')
    return metrics
