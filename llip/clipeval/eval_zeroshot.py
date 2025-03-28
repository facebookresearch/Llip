"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
import torch
import json
import os
from typing import List, Dict, Tuple, Any, Callable
from sklearn import metrics
import itertools
import pandas as pd
from torch import Tensor


def load_metadata(metadir="llip/clipeval", catalog_name="dataset_catalog.json"):
    with open(os.path.join(metadir, catalog_name)) as f:
        catalog = json.load(f)

    with open(os.path.join(metadir, "templates.json")) as f:
        all_templates = json.load(f)

    with open(os.path.join(metadir, "labels.json")) as f:
        all_labels = json.load(f)
    return catalog, all_templates, all_labels


def evaluate(
    d,
    val_loader,
    templates,
    labels,
    model,
    tokenizer,
    is_cmp=False,
    classnorm=False,
    is_itm=False,
):
    """
    Args:
        is_cmp: conditional for Cross Modal Model
    """
    print("Evaluating {}".format(d))

    # determines whether to return outputs or accuracy
    is_acc = d not in [
        "FGVCAircraft",
        "OxfordPets",
        "DollarStreet",
        "Caltech101",
        "Flowers102",
        "Kinetics700",
        "HatefulMemes",
    ]

    is_multilabel = d in {"DollarStreet"}

    is_retrieval = d in {"GeneCISFocusAttribute", "GeneCISFocusObject"}

    # zero shot retrieval
    if is_retrieval:
        if is_cmp:
            acc_or_outputs = evaluate_retrieval_zeroshot_cmp(
                val_loader, model, tokenizer
            )
        else:
            acc_or_outputs = evaluate_retrieval_zeroshot(
                val_loader, model, tokenizer)
    # zero shot classification
    else:
        if is_cmp:
            acc_or_outputs = validate_zeroshot_cmp(
                val_loader,
                templates,
                labels,
                model,
                tokenizer,
                is_acc,
                is_itm,
                classnorm,
                is_multilabel,
            )
        else:
            acc_or_outputs = validate_zeroshot(
                val_loader,
                templates,
                labels,
                model,
                tokenizer,
                is_acc,
                classnorm,
                is_multilabel,
            )

    if d in ["FGVCAircraft", "OxfordPets", "Caltech101", "Flowers102"]:
        if is_cmp:
            metric = {
                k: mean_per_class(acc_or_outputs[0][k], acc_or_outputs[1][k])
                for k in acc_or_outputs[0].keys()
            }
        else:
            metric = mean_per_class(*acc_or_outputs)
    elif d == "Kinetics700":
        if is_cmp:
            metric = {k: 0 for k in acc_or_outputs[0].keys()}
            for k in acc_or_outputs[0].keys():
                top1, top5 = accuracy(
                    acc_or_outputs[0][k], acc_or_outputs[1][k], topk=(1, 5)
                )
                metric[k] = (top1 + top5) / 2
                metric[k] = metric[k].item()
        else:
            top1, top5 = accuracy(*acc_or_outputs, topk=(1, 5))
            metric = (top1 + top5) / 2
            metric = metric.item()
    elif d == "HatefulMemes":
        if is_cmp:
            metric = {
                k: roc_auc(acc_or_outputs[0][k], acc_or_outputs[1][k])
                for k in acc_or_outputs[0].keys()
            }
        else:
            metric = roc_auc(*acc_or_outputs)
    elif d == "DollarStreet":
        if is_cmp:
            metric = {
                k: compute_dollarstreet_metrics(
                    acc_or_outputs, labels, is_cmp=is_cmp, temperature=k
                )
                for k in acc_or_outputs[0].keys()
            }
        else:
            metric = compute_dollarstreet_metrics(
                acc_or_outputs, labels, is_cmp=is_cmp)
    else:
        metric = acc_or_outputs

    return metric


@torch.no_grad()
def build_text_features(
    templates, labels, model, tokenizer, skip_text_projection=False, classnorm=False
):
    # TODO: add device
    text_features = []
    for label in labels:
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]

        texts = tokenizer(texts).to(
            next(model.parameters()).device, non_blocking=True)
        class_embeddings = model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embeddings = class_embeddings.mean(dim=0)
        text_features.append(class_embeddings)
    text_features = torch.stack(text_features, dim=0)
    mean, std = None, None
    if classnorm:
        mean, std = (
            text_features.mean(dim=0)[None, :],
            text_features.std(dim=0)[None, :],
        )
        text_features = (text_features - mean) / std
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, mean, std


@torch.inference_mode()
def build_text_features_cmp(
    templates, labels, model, tokenizer, skip_text_projection=False, classnorm=False
):
    # TODO: add device
    text_features = []
    Qs = []
    for label in labels:
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]

        texts = tokenizer(texts).to(
            next(model.parameters()).device, non_blocking=True)
        Q, class_embeddings = model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embeddings = class_embeddings.mean(dim=0)
        text_features.append(class_embeddings)
        Qs.append(Q.mean(0))
    text_features = torch.stack(text_features, dim=0)
    mean, std = None, None
    if classnorm:
        mean, std = (
            text_features.mean(dim=0)[None, :],
            text_features.std(dim=0)[None, :],
        )
        text_features = (text_features - mean) / std
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    Q = torch.stack(Qs, dim=0)
    return text_features, Q, mean, std


@torch.no_grad()
def validate_zeroshot(
    val_loader,
    templates,
    labels,
    model,
    tokenizer,
    is_acc,
    classnorm=False,
    is_multilabel=False,
):
    # switch to evaluate mode
    model.cuda()
    model.eval()

    total_top1 = 0
    total_images = 0

    all_outputs = []
    all_targets = []
    # for multilabel data
    all_metadata = []

    text_features = None

    for samples in val_loader:
        if text_features is None:
            print("=> encoding captions")
            text_features, mean, std = build_text_features(
                templates, labels, model, tokenizer, classnorm=classnorm
            )

        if isinstance(samples, tuple) or isinstance(samples, list):
            images, target = samples[0], samples[1]
        elif isinstance(samples, dict):
            images, target = samples["pixel_values"], samples["targets"]
        else:
            raise ValueError("unknown sample type", type(samples))

        images = images.cuda(non_blocking=True)
        # no need to move targets to cuda

        # encode images
        image_features = model.encode_image(images)

        if classnorm:
            image_features = (image_features - mean) / std

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image.cpu()
        if is_acc:
            target = target.cpu()
            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)
        else:
            all_outputs.append(logits_per_image)
            all_targets.append(target)
            if is_multilabel:
                all_metadata.append(samples[2])

    if is_acc:
        return 100 * total_top1 / total_images
    elif is_multilabel:
        targets = _format_targets_and_metadata(all_targets, all_metadata)
        return torch.cat(all_outputs), targets
    return torch.cat(all_outputs), torch.cat(all_targets)


def _format_targets_and_metadata(all_targets, all_metadata) -> Dict[str, Any]:
    """Reformats targets and meta data together.
    Example:
        {"region": ["africa", "asia", "asia"],
        "class_labels: ["a, b", "b", "a, b, c"]
        }
    """
    metadata_keys = all_metadata[0].keys()
    metadata_flat = {k: [] for k in metadata_keys}

    for metadata_dict in all_metadata:
        for k in metadata_keys:
            metadata_flat[k].extend(metadata_dict[k])

    targets_flat = list(itertools.chain.from_iterable(all_targets))
    metadata_flat["class_labels"] = targets_flat
    return metadata_flat


@torch.inference_mode()
def validate_zeroshot_cmp(
    val_loader,
    templates,
    labels,
    model,
    tokenizer,
    is_acc,
    is_itm=False,
    classnorm=False,
    is_multilabel: bool = False,
):
    """
    Args:
        is_multilabel: multilabel targets are expected as text labels.
    """
    # switch to evaluate mode
    model.cuda()
    model.eval()

    text_features = None

    temperatures = [1, 3, 5, 8, 10]
    total_top1 = {k: 0 for k in temperatures}
    total_images = {k: 0 for k in temperatures}
    all_outputs = {k: [] for k in temperatures}
    all_targets = {k: [] for k in temperatures}
    # for multilabel data
    all_metadata = {k: [] for k in temperatures}

    for samples in val_loader:
        if text_features is None:
            print("=> encoding captions")
            text_features, Q, mean, std = build_text_features_cmp(
                templates, labels, model, tokenizer, classnorm=classnorm
            )

        if isinstance(samples, tuple) or isinstance(samples, list):
            images, target = samples[0], samples[1]
        elif isinstance(samples, dict):
            images, target = samples["pixel_values"], samples["targets"]
        else:
            raise ValueError("unknown sample type", type(samples))

        images = images.cuda(non_blocking=True)
        # no need to move targets to cuda

        # encode images
        K, V = model.encode_image(images)
        predictor = model.pred

        if classnorm:
            V = (V - mean) / std

        for temp in temperatures:
            image_features = predictor(K, Q, V, temp)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            # cosine similarity as logits
            logits_per_image = (
                image_features * text_features[None]).sum(-1)
            logits_per_image = logits_per_image.cpu()
            if is_acc:
                target = target.cpu()
                # measure accuracy and record loss
                pred = logits_per_image.argmax(dim=1)
                correct = pred.eq(target).sum()
                total_top1[temp] += correct.item()
                total_images[temp] += images.size(0)
            else:
                all_outputs[temp].append(logits_per_image)
                all_targets[temp].append(target)
                if is_multilabel:
                    all_metadata[temp].append(samples[2])

    if is_acc:
        return {
            temp: 100 * total_top1[temp] / total_images[temp] for temp in temperatures
        }
    else:
        # temperature to outputs dictionary
        temperature_to_outputs = {k: torch.cat(
            v) for k, v in all_outputs.items()}
        if not is_multilabel:
            temperature_to_targets = {k: torch.cat(
                v) for k, v in all_targets.items()}
        else:
            temperature_to_targets = _format_temperature_to_targets_and_metadata(
                all_targets, all_metadata
            )

        return temperature_to_outputs, temperature_to_targets


def _format_temperature_to_targets_and_metadata(
    all_targets, all_metadata
) -> Dict[int, dict]:
    """
    Builds a dictionary mapping temperature to targets.

    Args:
        all_targets: {1: [["a, b"], ["b"], ["a, b, c"]]}
        all_metadata: {1: [{"region": "a", "income": "h"},
                       {"region": "b", "income": "l"}]}

    Example:
        {1: {"class_labels": ["a, b", "b", "a, b, c"],
            "region": ["africa", "asia", "asia"],
            }
        }
    """
    # flatten list of multilabels
    temperature_to_targets = dict()

    for temperature in all_targets:
        metadata = all_metadata[temperature]
        metadata_keys = metadata[0].keys()
        metadata_flat = {k: [] for k in metadata_keys}

        for metadata_dict in metadata:
            for k in metadata_keys:
                metadata_flat[k].extend(metadata_dict[k])

        targets_flat = list(
            itertools.chain.from_iterable(all_targets[temperature]))
        metadata_flat["class_labels"] = targets_flat

        temperature_to_targets[temperature] = metadata_flat

    return temperature_to_targets


@torch.no_grad()
def evaluate_retrieval_zeroshot(val_loader, model, tokenizer, topk=(1, 2, 3)):
    # switch to evaluate mode
    model.cuda()
    model.eval()

    topk_to_recall = {k: AverageMeter() for k in topk}

    for batch in val_loader:
        reference_image, caption, candidates, target_idx = batch

        reference_image_features = model.encode_image(
            reference_image.cuda(non_blocking=True)
        )
        text_features = model.encode_text(
            tokenizer(caption).cuda(non_blocking=True))
        # per https://github.com/facebookresearch/genecis/blob/main/utils/model_utils.py#L5
        # (batch, embedding dim)
        reference_features = 0.5 * reference_image_features + 0.5 * text_features
        reference_features = torch.nn.functional.normalize(
            reference_features, dim=-1)

        # candidates
        batch_size, num_candidates, image_size = (
            candidates.shape[0],
            candidates.shape[1],
            candidates.shape[-1],
        )
        gallery_set_batch = candidates.reshape(-1, 3, image_size, image_size)
        # (batch size * num candidates, embedding dim)
        gallery_batch_features = model.encode_image(
            gallery_set_batch.cuda(non_blocking=True)
        )
        # (batch size, num candidates, embedding dim)
        candidates_features = gallery_batch_features.reshape(
            -1, num_candidates, gallery_batch_features.shape[-1]
        )
        candidates_features = torch.nn.functional.normalize(
            candidates_features, dim=-1)

        # (batch size, num candidates, embedding dim)
        similarity_scores = candidates_features * \
            reference_features.unsqueeze(1)
        # (batch size, num candidates)
        similarity_scores = similarity_scores.sum(dim=-1)

        # Sort the similarities so first is highest, which corresponds to the prediction
        _, sort_idxs = similarity_scores.sort(dim=-1, descending=True)

        target_idx = target_idx.cuda(non_blocking=True)
        for k in topk:
            recall = get_recall(sort_idxs[:, :k], target_idx)
            topk_to_recall[k].update(recall, batch_size)

    topk_to_recall_average = {k: v.avg for k, v in topk_to_recall.items()}
    return topk_to_recall_average


@torch.no_grad()
def evaluate_retrieval_zeroshot_cmp(
    val_loader, model, tokenizer, topk=(1, 2, 3), temperatures=[1, 3, 5, 8, 10]
):
    # switch to evaluate mode
    model.cuda()
    model.eval()

    pred = model.pred

    temperature_to_topk_recall = {
        t: {k: AverageMeter() for k in topk} for t in temperatures
    }

    for batch in val_loader:
        reference_image, caption, candidates, target_idx = batch

        Q, _ = model.encode_text(tokenizer(caption).cuda(non_blocking=True))

        K, V = model.encode_image(reference_image.cuda(non_blocking=True))

        batch_size = reference_image.shape[0]

        for temp in temperatures:
            # n number of CLS tokens
            # d number of dimensions of each CLS token
            reference_features = _embed_image_and_text(Q, K, V, temp, pred)

            similarity_scores = []

            for candidate_batch in torch.transpose(candidates, 0, 1):
                # 1st candidate for all samples in batch
                # 2nd candidate for all samples in batch
                # ...
                # candidate_batch is of size (batch size, image dimensions)
                K_candidate, V_candidate = model.encode_image(
                    candidate_batch.cuda(non_blocking=True)
                )
                candidate_batch_features = _embed_image_and_text(
                    Q, K_candidate, V_candidate, temp, pred
                )

                similarity_score = (candidate_batch_features * reference_features).sum(
                    dim=-1
                )
                similarity_scores.append(similarity_score)

            similarity_scores = torch.stack(similarity_scores).T

            # Sort the similarities so first is highest, which corresponds to the prediction
            _, sort_idxs = similarity_scores.sort(dim=-1, descending=True)

            target_idx = target_idx.cuda(non_blocking=True)
            for k in topk:
                recall = get_recall(sort_idxs[:, :k], target_idx)
                temperature_to_topk_recall[temp][k].update(recall, batch_size)

    temperature_to_topk_recall_average = {
        temp: {k: v.avg for k, v in topk_to_recall.items()}
        for temp, topk_to_recall in temperature_to_topk_recall.items()
    }
    return temperature_to_topk_recall_average


def _embed_image_and_text(Q, K, V, temp, pred) -> Tensor:
    """
    Args:
        K: batch size, heads, CLS tokens, dim per token
        V: batch size, heads, CLS tokens, dim per token
        Q: batch size, heads, dim for one token

    Returns:
        batch size, embedding dim
    """
    # project
    reference_features = pred(K, Q, V, temp)

    # normalize
    reference_features = torch.nn.functional.normalize(
        reference_features, dim=-1)
    return reference_features


def get_recall(
    indices, targets
):  # recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    # One hot label branch
    if len(targets.size()) == 1:
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall

    # Multi hot label branch
    else:
        recall = []

        for preds, gt in zip(indices, targets):
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros(
                (max_val + 1,), device=preds.device, dtype=torch.float32
            ).scatter_(0, preds, 1)
            gt_binary = torch.zeros(
                (max_val + 1,), device=gt.device, dtype=torch.float32
            ).scatter_(0, gt.long(), 1)

            success = (preds_binary * gt_binary).sum() > 0

            if success:
                recall.append(1)
            else:
                recall.append(0)

        return torch.Tensor(recall).float().mean()


class AverageMeter(object):
    """Computes and stores the average and current value.
    Based on https://github.com/facebookresearch/genecis/blob/main/utils/gen_utils.py#L13
    """

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


def compute_dollarstreet_metrics(
    outputs_and_targets: Tuple[Dict, Dict],
    class_labels: List[str],
    is_cmp: bool = False,
    temperature: int = 1,
) -> Dict[str, float]:
    """
    Args:
        outputs_and_targets: tuple
            - outputs dictionary mapping temperature -> output logits
            - targts dictionary mapping temperature -> metadata
                - metadata; {"class_labels": list of multiple label text targets,
                             "region": list,
                             }
        class_labels: list of string class names.
    """
    all_outputs, all_targets = outputs_and_targets

    if is_cmp:
        # outputs (6555, 144) = (num samples, num classes)
        outputs, metadata = all_outputs[temperature], all_targets[temperature]
    else:
        outputs, metadata = all_outputs, all_targets

    # predictions
    confidence_top5, indices_top5 = torch.nn.functional.softmax(
        outputs, dim=-1).topk(5)
    confidence_top1, indices_top1 = torch.nn.functional.softmax(
        outputs, dim=-1).topk(1)

    results = []

    for i, sample_labels in enumerate(metadata["class_labels"]):
        correct_indices = [
            class_labels.index(c.strip()) for c in sample_labels.split(",")
        ]

        is_correct_top5 = len(set(correct_indices) &
                              set(indices_top5[i].tolist())) > 0
        is_correct_top1 = len(set(correct_indices) &
                              set(indices_top1[i].tolist())) > 0

        sample_confidence_top5 = confidence_top5[i].tolist()
        sample_confidence_top1 = confidence_top1[i].item()

        predicted_label_top1 = class_labels[indices_top1[i]]
        predicted_labels_top5 = [class_labels[i] for i in indices_top5[i]]

        sample_metadata = {k: metadata[k][i] for k in metadata}

        result = {
            "is_correct_top5": is_correct_top5,
            "is_correct_top1": is_correct_top1,
            "confidence_top5": sample_confidence_top5,
            "confidence_top1": sample_confidence_top1,
            "labels": sample_labels,
            "predicted_label_top1": predicted_label_top1,
            "predicted_labels_top5": predicted_labels_top5,
            **sample_metadata,
        }
        results.append(result)

    df = pd.DataFrame(results)
    metrics = _extract_dollarstreet_metrics_from_results(df)

    # optionally add results
    # metrics["sample_results"] = results

    return metrics


def _extract_dollarstreet_metrics_from_results(df: pd.DataFrame) -> dict:
    metrics = dict()
    per_region_accuracy = df.groupby("region")["is_correct_top5"].mean()
    metrics.update(per_region_accuracy)
    metrics["Overall Top5"] = df["is_correct_top5"].mean().item()
    return metrics


if __name__ == "__main__":
    logits = torch.randn(128, 10)
    targets = torch.randint(size=(128,), low=0, high=10)

    evaluate("imagenet", logits, targets)
