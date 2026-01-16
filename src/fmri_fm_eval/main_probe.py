# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import argparse
import datetime
import json
import importlib.metadata
import math
import time
import traceback
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import torch
import torch.nn as nn
import wandb
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

import fmri_fm_eval.utils as ut
from fmri_fm_eval.classifiers import ClassifierGrid, create_classifier, list_classififiers
from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.models.registry import create_model, list_models
from fmri_fm_eval.datasets.registry import create_dataset, list_datasets

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed probe eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = (
            f"{args.name_prefix}/"
            f"{args.dataset}__{args.model}__{args.representation}__{args.classifier}"
        )
    args.output_dir = f"{args.output_root}/{args.name}"
    output_dir = Path(args.output_dir)

    # remote backup location
    if args.remote_root:
        args.remote_dir = f"{args.remote_root}/{args.name}"
        if S3Path(args.remote_dir).exists():
            ut.rsync(args.remote_dir, args.output_dir)
    else:
        args.remote_dir = None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model probe eval")
    print(f"version: {importlib.metadata.version('fmri-fm-eval')}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating frozen backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))
    backbone.requires_grad_(False)
    backbone.to(device)
    print(f"backbone:\n{backbone}")

    # dataset
    print(f"creating dataset: {args.dataset} ({backbone.__space__})")
    try:
        dataset_dict = create_dataset(
            args.dataset, space=backbone.__space__, **(args.dataset_kwargs or {})
        )
    except Exception as exc:
        msg = traceback.format_exception_only(exc)[0]
        print(f"error loading dataset: {args.dataset} ({backbone.__space__}); exiting\n\n{msg}")

    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")
    train_dataset: HFDataset = dataset_dict["train"]
    args.num_classes = train_dataset.num_classes

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    # balanced class sampling for imbalanced classes
    if args.balanced_sampling:
        weights = 1 / (train_dataset.label_counts / train_dataset.label_counts.max())
        print(f"sampling with balanced class weights: {np.round(weights, 2)}")
        weights = weights[train_dataset.target_ids]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset))
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        drop_last=True,
    )

    eval_loaders_dict = {}
    for split, dataset in dataset_dict.items():
        eval_loaders_dict[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers else None,
            drop_last=False,
        )
    val_loader = eval_loaders_dict["validation"]

    # prediction heads
    print("running backbone on example batch to get embedding dim")
    embed_dim = get_embedding_dim(args, backbone, train_dataset, device)
    print(f"embedding feature dim ({args.representation}): {embed_dim}")

    print("initializing sweep of classifier heads")
    classifiers, param_groups = make_classifiers(args, embed_dim)
    model = ClassifierGrid(backbone, args.representation, classifiers)
    model.to(device)
    print(f"classifiers:\n{model.classifiers}")
    num_params = sum(p.numel() for p in model.classifiers.parameters())
    num_params_train = sum(p.numel() for p in model.classifiers.parameters() if p.requires_grad)
    print(f"classifier params (train): {num_params / 1e6:.1f}M ({num_params_train / 1e6:.1f}M)")

    # optimizer
    print("setting up optimizer")
    total_batch_size = args.batch_size * args.accum_iter
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        print(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        print(f"lr: {args.lr:.2e}")

    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # we use a fixed epoch length determined by steps_per_epoch so that the training
    # schedule is consistent across datasets with varying numbers of samples.
    if not args.steps_per_epoch:
        args.steps_per_epoch = len(train_loader) // args.accum_iter
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_epochs * args.steps_per_epoch
    lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps)
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # load checkpoint/resume training
    # checkpoint only includes classifiers since backbone is frozen to save space
    ckpt_meta = load_model(args, model.classifiers, optimizer)
    if ckpt_meta is not None:
        best_info = ckpt_meta["best_info"]
    else:
        best_info = {"score": float("-inf")}

    # training loss
    criterion = nn.CrossEntropyLoss(reduction="none")

    print(f"start training for {args.epochs} epochs")
    log_wandb = args.wandb and ut.is_main_process()
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            train_loader,
            optimizer,
            lr_schedule,
            epoch,
            device,
        )

        val_stats, _, _ = evaluate(
            args,
            model,
            criterion,
            val_loader,
            epoch,
            device,
            eval_name="validation",
        )

        if log_wandb:
            wandb.log(val_stats, (epoch + 1) * args.steps_per_epoch)

        hparam_id, hparam, cv_score = get_best_hparams(args, model, val_stats)
        hparam_fmt = format_hparam(hparam_id, hparam)
        hparam_scores = {
            metric: val_stats[f"validation/{metric}_{hparam_fmt}"]
            for metric in ["loss"] + args.metrics
        }
        hparam_scores_fmt = "  ".join(
            f"{metric}: {score:.3f}" for metric, score in hparam_scores.items()
        )
        print(
            f"cv: [{epoch}]  best hparam: {hparam} ({hparam_id:03d}) ('{hparam_fmt}')  "
            f"{hparam_scores_fmt}"
        )

        best_stats = {
            "id_best": hparam_id,
            "lr_best": hparam[0] * args.lr,
            "wd_best": hparam[1] * args.weight_decay,
            "train/loss_best": train_stats[f"train/loss_{hparam_fmt}"],
        }
        for metric, score in hparam_scores.items():
            best_stats[f"validation/{metric}_best"] = score

        if log_wandb:
            wandb.log(best_stats, (epoch + 1) * args.steps_per_epoch)

        merged_stats = {"epoch": epoch, **train_stats, **val_stats, **best_stats}
        with (output_dir / "train_log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        if cv_score > best_info["score"]:
            best_info = {
                "score": cv_score,
                "hparam": hparam,
                "hparam_id": hparam_id,
                "epoch": epoch,
            }
            is_best = True
        else:
            is_best = False

        save_model(
            args,
            epoch,
            model.classifiers,
            optimizer,
            meta={"best_info": best_info},
            is_best=is_best,
        )

        if args.remote_dir:
            print(f"backing up to remote: {args.remote_dir}")
            ut.rsync(args.remote_dir, output_dir)

    print("evaluating best model")
    best_ckpt = torch.load(
        output_dir / "checkpoint-best.pth", map_location="cpu", weights_only=True
    )
    model.classifiers.load_state_dict(best_ckpt["model"])
    best_info = best_ckpt["meta"]["best_info"]
    print(f"best model info:\n{json.dumps(best_info)}")

    hparam_id, hparam = best_info["hparam_id"], best_info["hparam"]
    hparam_fmt = format_hparam(hparam_id, hparam)

    header = {
        "model": args.model,
        "repr": args.representation,
        "clf": args.classifier,
        "dataset": args.dataset,
        "epoch": best_info["epoch"],
        "lr": hparam[0] * args.lr,
        "wd": hparam[1] * args.weight_decay,
        "hparam_id": hparam_id,
        "hparam": json.dumps(hparam),
    }
    eval_stats = {
        "eval/epoch_best": header["epoch"],
        "eval/id_best": header["hparam_id"],
        "eval/lr_best": header["lr"],
        "eval/wd_best": header["wd"],
    }
    table = []

    for split, loader in eval_loaders_dict.items():
        stats, preds, targets = evaluate(
            args,
            model,
            criterion,
            loader,
            args.epochs,
            device,
            eval_name=split,
        )
        record = {**header, "split": split}

        preds = preds[:, hparam_id]
        bootstrap_result = bootstrap_ci(args, preds, targets)

        record["loss"] = eval_stats[f"eval/{split}/loss"] = stats[f"{split}/loss_{hparam_fmt}"]
        for metric in args.metrics:
            score = stats[f"{split}/{metric}_{hparam_fmt}"]
            record[metric] = eval_stats[f"eval/{split}/{metric}"] = score
            record[f"{metric}_std"] = bootstrap_result[metric]["std"]

        table.append(record)
        np.savez(output_dir / f"preds_{split}.npz", preds=preds, targets=targets)

    table = pd.DataFrame.from_records(table)
    table_fmt = table.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results:\n\n{table_fmt}\n\n")
    table.to_csv(output_dir / "eval_table.csv", index=False)

    with (output_dir / "eval_log.json").open("w") as f:
        print(json.dumps(eval_stats), file=f)

    if log_wandb:
        wandb.log(eval_stats, args.epochs * args.steps_per_epoch)

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)


@torch.inference_mode()
def get_embedding_dim(
    args: DictConfig,
    backbone: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    loader = DataLoader(dataset, batch_size=1)
    example_batch = next(iter(loader))
    example_batch = ut.send_data(example_batch, device)

    cls_embeds, reg_embeds, patch_embeds = backbone(example_batch)
    all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
    embeds = all_embeds[args.representation]
    embed_dim = embeds.shape[-1]
    return embed_dim


def make_classifiers(args: DictConfig, embed_dim: int):
    # create sweep of classifier heads with varying hparams
    all_classifiers = {}
    param_groups = {}

    clf_fn = partial(
        create_classifier,
        name=args.classifier,
        in_dim=embed_dim,
        out_dim=args.num_classes,
        **(args.classifier_kwargs or {}),
    )

    # all classifiers get same init
    init_state = None

    for lr_multiplier, wd_multiplier in product(args.lr_scale_grid, args.wd_scale_grid):
        clf = clf_fn()
        if init_state is None:
            init_state = clf.state_dict()
        else:
            clf.load_state_dict(init_state)

        all_classifiers[(lr_multiplier, wd_multiplier)] = clf

        for name, param in clf.named_parameters():
            param_wd_multiplier = wd_multiplier

            if name.endswith(".bias") or "norm" in name:
                param_wd_multiplier = 0.0

            key = (lr_multiplier, param_wd_multiplier)
            if key not in param_groups:
                param_groups[key] = {
                    "params": [],
                    "lr_multiplier": lr_multiplier,
                    "wd_multiplier": param_wd_multiplier,
                }
            param_groups[key]["params"].append(param)

    param_groups = list(param_groups.values())
    return all_classifiers, param_groups


def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int):
    warmup = np.linspace(0.0, 1.0, warmup_steps)
    decay = np.cos(np.linspace(0, np.pi, max(total_steps - warmup_steps, 0)))
    decay = (decay + 1) / 2
    lr_schedule = base_lr * np.concatenate([warmup, decay])
    lr_schedule = lr_schedule[:total_steps]
    return lr_schedule


def train_one_epoch(
    args: DictConfig,
    model: ClassifierGrid,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
):
    model.train()
    model.backbone.eval()
    use_cuda = device.type == "cuda"
    log_wandb = args.wandb and ut.is_main_process()
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = args.steps_per_epoch * args.accum_iter if not args.debug else 10

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"train: [{epoch}]"
    all_meters = defaultdict(ut.SmoothedValue)

    num_classifiers = len(model.classifiers)

    data_loader = ut.infinite_data_wrapper(data_loader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)

        global_step = epoch * args.steps_per_epoch + (batch_idx + 1) // args.accum_iter
        need_update = (batch_idx + 1) % args.accum_iter == 0
        if need_update:
            lr = lr_schedule[global_step - 1]
            ut.update_lr(optimizer.param_groups, lr)

        target = batch.pop("target")

        # expand last dimension of target to match prediction
        # note that the num_classifiers dimension has to go at the end bc this is
        # what nn.CrossEntropyLoss expects.
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        # nb, no loss scaler. can add if needed.
        (loss / args.accum_iter).backward()

        if need_update:
            # grad clip per classifier separately
            all_grad = []
            for clf in model.classifiers:
                grad = nn.utils.clip_grad_norm_(clf.parameters(), args.clip_grad)
                all_grad.append(grad)
            total_grad = torch.stack(all_grad).norm()
            optimizer.step()
            optimizer.zero_grad()

        if need_update:
            log_metric_dict = {
                "lr": lr,
                "loss": loss_value,
                "grad": total_grad.item(),
            }
            metric_logger.update(**log_metric_dict)

            all_metric_dict = {}
            all_metric_dict.update(
                {
                    f"loss_{format_hparam(ii, hparam)}": all_loss[ii].item()
                    for ii, hparam in enumerate(model.hparams)
                }
            )
            all_metric_dict.update(
                {
                    f"grad_{format_hparam(ii, hparam)}": all_grad[ii].item()
                    for ii, hparam in enumerate(model.hparams)
                }
            )

            for k, v in all_metric_dict.items():
                all_meters[k].update(v)

            if log_wandb:
                wandb.log({f"train/{k}": v for k, v in log_metric_dict.items()}, global_step)
                wandb.log({f"train/{k}": v for k, v in all_metric_dict.items()}, global_step)

        if use_cuda:
            torch.cuda.synchronize()

    print(f"{header} Summary:", metric_logger)

    stats = {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({f"train/{k}": meter.global_avg for k, meter in all_meters.items()})
    return stats


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: ClassifierGrid,
    criterion: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
):
    model.eval()
    use_cuda = device.type == "cuda"
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = len(data_loader)
    if args.debug:
        epoch_num_batches = min(epoch_num_batches, 10)

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"eval ({eval_name}): [{epoch}]"

    num_classifiers = len(model.classifiers)

    logits = []
    targets = []

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)
        target = batch.pop("target")

        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            logit = model(batch)

        logits.append(logit.cpu().float())
        targets.append(target.cpu())

        if use_cuda:
            torch.cuda.synchronize()

    # average loss and acc over the full eval dataset
    logits = torch.cat(logits)
    targets = torch.cat(targets)

    total_loss = criterion(logits, targets)
    total_loss = total_loss.reshape(-1, num_classifiers).mean(dim=0).tolist()
    stats = {
        f"loss_{format_hparam(ii, hparam)}": total_loss[ii]
        for ii, hparam in enumerate(model.hparams)
    }

    preds = torch.argmax(logits, dim=1).numpy()  # [N, nc]
    targets = targets[:, 0].numpy()  # drop repeated targets [N]

    for metric in args.metrics:
        metric_fn = METRICS[metric]
        for ii, hparam in enumerate(model.hparams):
            stats[f"{metric}_{format_hparam(ii, hparam)}"] = metric_fn(targets, preds[:, ii])

    stats = {f"{eval_name}/{k}": v for k, v in stats.items()}

    return stats, preds, targets


def bootstrap_ci(args: DictConfig, preds: np.ndarray, targets: np.ndarray):
    random_state = sklearn.utils.check_random_state(args.seed)

    sample_scores = defaultdict(list)
    for _ in range(500):
        preds_, targets_ = sklearn.utils.resample(
            preds, targets, random_state=random_state, stratify=targets
        )
        for metric in args.metrics:
            metric_fn = METRICS[metric]
            sample_scores[metric].append(metric_fn(targets_, preds_))

    result = {}
    for metric, values in sample_scores.items():
        result[metric] = {"mean": np.mean(values), "std": np.std(values)}

    return result


METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
}


def format_hparam(idx: int, hparam: tuple[float, float]) -> str:
    lr, weight_decay = hparam
    return f"{idx:03d}_lr{lr:.1e}_wd{weight_decay:.1e}"


def get_best_hparams(args: DictConfig, model: ClassifierGrid, stats: dict[str, float]):
    metric = args.cv_metric
    if metric.startswith("neg_"):
        sign = -1
        metric = metric[4:]
    else:
        sign = 1
    scores = [
        sign * stats[f"validation/{metric}_{format_hparam(ii, hparam)}"]
        for ii, hparam in enumerate(model.hparams)
    ]
    best_id = int(np.argmax(scores))
    best_hparam = model.hparams[best_id]
    best_score = scores[best_id]
    return best_id, best_hparam, best_score


def save_model(args, epoch, model, optimizer, meta=None, is_best=None):
    output_dir = Path(args.output_dir)
    last_checkpoint_path = output_dir / "checkpoint-last.pth"
    best_checkpoint_path = output_dir / "checkpoint-best.pth"

    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": OmegaConf.to_container(args),
        "epoch": epoch,
        "meta": meta,
        "is_best": is_best,
    }

    print(f"saving checkpoint {last_checkpoint_path}")
    torch.save(to_save, last_checkpoint_path)
    if is_best:
        print(f"saving best checkpoint {best_checkpoint_path}")
        torch.save(to_save, best_checkpoint_path)


def load_model(args, model, optimizer):
    ckpt_path = Path(args.output_dir) / "checkpoint-last.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        args.start_epoch = ckpt["epoch"] + 1
        meta = ckpt["meta"]
        print(f"loaded model and optimizer state, resuming training from {args.start_epoch}")
    else:
        args.start_epoch = 0
        meta = None

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "classifier",
        type=str,
        help=f"[{', '.join(list_classififiers())}]",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=f"[{', '.join(list_datasets())}]",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model = args.model
    cfg.representation = args.representation
    cfg.classifier = args.classifier
    cfg.dataset = args.dataset
    main(cfg)
