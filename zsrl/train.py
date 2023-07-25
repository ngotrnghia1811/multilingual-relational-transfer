"""
Training loops for ZSCL-S, ZSCL-M, and ZSCL-R transfer settings.

  ZSCL-S: Fine-tune on a single source language; evaluate zero-shot on target languages.
  ZSCL-M: Fine-tune on multiple source languages selected by k-medoids clustering.
  ZSCL-R: ZSCL-M + GrDA adversarial training using multi-lingual unlabeled data.
"""

import os
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import tqdm

from .config import ZSRLConfig
from .data import IEDataset
from .model import ZSRLModel
from .gda import GrDAModule
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config: ZSRLConfig) -> torch.device:
    if config.data.use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{config.data.gpu_device}")
    return torch.device("cpu")


def build_optimizer(model: ZSRLModel, config: ZSRLConfig, num_training_steps: int):
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "encoder" in n],
            "lr": config.train.bert_learning_rate,
            "weight_decay": config.train.bert_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "encoder" not in n],
            "lr": config.train.learning_rate,
            "weight_decay": config.train.weight_decay,
        },
    ]
    optimizer = AdamW(param_groups)
    num_warmup = (
        num_training_steps // config.train.max_epoch * config.train.warmup_epoch
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: ZSRLModel,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    config: ZSRLConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0
    optimizer.zero_grad()

    progress = tqdm.tqdm(
        total=len(dataloader), ncols=80, desc=f"Train {epoch}", leave=False
    )
    for batch_idx, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        output = model(batch)
        loss = output.get("loss")
        if loss is None:
            progress.update(1)
            continue

        loss = loss / config.train.accumulate_step
        loss.backward()

        if (batch_idx + 1) % config.train.accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clipping
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * config.train.accumulate_step
            n_steps += 1

        progress.update(1)
    progress.close()
    return total_loss / max(n_steps, 1)


def train_zscls(
    config: ZSRLConfig,
    train_path: str,
    dev_path: str,
    test_path: str,
    output_dir: str,
) -> dict:
    """
    ZSCL-S: Standard fine-tuning on one source language.
    """
    set_seed(config.train.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.bert_model_name, cache_dir=config.model.bert_cache_dir
    )
    train_set = IEDataset(
        train_path, tokenizer, config.data.task, config.data.max_length
    )
    dev_set = IEDataset(
        dev_path, tokenizer, config.data.task, config.data.max_length, vocab=train_set.vocab
    )
    test_set = IEDataset(
        test_path, tokenizer, config.data.task, config.data.max_length, vocab=train_set.vocab
    )

    train_loader = DataLoader(
        train_set, batch_size=config.train.batch_size,
        shuffle=True, drop_last=True, collate_fn=train_set.collate_fn
    )
    dev_loader = DataLoader(
        dev_set, batch_size=config.train.eval_batch_size,
        shuffle=False, collate_fn=dev_set.collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=config.train.eval_batch_size,
        shuffle=False, collate_fn=test_set.collate_fn
    )

    model = ZSRLModel(config, train_set.vocab).to(device)
    n_steps = len(train_loader) * config.train.max_epoch // config.train.accumulate_step
    optimizer, scheduler = build_optimizer(model, config, n_steps)

    best_dev_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    results = []

    for epoch in range(config.train.max_epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device, epoch
        )
        dev_scores = evaluate(model, dev_loader, train_set.vocab, config.data.task, device)
        test_scores = evaluate(model, test_loader, train_set.vocab, config.data.task, device)

        dev_f1 = dev_scores.get("f1", 0.0)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(
                {"model": model.state_dict(), "vocab": train_set.vocab}, best_model_path
            )

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev": dev_scores,
            "test": test_scores,
        }
        results.append(result)
        logger.info(
            f"Epoch {epoch}: loss={train_loss:.4f} dev_f1={dev_f1:.4f}"
        )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results[-1] if results else {}


def train_zsclm(
    config: ZSRLConfig,
    source_paths: dict,
    dev_paths: dict,
    test_paths: dict,
    output_dir: str,
) -> dict:
    """
    ZSCL-M: Multi-source fine-tuning using cluster-selected source languages.
    """
    set_seed(config.train.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.bert_model_name, cache_dir=config.model.bert_cache_dir
    )

    train_sets = [
        IEDataset(path, tokenizer, config.data.task, config.data.max_length, language=lang)
        for lang, path in source_paths.items()
    ]
    combined_vocab = train_sets[0].vocab
    for ts in train_sets[1:]:
        for key in combined_vocab:
            for label, idx in ts.vocab.get(key, {}).items():
                if label not in combined_vocab[key]:
                    combined_vocab[key][label] = len(combined_vocab[key])

    all_train = ConcatDataset(train_sets)
    train_loader = DataLoader(
        all_train, batch_size=config.train.batch_size,
        shuffle=True, drop_last=True, collate_fn=train_sets[0].collate_fn
    )

    model = ZSRLModel(config, combined_vocab).to(device)
    n_steps = len(train_loader) * config.train.max_epoch
    optimizer, scheduler = build_optimizer(model, config, n_steps)

    best_per_lang = {}
    best_model_path = os.path.join(output_dir, "best_model.pt")
    results = []

    for epoch in range(config.train.max_epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device, epoch
        )
        epoch_dev = {}
        epoch_test = {}
        for lang, dev_path in dev_paths.items():
            dev_set = IEDataset(
                dev_path, tokenizer, config.data.task,
                config.data.max_length, vocab=combined_vocab, language=lang
            )
            dev_loader = DataLoader(
                dev_set, batch_size=config.train.eval_batch_size,
                shuffle=False, collate_fn=dev_set.collate_fn
            )
            dev_scores = evaluate(model, dev_loader, combined_vocab, config.data.task, device)
            epoch_dev[lang] = dev_scores
            if dev_scores.get("f1", 0) > best_per_lang.get(lang, {}).get("f1", 0):
                best_per_lang[lang] = dev_scores

            if lang in test_paths:
                test_set = IEDataset(
                    test_paths[lang], tokenizer, config.data.task,
                    config.data.max_length, vocab=combined_vocab, language=lang
                )
                test_loader = DataLoader(
                    test_set, batch_size=config.train.eval_batch_size,
                    shuffle=False, collate_fn=test_set.collate_fn
                )
                epoch_test[lang] = evaluate(
                    model, test_loader, combined_vocab, config.data.task, device
                )

        results.append({"epoch": epoch, "train_loss": train_loss, "dev": epoch_dev, "test": epoch_test})
        logger.info(f"Epoch {epoch}: loss={train_loss:.4f}")

    torch.save({"model": model.state_dict(), "vocab": combined_vocab}, best_model_path)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_per_lang


def train_zsclr(
    config: ZSRLConfig,
    source_paths: dict,
    unlabeled_paths: dict,
    dev_paths: dict,
    test_paths: dict,
    adjacency_matrix: np.ndarray,
    language_order: list,
    output_dir: str,
) -> dict:
    """
    ZSCL-R: Relational-transfer via GrDA adversarial training.

    Source labeled data provides task supervision; multi-lingual unlabeled data
    is used for adversarial language alignment guided by the language graph.
    """
    set_seed(config.train.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.bert_model_name, cache_dir=config.model.bert_cache_dir
    )

    train_sets = {
        lang: IEDataset(
            path, tokenizer, config.data.task, config.data.max_length, language=lang
        )
        for lang, path in source_paths.items()
    }
    unlabeled_sets = {
        lang: IEDataset(
            path, tokenizer, config.data.task, config.data.max_length, language=lang
        )
        for lang, path in unlabeled_paths.items()
    }

    combined_vocab = list(train_sets.values())[0].vocab
    source_indices = [language_order.index(lang) for lang in source_paths if lang in language_order]

    model = ZSRLModel(config, combined_vocab).to(device)

    assert config.grda is not None, "GrDAConfig required for ZSCL-R"
    hidden_size = model.encoder.config.hidden_size
    grda = GrDAModule(
        num_languages=len(language_order),
        input_dim=hidden_size,
        z_dim=config.grda.z_dim,
        hidden_dim=config.grda.hidden_dim,
        output_dim=hidden_size,
        num_pred_classes=len(combined_vocab.get("trigger_type", {"O": 0})),
        adjacency_matrix=adjacency_matrix,
        source_language_indices=source_indices,
        lambda_gan=config.grda.lambda_gan,
        sample_v=config.grda.sample_v,
        sample_v_g=config.grda.sample_v_g,
        lr_e=config.grda.lr_e,
        lr_d=config.grda.lr_d,
        lr_g=config.grda.lr_g,
        device=str(device),
    )

    all_src_train = ConcatDataset(list(train_sets.values()))
    train_loader = DataLoader(
        all_src_train, batch_size=config.train.batch_size,
        shuffle=True, drop_last=True, collate_fn=list(train_sets.values())[0].collate_fn
    )
    n_steps = len(train_loader) * config.train.max_epoch
    optimizer, scheduler = build_optimizer(model, config, n_steps)

    ul_loaders = {
        lang: DataLoader(
            ds, batch_size=config.train.batch_size,
            shuffle=True, collate_fn=ds.collate_fn
        )
        for lang, ds in unlabeled_sets.items()
    }
    ul_iters = {lang: iter(loader) for lang, loader in ul_loaders.items()}

    results = []
    for epoch in range(config.train.max_epoch):
        model.train()
        total_task_loss = 0.0
        n_steps_ep = 0
        optimizer.zero_grad()

        progress = tqdm.tqdm(total=len(train_loader), ncols=80, desc=f"ZSCLR {epoch}", leave=False)
        for batch_idx, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            x_seq_list = []
            y_seq_list = []
            for lang_i in language_order:
                try:
                    ul_batch = next(ul_iters[lang_i])
                except (StopIteration, KeyError):
                    if lang_i in ul_loaders:
                        ul_iters[lang_i] = iter(ul_loaders[lang_i])
                        try:
                            ul_batch = next(ul_iters[lang_i])
                        except StopIteration:
                            ul_batch = batch
                    else:
                        ul_batch = batch
                ul_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in ul_batch.items()
                }
                repr_i = model.get_encoder_output(ul_batch)
                x_seq_list.append(repr_i[:config.train.batch_size])

                if lang_i in source_paths:
                    lbl = batch.get("trigger_label_idxs", [[]])
                    y_flat = torch.tensor(
                        [l[0] if l else -1 for l in lbl], device=device
                    )
                else:
                    y_flat = torch.full((config.train.batch_size,), -1, device=device)
                y_seq_list.append(y_flat)

            min_bs = min(t.shape[0] for t in x_seq_list)
            x_seq = torch.stack([t[:min_bs] for t in x_seq_list], dim=0)
            y_seq = torch.stack([t[:min_bs] for t in y_seq_list], dim=0)

            grda_losses = grda.train_step(x_seq, y_seq)

            output = model(batch)
            task_loss = output.get("loss")
            if task_loss is not None:
                (task_loss / config.train.accumulate_step).backward()
                total_task_loss += task_loss.item()
                n_steps_ep += 1

            if (batch_idx + 1) % config.train.accumulate_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress.update(1)
        progress.close()

        epoch_dev = {}
        epoch_test = {}
        for lang, dev_path in dev_paths.items():
            dev_set = IEDataset(
                dev_path, tokenizer, config.data.task,
                config.data.max_length, vocab=combined_vocab, language=lang
            )
            dev_loader = DataLoader(
                dev_set, batch_size=config.train.eval_batch_size,
                shuffle=False, collate_fn=dev_set.collate_fn
            )
            epoch_dev[lang] = evaluate(model, dev_loader, combined_vocab, config.data.task, device)
            if lang in test_paths:
                test_set = IEDataset(
                    test_paths[lang], tokenizer, config.data.task,
                    config.data.max_length, vocab=combined_vocab, language=lang
                )
                test_loader = DataLoader(
                    test_set, batch_size=config.train.eval_batch_size,
                    shuffle=False, collate_fn=test_set.collate_fn
                )
                epoch_test[lang] = evaluate(model, test_loader, combined_vocab, config.data.task, device)

        avg_loss = total_task_loss / max(n_steps_ep, 1)
        results.append({"epoch": epoch, "train_loss": avg_loss, "dev": epoch_dev, "test": epoch_test})
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

    best_model_path = os.path.join(output_dir, "best_model.pt")
    torch.save({"model": model.state_dict(), "vocab": combined_vocab}, best_model_path)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results[-1] if results else {}
