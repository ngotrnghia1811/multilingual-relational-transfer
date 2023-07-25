"""
Evaluation metrics for MINION (event detection) and SMiLER (relation extraction).

Event detection scoring:
  - Trigger Identification (trigger_id): span match only (start/end must match)
  - Trigger Classification (trigger): span + type must match

Relation extraction scoring:
  - Entity F1: span + type must match
  - Relation F1: both entity spans + relation type must match (micro F1)
"""

import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


def _safe_f1(tp: int, pred_pos: int, gold_pos: int) -> Dict[str, float]:
    prec = tp / pred_pos if pred_pos > 0 else 0.0
    rec = tp / gold_pos if gold_pos > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def score_trigger_predictions(
    gold_labels: List[List[int]],
    pred_labels: List[List[int]],
    vocab: Dict,
) -> Dict[str, Dict[str, float]]:
    """
    Score BIO-tagged event trigger predictions.

    Returns scores for:
      - trigger_id: span identification only (ignore type)
      - trigger:    span + type (exact match)
      - average:    mean of trigger_id and trigger F1
    """
    o_idx = vocab.get("trigger_type", {}).get("O", 0)

    tp_id, tp_cls = 0, 0
    pred_pos, gold_pos = 0, 0

    for gold_seq, pred_seq in zip(gold_labels, pred_labels):
        gold_spans = _extract_spans(gold_seq, o_idx)
        pred_spans = _extract_spans(pred_seq, o_idx)
        gold_pos += len(gold_spans)
        pred_pos += len(pred_spans)
        for span in pred_spans:
            if (span[0], span[1]) in {(g[0], g[1]) for g in gold_spans}:
                tp_id += 1
            if span in gold_spans:
                tp_cls += 1

    id_scores = _safe_f1(tp_id, pred_pos, gold_pos)
    cls_scores = _safe_f1(tp_cls, pred_pos, gold_pos)
    avg_f1 = (id_scores["f1"] + cls_scores["f1"]) / 2.0

    return {
        "trigger_id": id_scores,
        "trigger": cls_scores,
        "average": {"f1": avg_f1, "precision": avg_f1, "recall": avg_f1},
        "f1": cls_scores["f1"],
    }


def _extract_spans(label_seq: List[int], o_idx: int) -> List[Tuple[int, int, int]]:
    """Extract (start, end, type) spans from BIO label sequence."""
    spans = []
    start, cur_type = None, None
    for i, lbl in enumerate(label_seq):
        if lbl == o_idx:
            if start is not None:
                spans.append((start, i, cur_type))
                start, cur_type = None, None
        else:
            if start is None:
                start, cur_type = i, lbl
            elif lbl != cur_type:
                spans.append((start, i, cur_type))
                start, cur_type = i, lbl
    if start is not None:
        spans.append((start, len(label_seq), cur_type))
    return spans


def evaluate_event_detection(
    model,
    dataloader: DataLoader,
    vocab: Dict,
    device: torch.device,
) -> Dict:
    """Evaluate event detection F1 (trigger identification and classification)."""
    model.eval()
    all_gold, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = model.predict(batch)
            logits = output.get("logits")
            if logits is None:
                continue
            preds = logits.argmax(dim=-1)

            for b_idx, gold_labels in enumerate(batch.get("trigger_label_idxs", [])):
                n = len(gold_labels)
                pred_seq = preds[b_idx, :n].cpu().tolist()
                all_gold.append(gold_labels)
                all_pred.append(pred_seq)

    return score_trigger_predictions(all_gold, all_pred, vocab)


def evaluate_relation_extraction(
    model,
    dataloader: DataLoader,
    vocab: Dict,
    device: torch.device,
) -> Dict:
    """Evaluate relation extraction micro F1 on entity detection and relation classification."""
    model.eval()

    rel_tp, rel_pp, rel_gp = 0, 0, 0
    ent_tp, ent_pp, ent_gp = 0, 0, 0

    rel_vocab = vocab.get("relation_type", {})
    no_rel_idx = rel_vocab.get("no_relation", 0)

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = model.predict(batch)
            entity_logits = output.get("entity_logits")
            rel_results = output.get("rel_results", [])

            gold_spans_batch = batch.get("entity_spans", [])
            gold_rels_batch = batch.get("relations", [])

            if entity_logits is not None:
                preds = entity_logits.argmax(dim=-1)
                for b_idx, gold_spans in enumerate(gold_spans_batch):
                    gold_set = {(s["start"], s["end"], s["entity_type"]) for s in gold_spans}
                    ent_gp += len(gold_set)
                    n = min(preds.shape[1], entity_logits.shape[1])
                    o_idx = vocab.get("entity_type", {}).get("O", 0)
                    pred_spans = _extract_spans(preds[b_idx, :n].cpu().tolist(), o_idx)
                    ent_pp += len(pred_spans)
                    etype_itos = {v: k for k, v in vocab.get("entity_type", {}).items()}
                    for s, e, t in pred_spans:
                        if (s, e, etype_itos.get(t, "")) in gold_set:
                            ent_tp += 1

            for b_idx, i, j, rel_logit in rel_results:
                pred_rel = rel_logit.argmax(-1).item()
                if pred_rel != no_rel_idx:
                    rel_pp += 1
                gold_rels = gold_rels_batch[b_idx] if b_idx < len(gold_rels_batch) else []
                for grel in gold_rels:
                    if grel.get("arg1") == i and grel.get("arg2") == j:
                        rel_gp += 1
                        pred_type = {v: k for k, v in rel_vocab.items()}.get(pred_rel, "")
                        if pred_type == grel.get("relation_type"):
                            rel_tp += 1

    ent_scores = _safe_f1(ent_tp, ent_pp, ent_gp)
    rel_scores = _safe_f1(rel_tp, rel_pp, rel_gp)
    return {
        "entity": ent_scores,
        "relation": rel_scores,
        "f1": rel_scores["f1"],
    }


def evaluate(
    model,
    dataloader: DataLoader,
    vocab: Dict,
    task: str,
    device: torch.device,
) -> Dict:
    """Dispatch to appropriate evaluation function based on task."""
    if task == "event_detection":
        return evaluate_event_detection(model, dataloader, vocab, device)
    else:
        return evaluate_relation_extraction(model, dataloader, vocab, device)
