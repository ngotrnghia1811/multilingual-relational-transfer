"""
XLM-RoBERTa-based Information Extraction model for ZSCL experiments.

Supports two tasks:
  - Event detection (MINION): BIO sequence labeling for trigger identification + type classification
  - Relation extraction (SMiLER): Entity span detection + relation type classification

Used in all three transfer settings (ZSCL-S, ZSCL-M, ZSCL-R).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Optional, Tuple


def get_token_repr(
    piece_reprs: torch.Tensor,
    token_lens: List[List[int]],
    strategy: str = "first",
) -> List[torch.Tensor]:
    """
    Map piece-level representations back to token-level.

    Strategies: 'first' (take first subword), 'mean' (average subwords).
    """
    token_reprs = []
    for b_idx in range(len(token_lens)):
        reprs = []
        offset = 1
        for tlen in token_lens[b_idx]:
            if offset >= piece_reprs.shape[1] - 1:
                break
            end = min(offset + tlen, piece_reprs.shape[1] - 1)
            if strategy == "first":
                reprs.append(piece_reprs[b_idx, offset])
            else:
                reprs.append(piece_reprs[b_idx, offset:end].mean(0))
            offset += tlen
        if reprs:
            token_reprs.append(torch.stack(reprs, dim=0))
        else:
            token_reprs.append(piece_reprs[b_idx, :1])
    return token_reprs


def pad_token_reprs(
    token_reprs: List[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad token representations to a uniform length within the batch."""
    max_len = max(r.size(0) for r in token_reprs)
    hidden_size = token_reprs[0].size(-1)
    padded = torch.zeros(len(token_reprs), max_len, hidden_size, device=device)
    mask = torch.zeros(len(token_reprs), max_len, device=device)
    for i, r in enumerate(token_reprs):
        padded[i, : r.size(0)] = r
        mask[i, : r.size(0)] = 1.0
    return padded, mask


class EventDetectionHead(nn.Module):
    """
    Sequence labeling head for event trigger detection (BIO tagging).
    Input: token-level XLM-R representations.
    Output: per-token label logits.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        token_reprs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.classifier(self.dropout(token_reprs))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
        return logits, loss


class EntityHead(nn.Module):
    """Entity type classification head for SMiLER."""

    def __init__(self, hidden_size: int, num_entity_types: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_entity_types)

    def forward(self, token_reprs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(token_reprs))


class RelationHead(nn.Module):
    """
    Relation classification head for SMiLER.
    Concatenates representations of two entity spans and classifies the relation.
    """

    def __init__(self, hidden_size: int, num_relation_types: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_relation_types)

    def forward(
        self,
        span1_repr: torch.Tensor,
        span2_repr: torch.Tensor,
    ) -> torch.Tensor:
        pair_repr = torch.cat([span1_repr, span2_repr], dim=-1)
        return self.classifier(self.dropout(pair_repr))

    def get_span_repr(
        self, token_reprs: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        return token_reprs[start:end].mean(0)


class ZSRLModel(nn.Module):
    """
    Zero-Shot Cross-Lingual IE model.

    XLM-RoBERTa backbone with task-specific classification heads.
    Supports event detection (MINION) and relation extraction (SMiLER).
    """

    def __init__(self, config, vocab: Dict):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.task = config.data.task

        self.encoder = AutoModel.from_pretrained(
            config.model.bert_model_name,
            cache_dir=config.model.bert_cache_dir,
        )
        hidden_size = self.encoder.config.hidden_size

        if self.task == "event_detection":
            num_trigger_types = len(vocab.get("trigger_type", {"O": 0}))
            self.task_head = EventDetectionHead(
                hidden_size, num_trigger_types, config.model.bert_dropout
            )
        else:
            num_entity_types = len(vocab.get("entity_type", {"O": 0}))
            num_relation_types = len(vocab.get("relation_type", {"no_relation": 0}))
            self.entity_head = EntityHead(
                hidden_size, num_entity_types, config.model.bert_dropout
            )
            self.relation_head = RelationHead(
                hidden_size, num_relation_types, config.model.bert_dropout
            )
            self.task_head = self.entity_head

    def encode(
        self,
        piece_idxs: torch.Tensor,
        attention_masks: torch.Tensor,
        token_lens: List[List[int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode input and return (piece-level, token-level) representations."""
        outputs = self.encoder(input_ids=piece_idxs, attention_mask=attention_masks)
        piece_reprs = outputs.last_hidden_state
        token_reprs = get_token_repr(
            piece_reprs, token_lens, self.config.model.multi_piece_strategy
        )
        return piece_reprs, token_reprs

    def get_encoder_output(self, batch: Dict) -> torch.Tensor:
        """Return CLS-token representation (used for GrDA domain adaptation)."""
        outputs = self.encoder(
            input_ids=batch["piece_idxs"],
            attention_mask=batch["attention_masks"],
        )
        return outputs.last_hidden_state[:, 0]

    def forward(self, batch: Dict) -> Dict:
        device = batch["piece_idxs"].device
        piece_reprs, token_reprs = self.encode(
            batch["piece_idxs"], batch["attention_masks"], batch["token_lens"]
        )
        padded, tok_mask = pad_token_reprs(token_reprs, device)

        if self.task == "event_detection":
            labels = None
            label_list = batch.get("trigger_label_idxs")
            if label_list is not None:
                max_len = padded.size(1)
                labels = torch.full(
                    (len(label_list), max_len), -1, dtype=torch.long, device=device
                )
                for i, lbls in enumerate(label_list):
                    n = min(len(lbls), max_len)
                    if n > 0:
                        labels[i, :n] = torch.tensor(lbls[:n], device=device)

            logits, loss = self.task_head(padded, labels)
            return {"loss": loss, "logits": logits, "token_reprs": padded}

        else:
            entity_logits = self.entity_head(padded)
            rel_results = []
            entity_spans_batch = batch.get("entity_spans", [])
            for b_idx, spans in enumerate(entity_spans_batch):
                t_repr = token_reprs[b_idx]
                for i, (s1, e1, _) in enumerate(spans):
                    for j, (s2, e2, _) in enumerate(spans):
                        if i >= j:
                            continue
                        e1_end = min(e1, t_repr.size(0))
                        e2_end = min(e2, t_repr.size(0))
                        span1 = t_repr[s1:e1_end].mean(0) if e1_end > s1 else t_repr[0]
                        span2 = t_repr[s2:e2_end].mean(0) if e2_end > s2 else t_repr[0]
                        rel_logit = self.relation_head(
                            span1.unsqueeze(0), span2.unsqueeze(0)
                        )
                        rel_results.append((b_idx, i, j, rel_logit))
            return {"loss": None, "entity_logits": entity_logits, "rel_results": rel_results}

    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            return self.forward(batch)
