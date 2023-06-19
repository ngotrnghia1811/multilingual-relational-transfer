"""
Data loading for MINION (event detection) and SMiLER (relation extraction).

MINION: 9 languages, 16 event types, BIO sequence labeling.
SMiLER: 14 languages, 36 relation types, entity pair classification.

Input format (JSON):
    [
        {
            "doc_id": "...",
            "sentences": [
                {
                    "sent_id": "...",
                    "tokens": ["tok1", "tok2", ...],
                    "trigger_labels": ["O", "B-Attack", "I-Attack", "O", ...],
                    "entities": [{"start": 0, "end": 2, "entity_type": "PER"}, ...],
                    "relations": [{"arg1": 0, "arg2": 1, "relation_type": "per:employee_of"}, ...]
                },
                ...
            ]
        },
        ...
    ]
"""

import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any


def build_vocab(instances: List[Dict], task: str = "event_detection") -> Dict[str, Dict]:
    """Build label vocabularies from dataset instances."""
    if task == "event_detection":
        trigger_type = {"O": 0}
        for inst in instances:
            for label in inst.get("trigger_labels", []):
                if label not in trigger_type:
                    trigger_type[label] = len(trigger_type)
        return {"trigger_type": trigger_type}
    else:
        entity_type = {"O": 0}
        relation_type = {"no_relation": 0}
        for inst in instances:
            for ent in inst.get("entities", []):
                et = ent.get("entity_type", "O")
                if et not in entity_type:
                    entity_type[et] = len(entity_type)
            for rel in inst.get("relations", []):
                rt = rel.get("relation_type", "no_relation")
                if rt not in relation_type:
                    relation_type[rt] = len(relation_type)
        return {"entity_type": entity_type, "relation_type": relation_type}


def load_jsonl_or_json(path: str) -> List[Dict]:
    """Load data from a JSON or JSONL file."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        docs = json.loads(content)
        for doc in docs:
            if "sentences" in doc:
                for sent in doc["sentences"]:
                    sent.setdefault("doc_id", doc.get("doc_id", ""))
                    sentences.append(sent)
            else:
                sentences.append(doc)
    else:
        for line in content.splitlines():
            if line.strip():
                sentences.append(json.loads(line))
    return sentences


class IEDataset(Dataset):
    """
    Dataset for Information Extraction tasks (MINION or SMiLER).

    Handles tokenization using any HuggingFace tokenizer, converting
    word-level annotations to subword-aligned piece indices.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        task: str = "event_detection",
        max_length: int = 128,
        vocab: Optional[Dict] = None,
        language: str = "english",
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        self.language = language

        self.raw_instances = load_jsonl_or_json(data_path)

        if vocab is None:
            self.vocab = build_vocab(self.raw_instances, task)
        else:
            self.vocab = vocab

        self.examples = self._numberize()

    def _numberize(self) -> List[Dict]:
        examples = []
        for inst in self.raw_instances:
            tokens = inst.get("tokens", [])
            if not tokens:
                continue

            pieces = []
            token_lens = []
            for tok in tokens:
                tok_pieces = self.tokenizer.tokenize(tok)
                if not tok_pieces:
                    tok_pieces = [self.tokenizer.unk_token]
                pieces.extend(tok_pieces)
                token_lens.append(len(tok_pieces))

            max_pieces = self.max_length - 2
            pieces = pieces[:max_pieces]
            trimmed_token_lens = []
            total = 0
            for tlen in token_lens:
                if total + tlen > max_pieces:
                    break
                trimmed_token_lens.append(tlen)
                total += tlen
            token_lens = trimmed_token_lens

            piece_idxs = (
                [self.tokenizer.cls_token_id]
                + self.tokenizer.convert_tokens_to_ids(pieces)
                + [self.tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(piece_idxs)
            pad_len = self.max_length - len(piece_idxs)
            piece_idxs += [self.tokenizer.pad_token_id or 0] * pad_len
            attention_mask += [0] * pad_len

            n_tokens = len(token_lens)

            if self.task == "event_detection":
                vocab = self.vocab.get("trigger_type", {})
                raw_labels = inst.get("trigger_labels", ["O"] * len(tokens))
                trigger_label_idxs = [vocab.get(lbl, 0) for lbl in raw_labels[:n_tokens]]
            else:
                trigger_label_idxs = []

            entity_spans = inst.get("entities", [])
            relations = inst.get("relations", [])

            examples.append(
                {
                    "sent_id": inst.get("sent_id", ""),
                    "tokens": tokens[:n_tokens],
                    "piece_idxs": piece_idxs,
                    "token_lens": token_lens,
                    "attention_mask": attention_mask,
                    "trigger_label_idxs": trigger_label_idxs,
                    "entity_spans": entity_spans,
                    "relations": relations,
                    "language": self.language,
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

    def collate_fn(self, batch: List[Dict]) -> Dict:
        piece_idxs = torch.tensor([b["piece_idxs"] for b in batch], dtype=torch.long)
        attention_masks = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        token_lens = [b["token_lens"] for b in batch]
        trigger_label_idxs = [b["trigger_label_idxs"] for b in batch]

        return {
            "sent_ids": [b["sent_id"] for b in batch],
            "tokens": [b["tokens"] for b in batch],
            "piece_idxs": piece_idxs,
            "attention_masks": attention_masks,
            "token_lens": token_lens,
            "trigger_label_idxs": trigger_label_idxs,
            "entity_spans": [b["entity_spans"] for b in batch],
            "relations": [b["relations"] for b in batch],
            "languages": [b["language"] for b in batch],
        }
