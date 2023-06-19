"""
Preprocessing script for the MINION multilingual event detection dataset.

MINION (Multilingual INfOrmation extractioN) contains event detection annotations
in 9 languages: English (eng), Hindi (hin), Japanese (jpn), Korean (kor),
Polish (pol), Portuguese (por), Spanish (spa), Swedish (swe), Turkish (tur).

16 event types from ACE 2005 ontology.

Usage:
    python scripts/preprocess_minion.py \
        --input_dir /path/to/raw/minion/ \
        --output_dir data/minion/

Input format: MINION provides CoNLL-style BIO-tagged files per language.
Output format: JSON files per language with sentence-level annotations.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple


LANGUAGES = ["eng", "hin", "jpn", "kor", "pol", "por", "spa", "swe", "tur"]
EVENT_TYPES = [
    "Attack", "Transport", "Die", "Meet", "Arrest-Jail", "Phone-Write",
    "Transfer-Money", "Transfer-Ownership", "Start-Organization",
    "End-Organization", "Elect", "Start-Position", "End-Position",
    "Demonstrate", "Charge-Indict", "Convict",
]


def read_conll_file(path: str) -> List[Dict]:
    """Parse a CoNLL-style BIO file into sentence instances."""
    sentences = []
    tokens, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "trigger_labels": labels})
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])

    if tokens:
        sentences.append({"tokens": tokens, "trigger_labels": labels})

    return sentences


def convert_to_json(sentences: List[Dict], lang: str) -> List[Dict]:
    """Convert sentence list to JSON document format."""
    docs = []
    for i, sent in enumerate(sentences):
        docs.append(
            {
                "doc_id": f"{lang}-doc-{i // 10}",
                "sent_id": f"{lang}-{i:06d}",
                "tokens": sent["tokens"],
                "trigger_labels": sent["trigger_labels"],
                "entities": [],
                "relations": [],
            }
        )
    return docs


def split_train_dev_test(
    instances: List[Dict], dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
) -> Tuple[List, List, List]:
    """Split instances into train/dev/test sets."""
    import random

    rng = random.Random(seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_dev = max(1, int(n * dev_ratio))
    test = shuffled[:n_test]
    dev = shuffled[n_test : n_test + n_dev]
    train = shuffled[n_test + n_dev :]
    return train, dev, test


def process_language(
    input_dir: str, output_dir: str, lang: str, has_splits: bool = True
):
    """Process a single language from MINION."""
    lang_out = os.path.join(output_dir, lang)
    os.makedirs(lang_out, exist_ok=True)

    if has_splits:
        for split in ["train", "dev", "test"]:
            src = os.path.join(input_dir, lang, f"{split}.conll")
            if not os.path.exists(src):
                src = os.path.join(input_dir, lang, f"{split}.txt")
            if not os.path.exists(src):
                print(f"  WARNING: {src} not found, skipping {lang}/{split}")
                continue
            sents = read_conll_file(src)
            docs = convert_to_json(sents, lang)
            out_path = os.path.join(lang_out, f"{split}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
            print(f"  {lang}/{split}: {len(docs)} sentences -> {out_path}")
    else:
        src = os.path.join(input_dir, lang, "all.conll")
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found")
            return
        sents = read_conll_file(src)
        docs = convert_to_json(sents, lang)
        train, dev, test = split_train_dev_test(docs)
        for split, data in [("train", train), ("dev", dev), ("test", test)]:
            out_path = os.path.join(lang_out, f"{split}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  {lang}/{split}: {len(data)} sentences -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MINION dataset for ZSRL experiments"
    )
    parser.add_argument("--input_dir", required=True, help="Path to raw MINION data")
    parser.add_argument(
        "--output_dir", default="data/minion/", help="Output directory for processed JSON"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Languages to process",
    )
    parser.add_argument(
        "--has_splits",
        action="store_true",
        default=True,
        help="Whether the raw data has predefined train/dev/test splits",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for lang in args.languages:
        print(f"Processing {lang}...")
        process_language(args.input_dir, args.output_dir, lang, args.has_splits)

    print(f"\nDone. Processed data written to {args.output_dir}")


if __name__ == "__main__":
    main()
