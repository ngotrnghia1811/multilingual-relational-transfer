"""
Preprocessing script for the SMiLER multilingual relation extraction dataset.

SMiLER (Samsung Multilingual Entity and Relation extraction) contains 36 relation types
across 14 languages: Arabic (ara), German (deu), English (eng), Farsi (fas),
French (fra), Hindi (hin), Italian (ita), Japanese (jpn), Korean (kor),
Dutch (nld), Polish (pol), Portuguese (por), Russian (rus), Spanish (spa).

Dataset paper: https://aclanthology.org/2021.eacl-main.166/
Download: https://github.com/SamsungLabs/SaMER

Usage:
    python scripts/preprocess_smiler.py \
        --input_dir /path/to/raw/smiler/ \
        --output_dir data/smiler/
"""

import os
import json
import argparse
import csv
from typing import List, Dict, Tuple


LANGUAGES = ["ara", "deu", "eng", "fas", "fra", "hin", "ita", "jpn", "kor", "nld", "pol", "por", "rus", "spa"]

RELATION_TYPES = [
    "no_relation", "country_of_birth", "country_of_death", "countries_of_residence",
    "place_of_birth", "place_of_death", "places_lived", "institution",
    "place_of_burial", "nationality", "employee_of", "member_of", "head_of_state",
    "head_of_government", "minister", "head_of_org", "founded", "founded_by",
    "headquarters", "subsidiary", "product_or_technology", "has_sibling",
    "has_spouse", "has_child", "has_parent", "affiliated_with", "knows", "title",
    "awarded_for", "award_winner", "educated_at", "has_employer",
    "influenced_by", "starred_in", "creator", "director",
]


def read_smiler_tsv(path: str) -> List[Dict]:
    """Parse SMiLER TSV format: sentence, entity1, entity2, label."""
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            tokens = row.get("sentence", "").split()
            e1_text = row.get("entity1", "").strip()
            e2_text = row.get("entity2", "").strip()
            relation = row.get("relation", "no_relation").strip()

            e1_start = next(
                (j for j in range(len(tokens)) if " ".join(tokens[j:j + len(e1_text.split())]) == e1_text),
                0,
            )
            e2_start = next(
                (j for j in range(len(tokens)) if " ".join(tokens[j:j + len(e2_text.split())]) == e2_text),
                min(1, len(tokens) - 1),
            )
            e1_end = e1_start + len(e1_text.split())
            e2_end = e2_start + len(e2_text.split())

            instances.append(
                {
                    "sent_id": f"{i:08d}",
                    "tokens": tokens,
                    "trigger_labels": ["O"] * len(tokens),
                    "entities": [
                        {"start": e1_start, "end": e1_end, "entity_type": "ARG"},
                        {"start": e2_start, "end": e2_end, "entity_type": "ARG"},
                    ],
                    "relations": [
                        {"arg1": 0, "arg2": 1, "relation_type": relation}
                    ],
                }
            )
    return instances


def process_language(input_dir: str, output_dir: str, lang: str):
    """Process a single language from SMiLER."""
    lang_out = os.path.join(output_dir, lang)
    os.makedirs(lang_out, exist_ok=True)

    for split in ["train", "dev", "test"]:
        candidates = [
            os.path.join(input_dir, lang, f"{split}.tsv"),
            os.path.join(input_dir, lang, f"{split}.txt"),
            os.path.join(input_dir, f"{lang}_{split}.tsv"),
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            print(f"  WARNING: no file for {lang}/{split}, skipping")
            continue

        instances = read_smiler_tsv(src)
        out_path = os.path.join(lang_out, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(instances, f, ensure_ascii=False, indent=2)
        print(f"  {lang}/{split}: {len(instances)} instances -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SMiLER dataset for ZSRL experiments"
    )
    parser.add_argument("--input_dir", required=True, help="Path to raw SMiLER data")
    parser.add_argument(
        "--output_dir", default="data/smiler/", help="Output directory for processed JSON"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Languages to process",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for lang in args.languages:
        print(f"Processing {lang}...")
        process_language(args.input_dir, args.output_dir, lang)

    print(f"\nDone. Processed data written to {args.output_dir}")


if __name__ == "__main__":
    main()
