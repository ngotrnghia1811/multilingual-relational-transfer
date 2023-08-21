"""
Entry point for all ZSRL training modes: ZSCL-S, ZSCL-M, ZSCL-R.

Usage:
    # ZSCL-S: fine-tune on English, evaluate on all MINION languages
    python train.py --config configs/minion_zscls.yaml --source_lang eng

    # ZSCL-M: multi-source training on cluster medoids (minion)
    python train.py --config configs/minion_zsclm.yaml

    # ZSCL-R: GrDA adversarial relational transfer (minion)
    python train.py --config configs/minion_zsclr.yaml
"""

import os
import argparse
import logging
import json
import numpy as np

from zsrl.config import ZSRLConfig
from zsrl.train import train_zscls, train_zsclm, train_zsclr
from zsrl.cluster import kmedoids_cluster, build_language_graph
from zsrl.distance import compute_all_distances, combined_metric

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Lingual Transfer for IE")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--transfer_mode", default=None, help="Override transfer_mode in config")
    parser.add_argument("--source_lang", default=None, help="Source language (ZSCL-S only)")
    parser.add_argument("--selection_mode", default="inter", help="inter|intra|random (ZSCL-M)")
    parser.add_argument("--bert_model_name", default=None, help="Override bert_model_name")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def get_data_paths(config: ZSRLConfig, lang: str, split: str) -> str:
    return os.path.join(config.data.data_dir, lang, f"{split}.json")


def main():
    args = parse_args()
    config = ZSRLConfig.from_yaml(args.config)

    if args.transfer_mode:
        config.transfer_mode = args.transfer_mode
    if args.bert_model_name:
        config.model.bert_model_name = args.bert_model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.train.seed = args.seed

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.dump(config.to_dict(), f)

    mode = config.transfer_mode

    if mode == "zscls":
        source = args.source_lang or config.data.source_languages[0]
        train_path = get_data_paths(config, source, "train")
        dev_path = get_data_paths(config, source, "dev")
        test_path = get_data_paths(config, source, "test")
        logger.info(f"ZSCL-S | source={source} | model={config.model.bert_model_name}")
        result = train_zscls(
            config, train_path, dev_path, test_path,
            os.path.join(config.output_dir, source)
        )
        logger.info(f"Final dev F1: {result.get('dev', {}).get('f1', 'n/a'):.4f}")

    elif mode == "zsclm":
        sources = config.data.source_languages
        source_paths = {lang: get_data_paths(config, lang, "train") for lang in sources}
        dev_paths = {
            lang: get_data_paths(config, lang, "dev")
            for lang in config.data.target_languages
            if os.path.exists(get_data_paths(config, lang, "dev"))
        }
        test_paths = {
            lang: get_data_paths(config, lang, "test")
            for lang in config.data.target_languages
            if os.path.exists(get_data_paths(config, lang, "test"))
        }
        logger.info(f"ZSCL-M | sources={sources} | model={config.model.bert_model_name}")
        results = train_zsclm(config, source_paths, dev_paths, test_paths, config.output_dir)
        for lang, scores in results.items():
            logger.info(f"  {lang}: F1={scores.get('f1', 0):.4f}")

    elif mode == "zsclr":
        sources = config.data.source_languages
        all_langs = config.data.all_languages
        source_paths = {lang: get_data_paths(config, lang, "train") for lang in sources}
        unlabeled_paths = {
            lang: get_data_paths(config, lang, "train")
            for lang in all_langs
            if os.path.exists(get_data_paths(config, lang, "train"))
        }
        dev_paths = {
            lang: get_data_paths(config, lang, "dev")
            for lang in config.data.target_languages
            if os.path.exists(get_data_paths(config, lang, "dev"))
        }
        test_paths = {
            lang: get_data_paths(config, lang, "test")
            for lang in config.data.target_languages
            if os.path.exists(get_data_paths(config, lang, "test"))
        }

        logger.info("Computing linguistic distances for language graph...")
        try:
            distances = compute_all_distances(all_langs)
            d_comb = combined_metric(distances)
        except Exception as e:
            logger.warning(f"lang2vec unavailable ({e}), using random graph")
            n = len(all_langs)
            d_comb = np.random.rand(n, n)
            d_comb = (d_comb + d_comb.T) / 2
            np.fill_diagonal(d_comb, 0)

        n_clusters = len(sources)
        labels, medoids = kmedoids_cluster(d_comb, n_clusters)
        A = build_language_graph(all_langs, labels, medoids, connect_medoids=True)

        np.save(os.path.join(config.output_dir, "lang_graph.npy"), A)
        with open(os.path.join(config.output_dir, "lang_order.json"), "w") as f:
            json.dump(all_langs, f)

        logger.info(f"ZSCL-R | sources={sources} | model={config.model.bert_model_name}")
        result = train_zsclr(
            config, source_paths, unlabeled_paths, dev_paths, test_paths,
            A, all_langs, config.output_dir
        )
        logger.info("Training complete.")

    else:
        raise ValueError(f"Unknown transfer_mode: '{mode}'. Use zscls|zsclm|zsclr")


if __name__ == "__main__":
    main()
