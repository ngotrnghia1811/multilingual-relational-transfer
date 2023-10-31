# multilingual-relational-transfer

This repository contains the code for:

**Zero-shot Cross-lingual Transfer Learning with Multiple Source and Target Languages for Information Extraction: Language Selection and Adversarial Training** \
*Nghia Trung Ngo, Duy Phung, Thien Huu Nguyen* \
Findings of ACL 2023 &nbsp;|&nbsp; [Paper](https://aclanthology.org/2023.findings-acl.533)

## Method

Most cross-lingual transfer work studies single source → single target transfer. This paper addresses the general **multi-source, multi-target** (ZSCL-M) setting, answering three research questions:

1. **Which linguistic distances best explain pairwise transfer performance** (ZSCL-S)?
2. **How to select optimal source languages** for multi-target transfer (ZSCL-M)?
3. **Can linguistic graph structure improve adversarial transfer** with unlabeled data (ZSCL-R)?

The key findings:
- A **combined metric** `d_comb = 0.4·d_ander-syntax + 0.2·d_inner-phonology + 0.4·d_ander-inventory` achieves the highest correlation with transfer performance (Pearson r > 0.6) across all settings.
- **K-medoids clustering** on `d_comb` identifies optimal source languages, yielding +3.0–3.7 F1 over random selection in multi-source experiments.
- **Graph-relational Domain Adaptation (GrDA)** with the language cluster graph as domain graph further improves multi-source performance (+1.5–2.0 F1), whereas naive DANN hurts (−3 to −12 F1).

## Installation

```bash
git clone https://github.com/nghia-ngo/multilingual-relational-transfer.git
cd multilingual-relational-transfer
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

See [data/README.md](data/README.md) for full instructions.

**MINION** (event detection, 9 languages):
```bash
python scripts/preprocess_minion.py --input_dir /path/to/minion/ --output_dir data/minion/
```

**SMiLER** (relation extraction, 14 languages):
```bash
python scripts/preprocess_smiler.py --input_dir /path/to/smiler/ --output_dir data/smiler/
```

## Usage

### ZSCL-S: Single-source Transfer

Fine-tune on one source language, evaluate zero-shot on all others:

```bash
python train.py --config configs/minion_zscls.yaml --source_lang eng
python train.py --config configs/smiler_zscls.yaml --source_lang ita
```

To run all 9×9 = 81 pairs for MINION (reproduce ZSCL-S heatmap):
```bash
for src in eng hin jpn kor pol por spa swe tur; do
    bash scripts/train_zscls.sh minion $src base
done
```

### ZSCL-M: Multi-source Transfer

Train on medoid languages (selected by `d_comb` clustering), evaluate on all:

```bash
# MINION: sources = {tur, por}  (medoids* config)
python train.py --config configs/minion_zsclm.yaml

# SMiLER: sources = {ita, nld, fas}  (medoids* config)
python train.py --config configs/smiler_zsclm.yaml
```

### ZSCL-R: Relational Transfer (GrDA)

Multi-source + GrDA adversarial training using all-language unlabeled data:

```bash
python train.py --config configs/minion_zsclr.yaml
python train.py --config configs/smiler_zsclr.yaml
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bert_model_name` | `xlm-roberta-base` | Backbone (also tested: small, large) |
| `bert_learning_rate` | 1e-5 | LR for encoder parameters |
| `learning_rate` | 1e-3 | LR for task head parameters |
| `batch_size` | 16 | Per-GPU batch size |
| `max_epoch` | 50 | Training epochs |
| `grda.lambda_gan` | 0.1 | GrDA adversarial loss weight |
| `grda.z_dim` | 64 | Language graph embedding dimension |

## Language Clustering

The combined distance metric clusters languages as follows:

**MINION** (k=2 clusters):

| Cluster | Languages | Medoid |
|---------|-----------|--------|
| `tur*` | kor, hin, jpn, tur | **tur** |
| `por*` | eng, pol, spa, por, swe | **por** |

**SMiLER** (k=4 clusters):

| Cluster | Languages | Medoid |
|---------|-----------|--------|
| `ita*` | fra, ita, pol, por, rus, spa | **ita** |
| `nld*` | deu, eng, nld | **nld** |
| `fas*` | ara, fas, hin, kor | **fas** |
| `jpn*` | jpn | **jpn** |

Red edges connect medoids across clusters in the language-relational graph (Figure 2 in paper).

## Results

**ZSCL-M: Improvement over Random (F1 points), inter-cluster (medoids\*) config**

| Dataset | SMALL | BASE | LARGE | MODEL_AVG |
|---------|------:|-----:|------:|----------:|
| MINION  | +3.5  | +2.9 | +2.5  | **+3.0**  |
| SMiLER  | +4.2  | +3.8 | +3.1  | **+3.7**  |

**ZSCL-R vs ZSCL-M: Improvement (F1 points), inter-cluster (medoids\*) config**

| Dataset | ZSCL-R (ours) | DANN baseline |
|---------|:-------------:|:-------------:|
| MINION  | **+1.5**      | −3.0          |
| SMiLER  | **+2.0**      | −6.9          |

DANN uniformly aligns all language representations, which hurts at scale (−12.1 F1 for SMiLER SMALL). GrDA flexibly aligns according to the language graph, achieving consistent improvements.

## Project Structure

```
multilingual-relational-transfer/
├── train.py                  # Main entry point
├── zsrl/
│   ├── config.py             # ZSRLConfig, ModelConfig, TrainConfig, DataConfig, GrDAConfig
│   ├── distance.py           # URIEL linguistic distances (Hamming/Jaccard/Inner/Anderberg)
│   ├── cluster.py            # K-medoids clustering and language graph construction
│   ├── data.py               # IEDataset for MINION and SMiLER
│   ├── model.py              # XLM-R-based IE model (event detection + RE heads)
│   ├── gda.py                # GrDA: LanguageGraphEmbedding, Discriminator, AdaptiveEncoder
│   ├── train.py              # train_zscls(), train_zsclm(), train_zsclr()
│   └── evaluate.py           # Trigger F1 and relation micro-F1
├── configs/
│   ├── minion_zscls.yaml     # ZSCL-S for MINION
│   ├── minion_zsclm.yaml     # ZSCL-M for MINION (medoids* config)
│   ├── minion_zsclr.yaml     # ZSCL-R for MINION
│   ├── smiler_zscls.yaml     # ZSCL-S for SMiLER
│   ├── smiler_zsclm.yaml     # ZSCL-M for SMiLER (medoids* config)
│   └── smiler_zsclr.yaml     # ZSCL-R for SMiLER
├── scripts/
│   ├── preprocess_minion.py
│   ├── preprocess_smiler.py
│   ├── train_zscls.sh
│   ├── train_zsclm.sh
│   └── train_zsclr.sh
└── data/
    └── README.md             # Data download and format instructions
```

## Citation

```bibtex
@inproceedings{ngo-etal-2023-zero,
    title     = {Zero-shot Cross-lingual Transfer Learning with Multiple Source and Target Languages
                 for Information Extraction: Language Selection and Adversarial Training},
    author    = {Ngo, Nghia Trung and Phung, Duy and Nguyen, Thien Huu},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
    year      = {2023},
    url       = {https://aclanthology.org/2023.findings-acl.533},
    pages     = {8421--8436},
}
```

## License

MIT
