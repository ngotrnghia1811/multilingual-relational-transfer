# Data

This directory stores preprocessed datasets for MINION and SMiLER experiments.

## Directory Structure

```
data/
├── minion/           # Event detection (9 languages)
│   ├── eng/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── hin/
│   ├── jpn/
│   ├── kor/
│   ├── pol/
│   ├── por/
│   ├── spa/
│   ├── swe/
│   └── tur/
└── smiler/           # Relation extraction (14 languages)
    ├── ara/
    ├── deu/
    ├── eng/
    ├── fas/
    ├── fra/
    ├── hin/
    ├── ita/
    ├── jpn/
    ├── kor/
    ├── nld/
    ├── pol/
    ├── por/
    ├── rus/
    └── spa/
```

## MINION

**Task**: Multilingual event trigger detection (BIO sequence labeling)  
**Languages**: 9 — English (eng), Hindi (hin), Japanese (jpn), Korean (kor), Polish (pol),
Portuguese (por), Spanish (spa), Swedish (swe), Turkish (tur)  
**Event types**: 16 (ACE 2005 ontology)

**Download**:
1. Request access at https://github.com/bltlab/minion
2. Place the raw files under a local directory (e.g., `/path/to/minion_raw/`)
3. Run preprocessing:
   ```bash
   python scripts/preprocess_minion.py \
       --input_dir /path/to/minion_raw/ \
       --output_dir data/minion/
   ```

## SMiLER

**Task**: Multilingual entity and relation extraction  
**Languages**: 14 — Arabic (ara), German (deu), English (eng), Farsi (fas), French (fra),
Hindi (hin), Italian (ita), Japanese (jpn), Korean (kor), Dutch (nld), Polish (pol),
Portuguese (por), Russian (rus), Spanish (spa)  
**Relation types**: 36

**Download**:
1. Clone the SMiLER repository:
   ```bash
   git clone https://github.com/SamsungLabs/SaMER
   ```
2. Run preprocessing:
   ```bash
   python scripts/preprocess_smiler.py \
       --input_dir SaMER/data/ \
       --output_dir data/smiler/
   ```

## JSON Format

After preprocessing, each split file contains a list of sentence objects:

```json
[
  {
    "sent_id": "eng-000001",
    "tokens": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    "trigger_labels": ["O", "O", "O", "O", "O", "O", "O"],
    "entities": [
      {"start": 0, "end": 2, "entity_type": "PER"},
      {"start": 5, "end": 6, "entity_type": "LOC"}
    ],
    "relations": [
      {"arg1": 0, "arg2": 1, "relation_type": "place_of_birth"}
    ]
  }
]
```

- `trigger_labels`: BIO tags for event detection (MINION). Labels like `B-Attack`, `I-Attack`, `O`.
- `entities`: token span annotations for relation extraction (SMiLER).
- `relations`: entity-pair relation labels for relation extraction (SMiLER).
