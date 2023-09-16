# ML Components
```.
├── LICENSE
├── README.md
├── data
│   ├── ner
│   └── search
├── entity
│   ├── config.py
│   ├── experimental
│   │   ├── bert_base_uncased.py
│   │   └── roberta_base_uncased.py
│   └── model.py
├── requirements.txt
└── search
    ├── config.py
    ├── experimental
    │   └── mini_lm_l6_v2.py
    └── models.py
```

## Utilities
- __Entity__: Entity Recognitition from Text Documents.
    - __Experimental__:
        1. Compute loss for all Entitities
        2. Mask all `OBJECT` terms and compute loss
        
```
.
├── config.py
├── experimental
│   ├── bert_base_uncased.py
│   └── roberta_base_uncased.py
└── model.py
```
- __Search__: Utitility for Retrieval Augmented Systems for efficient ChatBot response fetching.
```
.
├── config.py
├── experimental
│   └── mini_lm_l6_v2.py
└── models.py
```

## Data
```
ml_components
```
