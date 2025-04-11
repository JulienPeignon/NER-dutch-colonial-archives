# **Unsilencing Colonial Archives — Transformer Edition** 🚀  
_Replicating and extending the ideas from [“Unsilencing Colonial Archives via Automated Entity Recognition”](https://arxiv.org/abs/2210.02194) with a custom Transformer architecture._

## Overview 📜
This repository showcases an experimental replication of the approach presented in [“Unsilencing Colonial Archives via Automated Entity Recognition”](https://arxiv.org/abs/2210.02194), using a **self-programmed Transformer** to detect various entities (including named and unnamed persons) in historical VOC (Dutch East India Company) testament texts.

The original paper proposes a tailor-made annotation typology to tackle challenging archival data, spotlighting how automation can **broaden access** to marginalized voices often left out of conventional archival indexes. Here, instead of relying on off-the-shelf models, we’re **rolling our own** Transformer 🛠️.

## Project Structure 🗂️

```plaintext
├── data/
│   ├── raw/                     # Raw input files
│   └── tokenized/               # Tokenized datasets
├── src/
│   ├── data_processing/         # Tokenization, loading, and descriptive stats
│   ├── model/                   # Transformer implementation and training logic
│   └── configuration/           # Device setup and logging config
├── checkpoints/                 # Saved model checkpoints
├── main.ipynb                   # Main notebook for exploratory work
│ 
├── requirements.txt             # Python dependencies
├── README.md                    # Project description and setup instructions
├── LICENSE                      # License file (MIT)
└── .pre-commit-config.yaml      # Pre-commit hook setup
```

## Installation & Setup ⚙️

### Clone the Repository

```bash
git clone https://github.com/JulienPeignon/NER-dutch-colonial-archives
cd unsilencing-colonial-archives-transformer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Results 📊

| Model                    | Precision (Fuzzy) | Recall (Fuzzy) | F1 Score (Fuzzy) |
|--------------------------|------------------:|---------------:|-----------------:|
| **CRF Baseline (Paper)** | 0.73              | 0.56           | 0.63             |
| **Transformer (Ours)**   | 0.XX              | 0.XX           | 0.XX             |

---

## Contributing 🙌

Feel free to submit an issue or PR to improve the codebase.

---

## License 📝

This project is licensed under the **MIT License**. Refer to the [LICENSE](LICENSE) file for details.

---

## References

- **Paper**: [“Unsilencing Colonial Archives via Automated Entity Recognition”](https://arxiv.org/abs/2210.02194)
- **Dataset & Annotations**: Provided by the authors in their shared tasks.
