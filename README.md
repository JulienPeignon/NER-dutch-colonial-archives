# **Unsilencing Colonial Archives — Transformer Edition** 🚀  
_Replicating and extending the ideas from [“Unsilencing Colonial Archives via Automated Entity Recognition”](https://arxiv.org/abs/2210.02194) with a custom Transformer architecture._

## Overview 📜
This repository showcases an experimental replication and extension of the approach presented in [“Unsilencing Colonial Archives via Automated Entity Recognition”](https://arxiv.org/abs/2210.02194), applied to historical VOC (Dutch East India Company) testament texts.

The original paper introduces a tailor-made annotation typology to surface both named and unnamed entities—especially marginalized individuals omitted from conventional archival indexes—and evaluates several NER models including a CRF baseline and BERTje + BiLSTM-CRF.

In this project, we go further by implementing two custom architectures:
- A **Transformer decoder with CRF head** 🧪
- A **BERTje + CRF** model 🧠

## Project Structure 🗂️

```plaintext
├── data/
│   ├── raw/                     # Raw input files
│   └── tokenized/               # Tokenized datasets
├── src/
│   ├── data_processing/         # Tokenization, loading, and descriptive stats
│   ├── model/                   # Transformer implementation and training logic
│   └── configuration/           # Device setup and logging config
├── outputs/
│   └── *.pt                     # Trained model weights
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

| Model                          | Precision (Fuzzy) | Recall (Fuzzy) | F1 Score (Fuzzy) |
|--------------------------------|------------------:|---------------:|-----------------:|
| **CRF Baseline (Paper)**       | 0.73              | 0.56           | 0.63             |
| **Transformer + CRF (Ours)**   | 0.398             | 0.232          | 0.293            |
| **BERTje + CRF (Ours)**        | **0.674**         | **0.743**      | **0.707**        |
| **BERTje + BiLSTM-CRF (Paper)**| ~0.68–0.70        | ~0.58–0.59     | 0.63             |

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
