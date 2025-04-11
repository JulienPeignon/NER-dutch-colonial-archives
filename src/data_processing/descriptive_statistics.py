import pandas as pd
import numpy as np


def descriptive_statistics(sentences, labels):
    # Number of sentences
    num_sentences = len(sentences)

    # Sentence lengths (in number of words)
    sentence_lengths = [len(s) for s in sentences]
    length_summary = {
        "num_sentences": num_sentences,
        "min_length": int(np.min(sentence_lengths)),
        "mean_length": float(np.mean(sentence_lengths)),
        "max_length": int(np.max(sentence_lengths)),
    }

    # IOB label statistics
    label_stats = {}
    all_labels = set(lab for sent in labels for lab in sent)
    for label in sorted(all_labels):
        counts = [s.count(label) for s in labels]
        label_stats[label] = {
            "min": int(np.min(counts)),
            "mean": float(np.mean(counts)),
            "max": int(np.max(counts)),
        }

    # Display
    print("Corpus Summary:")
    print(pd.DataFrame([length_summary]))

    print("\nIOB Tag Statistics (per sentence):")
    df_labels = pd.DataFrame(label_stats).T
    df_labels.columns = ["min", "mean", "max"]
    print(df_labels)
