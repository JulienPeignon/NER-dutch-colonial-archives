import pandas as pd
import numpy as np
from collections import defaultdict


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

    # IOB label statistics + unique words
    label_stats = {}
    label_word_sets = defaultdict(set)
    all_labels = set(lab for sent in labels for lab in sent)

    for sent_tokens, sent_labels in zip(sentences, labels):
        for token, label in zip(sent_tokens, sent_labels):
            label_word_sets[label].add(token)

    for label in sorted(all_labels):
        counts = [s.count(label) for s in labels]
        label_stats[label] = {
            "min": int(np.min(counts)),
            "mean": float(np.mean(counts)),
            "max": int(np.max(counts)),
            "n_unique_words": len(label_word_sets[label]),
        }

    # Display
    print("Corpus Summary:")
    print(pd.DataFrame([length_summary]))

    print("\nIOB Tag Statistics (per sentence):")
    df_labels = pd.DataFrame(label_stats).T
    df_labels.columns = ["min", "mean", "max", "n_unique_words"]
    print(df_labels)
