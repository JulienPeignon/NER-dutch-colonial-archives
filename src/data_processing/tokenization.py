import os
import json
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def create_tokenized_dataset(
    sentences, labels, save_path="data/tokenized/tokenized_dataset.json"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    unique_labels = sorted(
        set(label for sent_labels in labels for label in sent_labels)
    )
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    dataset = Dataset.from_list(
        [{"tokens": t, "ner_tags": l} for t, l in zip(sentences, labels)]
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
        )

        all_labels = []
        for batch_index, word_ids in enumerate(
            tokenized_inputs.word_ids(batch_index=i)
            for i in range(len(examples["tokens"]))
        ):
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label2id[examples["ner_tags"][batch_index][word_idx]]
                    )
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    tokenized_dataset.to_json(save_path)
    print(f"Tokenized dataset saved at: {save_path}")

    return tokenized_dataset
