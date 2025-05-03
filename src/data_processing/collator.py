import torch


def make_collator(tokenizer, label2id):
    """
    Returns a custom collate_fn that pads input_ids, attention_mask and labels,
    using tokenizer.pad_token_id and label2id["O"].
    """

    def collate_fn(batch):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["input_ids"]) for x in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["attention_mask"]) for x in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["labels"]) for x in batch],
                batch_first=True,
                padding_value=label2id["O"],
            ),
        }

    return collate_fn
