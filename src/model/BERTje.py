import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoConfig


class BertForNER(nn.Module):
    """
    BERTje + linear + CRF for sequence‑label NER.
    Keeps the same .forward signature as TransformerForNER:
        loss = model(input_ids, attention_mask, labels)      # training
        preds = model(input_ids, attention_mask)             # inference
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "GroNLP/bert-base-dutch-cased",
        drop_prob: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()

        # Load BERTje backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        # BERT forward pass
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_out.last_hidden_state  # (B, L, H)

        # Classify each token
        logits = self.classifier(self.dropout(seq_out))  # (B, L, C)

        # CRF loss or decode
        if labels is not None:
            # attention_mask is already 1 for real tokens, 0 for [PAD]
            loss = -self.crf(
                logits, labels, mask=attention_mask.bool(), reduction="mean"
            )
            return loss
        else:
            preds = self.crf.decode(logits, mask=attention_mask.bool())
            return preds
