import torch
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF


class AttentionLayer(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, hidden_size, n_head):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        batch_size, length, _ = q.shape
        d_head = self.hidden_size // self.n_head

        q = self.w_q(q).view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        k = (
            self.w_k(k)
            .view(batch_size, k.shape[1], self.n_head, d_head)
            .transpose(1, 2)
        )
        v = (
            self.w_v(v)
            .view(batch_size, v.shape[1], self.n_head, d_head)
            .transpose(1, 2)
        )

        scores = (q @ k.transpose(-2, -1)) * (d_head**-0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, length, self.hidden_size)
        )
        return self.w_o(out)


class EncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm and GELU."""

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = AttentionLayer(hidden_size, n_head)
        self.dropout1 = nn.Dropout(drop_prob)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        _x = self.norm1(x)
        attn_out = self.attention(_x, _x, _x, src_mask)
        x = x + self.dropout1(attn_out)

        _x = self.norm2(x)
        ffn_out = self.linear2(F.gelu(self.linear1(_x)))
        x = x + self.dropout2(ffn_out)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_size, max_len):
        super().__init__()
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, hidden_size, 2).float()
        encoding = torch.zeros(max_len, hidden_size)
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))
        self.encoding = encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, : x.size(1), :].to(x.device)


class TransformerEmbedding(nn.Module):
    """Token + positional embedding."""

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob, pad_idx):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.drop(self.tok_emb(x) + self.pos_emb(x))


class TransformerForNER(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        hidden_size=256,
        ffn_hidden=512,
        n_head=8,
        n_layers=4,
        max_len=512,
        drop_prob=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = TransformerEmbedding(
            vocab_size, hidden_size, max_len, drop_prob, pad_idx
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embedding(input_ids)
        src_mask = self.make_src_mask(input_ids)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        logits = self.classifier(self.dropout(x))

        if labels is not None:
            return -self.crf(
                logits, labels, mask=attention_mask.bool(), reduction="mean"
            )
        else:
            return self.crf.decode(logits, mask=attention_mask.bool())
