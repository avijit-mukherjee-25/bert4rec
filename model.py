
import torch
import torch.nn as nn

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, context_len=32, d_model=256, num_heads=4, num_layers=2, dropout=0.1):
        super(BERT4Rec, self).__init__()

        self.d_model = d_model
        self.context_len = context_len
        self.item_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(context_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_seq):
        """
        input_seq: [B, T] - sequence of item IDs (0 = PAD and MASK)
        logits: [B, T, vocab_size]
        """
        B, T = input_seq.size()
        positions = torch.arange(T, device=input_seq.device).unsqueeze(0).expand(B, T)

        x = self.item_embedding(input_seq) + self.position_embedding(positions)

        attention_mask = (input_seq != 0) # avoid attending padded and masked tokens

        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = self.layer_norm(x)

        logits = self.output_layer(x)  # [B, T, vocab_size]
        return logits