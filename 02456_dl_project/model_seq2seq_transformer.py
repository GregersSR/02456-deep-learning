import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention is All You Need" paper.
    
    Largely borrowed from the lecture material.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)     # → [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        output_seq_len: int = 10,
        d_model: int = 128, 
        nhead: int = 4, 
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_seq_len = output_seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,              
        )

        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers, norm=encoder_norm)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,  
        )

        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers, norm=decoder_norm)

        self.out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt=None):

        device = src.device

        # Encoder
        src_proj = self.input_proj(src)       # [B, 30, d_model]
        src_emb = self.pos_enc(src_proj)      # [B, 30, d_model]
        enc = self.encoder(src_emb)            # [B, 30, d_model]

        # Decoder

        if self.training:
            if tgt is None:
                raise ValueError("Target sequence 'tgt' must be provided during training for teacher forcing")
            if src.device != tgt.device:
                raise ValueError("src and tgt tensors must be on the same device")
            
            # Teacher forcing - using ground truth
            start_token = src[:,-1:,:]
            shifted_tgt = torch.cat([start_token, tgt[:,:-1,:]], dim=1)
            tgt_proj = self.input_proj(shifted_tgt)
            tgt_emb = self.pos_enc(tgt_proj)

            # Autoregressive mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)

            dec_out = self.decoder(tgt_emb, enc, tgt_mask=tgt_mask)
            out = self.out(dec_out)

        else:
            generated = src[:,-1:,:]

            for i in range(self.output_seq_len):
                tgt_proj = self.input_proj(generated)
                tgt_emb = self.pos_enc(tgt_proj)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)

                dec_out = self.decoder(tgt_emb, enc, tgt_mask=tgt_mask)

                next_output = self.out(dec_out[:,-1:,:])

                generated = torch.cat([generated, next_output], dim=1)

            out = generated[:,1:,:]


        return out
