import torch

from models.transformer_encoder import build_transformer_encoder


def main():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 5
    input_dim = 3
    d_model = 8
    n_heads = 2
    n_layers = 1
    dropout = 0.1

    encoder = build_transformer_encoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout_p=dropout,
    )

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = encoder(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    with torch.no_grad():
        src = encoder.encoder_layer(dummy_input) * torch.sqrt(torch.tensor(d_model, dtype=dummy_input.dtype))
        src = encoder.pos_encoder(src)
        seq_len = src.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(src.dtype)
        print("Mask matrix:\n", mask)


if __name__ == "__main__":
    main()
