from torch.nn import TransformerEncoder, TransformerEncoderLayer


def Transformer(
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    dropout=0.2,
    batch_first=True,
    num_layers=3,
) -> TransformerEncoder:
    """
    Transformer encoder for sequence data, composed of multiple TransformerEncoderLayer instances.
    Wrapper around the PyTorch TransformerEncoder class.
    """
    layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
    )
    return TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
