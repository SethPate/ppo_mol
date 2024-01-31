"""
Shared data for tests.
"""

cfg = dict(
    mode="pretrain",
    embed_dim=768,
    n_heads=12,
    n_layers=12,
    n_vocab=40,
    block_size=128,
    dropout=0.0,
    name="guacamol",
    data_f="guacamol_v1_small.smiles",
    smiles_vocab_f="smiles_vocab.yaml",
    batch_size=4,
    device="cpu",
    lr=0.001,
    n_samples=4,
    sample_len=4,
    save_every=1,
    eval_every=1)
