from transformer import Encoder, EncoderLayer, MutilHeadAttention, PositionalEncoding, PositionwiseFeedForward
import torch


def get_sequence_features(sequence, w2v):
    tokens = [token.lower() for token in sequence.split(sep=' ')]
    features = []
    for token in tokens:
        features.append(w2v.vectors[w2v.stoi[token]])
    return torch.stack(features, dim=0).unsqueeze(0)


def get_encoder(head_num, d_model, d_ff, dropout=0.0, num_layers=1):
    self_attn = MutilHeadAttention(head_num, d_model, dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    encoder = Encoder(EncoderLayer(d_model, self_attn, feed_forward, dropout), 1)
    return encoder


def get_positional_features(input_features):
    d_model = input_features.size(-1)
    positional = PositionalEncoding(d_model, dropout=0.0)
    return positional(input_features)
