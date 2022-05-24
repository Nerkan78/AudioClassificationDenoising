import torch
from torch import nn


class Reconstructor(nn.Module):
  def __init__(self, n_features, embedding_dim=64):
    super(Reconstructor, self).__init__()
    self.n_features = n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.embedding_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.embedding_dim,
      hidden_size=n_features,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, -1, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return x