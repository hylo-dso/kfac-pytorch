import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNCell(nn.Module):
    """Custom Elman RNN cell with tanh or ReLU non-linearity.

    h' = tanh(W_{ih}x + b_{ih} + W_{hh}h + b_{hh})

    Uses torch.nn.Linear layers for easy compatibility with KFAC. Based on
    torch.nn.modules.RNNCell.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RNNCell, self).__init__()

        self.linear_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        if nonlinearity.lower() == 'tanh':
            self.nonlin = torch.tanh
        elif nonlinearity.lower() == 'relu':
            self.nonlin = F.relu
        else:
            raise ValueError('Unknown linearity {}'.format(nonlinearity))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input, hidden=None):
        """Compute forward pass.

        Args:
          input: shape (batch, input_size)
          hidden: shape (batch, hidden_size)

        Returns:
          h' of shape (batch, hidden_size)
        """
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, 
                                 device=input.device)
        return self.nonlin(self.linear_i(input) + self.linear_h(hidden))


class LSTMCell(nn.Module):
    """Custom LSTM cell.

    Uses torch.nn.Linear layers for easy compatibility with KFAC. Based on
    torch.nn.modules.LSTMCell.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()

        self.linear_i_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_i_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_f_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_f_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_g_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_g_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_o_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_o_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input, hidden=None):
        """Compute forward pass.

        Args:
          input: shape (batch, input_size)
          hidden: tuple (h, c) where h and c have shape (batch, hidden_size)

        Returns:
          (h', c') where h' and c' have shape (batch, hidden_size)
        """
        if hidden is None:
            hidden = (torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, 
                                  device=input.device),
                      torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, 
                                  device=input.device))
        h, c = hidden
        i = torch.sigmoid(self.linear_i_i(input) + self.linear_i_h(h))
        f = torch.sigmoid(self.linear_f_i(input) + self.linear_f_h(h))
        g = torch.tanh(self.linear_g_i(input) + self.linear_g_h(h))
        o = torch.sigmoid(self.linear_o_i(input) + self.linear_o_h(h))
        c_prime = (f * c) + (i * g)
        h_prime = o * torch.tanh(c_prime)
        return h_prime, c_prime

