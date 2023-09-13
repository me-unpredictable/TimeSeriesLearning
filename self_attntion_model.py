
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    A custom self attention layer
    """

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.Q = nn.Linear(in_feat, out_feat)  # Query
        self.K = nn.Linear(in_feat, out_feat)  # Key
        self.V = nn.Linear(in_feat, out_feat)  # Value
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print('in attention:',x.shape)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        d = K.shape[0]  # dimension of key vector
        QK_d = (Q @ K.T) / (d) ** 0.5
        prob = self.softmax(QK_d)
        attention = prob @ V
        return attention


class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_size, hidden,output_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden*output_size) # here
        self.fc1 = nn.Linear(hidden * seq_size, vocab_size)  # here we are changing seq2seq to seq2seq+2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        x = self.attention(x).view(4, -1)
        x = self.fc1(x)
        # print('model op shape:',x.shape)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs
