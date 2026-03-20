import torch.nn as nn

from SESTN.mambaForEEG import Mamba, RMSNorm
from SESTN.mamba_lmForEEG import MambaLMConfig


class MambaLMPara(nn.Module):
    def __init__(self, lm_config: MambaLMConfig, args):
        super(MambaLMPara, self).__init__()
        self.lm_config = lm_config
        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout
        self.l_relu = args.lr
        self.config = lm_config.to_mamba_config()

        # self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(310, self.lm_config.vocab_size, bias=False)
        nn.init.kaiming_uniform_(self.lm_head.weight)
        # self.lm_head.weight = self.embedding.weight
        # self.mlp = nn.Linear(self.lm_config.vocab_size * 3, 4, bias=False)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        # tokens : (B, S, C, F)
        # logits : (B, L, vocab_size)
        x = self.mamba(input)
        x = self.norm_f(x)  # .view(-1, 3, 5, 310)
        logits = self.lm_head(x)
        # logits = self.mlp(logits.flatten(start_dim=1))
        # logits = self.softmax(logits)
        return logits
