import torch.nn as nn


class GeneralDiscriminator(nn.Module):
    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat, is_san=False):
        os = [d(y, y_hat, is_san=is_san) for d in self.discriminators]
        return [sum(o, []) for o in zip(*os)]
