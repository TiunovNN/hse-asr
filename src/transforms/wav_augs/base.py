from torch import Tensor, nn


class InstanceWavAug(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self._aug = aug

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
