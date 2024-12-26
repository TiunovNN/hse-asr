import torch_audiomentations
from .base import InstanceWavAug


class AddColoredNoise(InstanceWavAug):
    def __init__(self, *args, **kwargs):
        super().__init__(aug=torch_audiomentations.AddColoredNoise(*args, **kwargs))
