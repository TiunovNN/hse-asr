import torch_audiomentations
from .base import InstanceWavAug


class Gain(InstanceWavAug):
    def __init__(self, *args, **kwargs):
        super().__init__(aug=torch_audiomentations.Gain(*args, **kwargs))
