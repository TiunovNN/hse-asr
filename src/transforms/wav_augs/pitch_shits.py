import torch_audiomentations
from .base import InstanceWavAug


class PitchShift(InstanceWavAug):
    def __init__(self, *args, **kwargs):
        print(f'{args=} {kwargs=}')
        super().__init__(aug=torch_audiomentations.PitchShift(*args, **kwargs))
