import torch_audiomentations
from .audiomentation_base import BaseAudiomentation


class Gain(BaseAudiomentation):
    def __init__(self, *args, **kwargs):
        super().__init__(aug=torch_audiomentations.Gain(*args, **kwargs))
