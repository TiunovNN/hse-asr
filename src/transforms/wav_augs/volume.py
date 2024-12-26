from torchaudio.transforms import Vol
from .audiomentation_base import InstanceWavAug


class Volume(InstanceWavAug):
    def __init__(self, *args, **kwargs):
        super().__init__(aug=Vol(*args, **kwargs))
