from torchaudio.transforms import Vol
from .audiomentation_base import BaseAudiomentation


class Volume(BaseAudiomentation):
    def __init__(self, *args, **kwargs):
        super().__init__(aug=Vol(*args, **kwargs))
