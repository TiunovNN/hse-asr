import statistics

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class WERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        wers = []
        pred_texts = self.text_encoder.ctc_decode(log_probs, log_probs_length)
        for pred_text, target_text in zip(pred_texts, text):
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        return statistics.fmean(wers)
