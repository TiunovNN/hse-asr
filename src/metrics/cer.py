import statistics

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class CERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(log_probs.cpu(), lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[: int(length), :])
            cers.append(calc_cer(target_text, pred_text))
        return statistics.fmean(cers)
