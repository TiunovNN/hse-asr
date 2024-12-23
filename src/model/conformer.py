from torch import nn
from torchaudio.models import Conformer as ConformerTorch

from src.model import BaselineModel


class Conformer(BaselineModel):
    """
    Conformer Model
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        fc_hidden=512,
        num_heads=8,
        depthwise_conv_kernel_size=31,
        **kwargs,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__(n_feats, n_tokens, fc_hidden)

        self.model = ConformerTorch(
            input_dim=n_feats,
            ffn_dim=fc_hidden,
            num_heads=num_heads,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            **kwargs,
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        output, length = self.model(spectrogram.transpose(1, 2), spectrogram_length)
        output = self.net(output)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}
