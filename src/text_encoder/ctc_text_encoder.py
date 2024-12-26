import atexit
import re
from string import ascii_lowercase
from itertools import islice, chain, pairwise
from functools import cache

import kenlm
import multiprocessing
import torch
from concurrent.futures import ProcessPoolExecutor
from pyctcdecode import build_ctcdecoder
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier

@cache
def create_pool():
    pool = ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork"))
    atexit.register(pool.shutdown)
    return pool

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beams: int = 1, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.empty_ind = 0
        self.ctc_decoder = build_ctcdecoder(self.vocab)
        self.beams = beams

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set(char for char in text if char not in self.char2ind)
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join(self.ind2char[int(ind)] for ind in inds).strip()

    def ctc_decode(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor) -> list[str]:
        """
        CTC decoding

        Args:
            log_probs (torch.Tensor): CPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.

        """
        if self.beams == 1:
            predictions = torch.argmax(log_probs, dim=-1).cpu()
            lengths = log_probs_length.detach().cpu().numpy()
            results = []
            for pred_vec, length in zip(predictions, lengths):
                pred_vec = islice(pred_vec, length)
                pred_vec = map(int, pred_vec)
                pred_vec = (
                    right
                    for left, right in pairwise(chain([-1], pred_vec))
                    if left != right
                )
                pred_vec = (
                    self.ind2char[i]
                    for i in pred_vec
                    if i != self.empty_ind
                )
                results.append(''.join(pred_vec))
            return results

        pool = create_pool()
        return self.ctc_decoder.decode_batch(
            pool,
            log_probs.cpu().detach().numpy(),
            beam_width=self.beams,
        )

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class CTCTextEncoderWithLM(CTCTextEncoder):
    # I have no idea why these values is correct, but I have taken them "as is" from
    # https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26

    def __init__(self, lm_model, unigram_file, **kwargs):
        super().__init__(**kwargs)
        self.kenlm_model = kenlm.Model(lm_model)
        with open(unigram_file) as f:
            self.unigram_list = [
                line.lower()
                for line in filter(None, map(str.strip, f))
            ]

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            lm_model,
            self.unigram_list,
        )
