import re
from string import ascii_lowercase

import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from pyctcdecode import build_ctcdecoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
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

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(self.char2ind[char] for char in text).unsqueeze(0)
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
        return ''.join(self.ind2char[int(ind)] for ind in inds).strip()

    def ctc_decode(self, log_probs: torch.Tensor) -> str:
        """
        CTC decoding

        Args:
            log_probs (torch.Tensor): CPU tensor of shape `(frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.

        """
        return self.ctc_decoder.decode(log_probs)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class CTCBeamSearchTextEncoder(CTCTextEncoder):
    # I have no idea why these values is correct, but I have taken them "as is" from
    # https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26

    def __init__(self, alphabet=None, lm_model_name=None **kwargs):
        super().__init__(alphabet, **kwargs)
        lexicon = None
        tokens = self.vocab
        lm = None

        if lm_model_name:
            self.lm_files = download_pretrained_files(lm_model_name)
            lexicon = self.lm_files.lexicon
            tokens = self.lm_files.tokens
            lm = self.lm_files.lm

        self.beam_search_decoder = ctc_decoder(
            lexicon=lexicon,
            tokens=tokens,
            lm=lm,
            nbest=1,
            beam_size=1500,
            lm_weight=self.LM_WEIGHT,
            word_score=self.WORD_SCORE,
        )

    def ctc_decode(self, log_probs: torch.Tensor) -> str:
        """
        CTC decoding

        Args:
            log_probs (torch.Tensor): CPU tensor of shape `(frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.

        """
        beam_search_result = self.beam_search_decoder([inds])
        beam_search_transcript = ' '.join(beam_search_result[0][0].words).strip()
        return beam_search_transcript
