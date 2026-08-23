"""Word similarity based on BERT's pretrained token embedding table."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class SimilarityResult:
    closest_word: str
    candidate1_score: float
    candidate2_score: float


class BertWordSimilarity:
    """Compare words with the static input embeddings from a pretrained BERT."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.model_name = model_name
        # Prefer an existing cache so a deployed server can start without network.
        # On the first run, fall back to downloading the pretrained files.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except (OSError, AttributeError):
            model = AutoModel.from_pretrained(model_name)
        self.embedding_table = model.get_input_embeddings()
        self.embedding_table.eval()

    def word_vector(self, word: str) -> torch.Tensor:
        """Return the mean embedding of a word's BERT WordPiece tokens."""
        word = word.strip()
        if not word:
            raise ValueError("word cannot be empty")

        token_ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"word cannot be tokenized: {word!r}")

        ids = torch.tensor(token_ids, dtype=torch.long)
        with torch.no_grad():
            return self.embedding_table(ids).mean(dim=0)

    def compare(self, candidate1: str, candidate2: str, new_word: str) -> SimilarityResult:
        """Return the candidate whose cosine similarity to new_word is greater."""
        vector1 = self.word_vector(candidate1)
        vector2 = self.word_vector(candidate2)
        target = self.word_vector(new_word)

        score1 = F.cosine_similarity(vector1, target, dim=0).item()
        score2 = F.cosine_similarity(vector2, target, dim=0).item()
        closest = candidate1 if score1 >= score2 else candidate2
        return SimilarityResult(closest, score1, score2)
