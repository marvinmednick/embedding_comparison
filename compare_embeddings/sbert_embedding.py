from sentence_transformers import SentenceTransformer
from polls.models import Embedding768
from log_setup import get_logger
from utils import hybrid_token_splitter, ensure_specific_nltk_resources
import warnings
from contextlib import contextmanager
import logging


@contextmanager
def suppress_specific_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length for this model*")
        yield


logger = get_logger(__name__)


class SbertPatentEmbedding():
    def __init__(self, window_percent=0.95, overlap_percent=0.2):
        self.model_name = 'AI-Growth-Lab/PatentSBERTa'
        self.model = SentenceTransformer(self.model_name)
        # self.model.tokenizer.clean_up_tokenization_spaces = True

        self.window_size = self.model.max_seq_length
        self.chunk_size = int(self.window_size * window_percent)
        self.overlap_size = int(self.window_size * overlap_percent)
        ensure_specific_nltk_resources()
        logger.info("Using %s.  Window size is %s  Chunk size %s  Overlap: %s", self.model_name, self.window_size, self.chunk_size, self.overlap_size)

    def __call__(self, input_docs) -> list[list[float]]:
        embeddings = [self.generate_embedding(doc) for doc in input_docs]
        return embeddings

    def generate_embedding(self, document: str) -> list[float]:
        return self.model.encode(document, show_progress_bar=False).tolist()

    def chunk(self, document) -> (list[str], int):
        chunked_text = hybrid_token_splitter(document,
                                             self,
                                             chunk_size_tokens=self.chunk_size,
                                             chunk_overlap_tokens=self.overlap_size)
        return chunked_text, len(chunked_text)

    def tokenize(self, document: str, disable_warning=False):
        old_level = logging.getLogger("transformers.tokenization_utils_base").level
        if disable_warning:
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # No warning on sample size while we are checking tokens
        tokens = self.model.tokenizer(document)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(old_level)  # Restore level

        return tokens['input_ids']

    def get_embed_params(self):
        lookup_params = {
            'name': "PATENT_SBERT",
        }

        defaults = {
            'name': "PATENT_SBERT",
            'size': 768,
            'short_name': 'psbert',
            'description': "Sbert embedding that has been tuned for patents sentanceTransformer model AB1I-Growth-Lab/PatentSBERTa"
        }
        model = Embedding768

        return lookup_params, defaults, model
