from sentence_transformers import SentenceTransformer
from polls.models import Embedding768
from log_setup import get_logger
from utils import hybrid_token_splitter, ensure_specific_nltk_resources


logger = get_logger(__name__)


class SbertPatentEmbedding():
    def __init__(self):
        self.model_name = 'AI-Growth-Lab/PatentSBERTa'
        self.model = SentenceTransformer(self.model_name)
        self.window_size = self.model.max_seq_length
        self.chunk_size = int(self.window_size * 0.9)
        self.overlap_size = int(self.window_size * 0.1)
        ensure_specific_nltk_resources()
        logger.info("Using %s.  Window size is %s  Chunk size %s  Overlap: %s", self.model_name, self.window_size, self.chunk_size, self.overlap_size)

    def __call__(self, input_docs) -> list[list[float]]:
        embeddings = [self.generate_embedding(doc) for doc in input_docs]
        return embeddings

    def generate_embedding(self, document: str) -> list[float]:
        return self.model.encode(document, show_progress_bar=False).tolist()

    def chunk(self, document) -> (list[str], int):
        chunked_text = hybrid_token_splitter(document,
                                             self.tokenize,
                                             chunk_size_tokens=self.chunk_size,
                                             chunk_overlap_tokens=self.overlap_size)
        return chunked_text, len(chunked_text)

    def tokenize(self, document: str):
        return self.model.tokenizer(document)

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

