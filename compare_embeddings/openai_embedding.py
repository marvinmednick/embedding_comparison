from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import tiktoken
from polls.models import Embedding1536
from log_setup import get_logger
from utils import hybrid_token_splitter, ensure_specific_nltk_resources

load_dotenv()  # take environment variables from .env.

logger = get_logger(__name__)


class OpenAIEmbedding():

    def __init__(self, model="text-embedding-3-small"):
        self.client = AzureOpenAI(
          api_key=os.getenv("AZURE_OPENAI_API_KEY"),
          api_version="2024-02-01",
          azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.window_size = 8192
        self.chunk_size = int(self.window_size * 0.9)
        self.overlap_size = int(self.window_size * 0.1)
        ensure_specific_nltk_resources()
        logger.info("Using %s. Input window size is %d Chunk Size: %d  Overlap: %d", self.model, self.window_size, self.chunk_size, self.overlap_size)

    def generate_embedding(self, text):
        embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return embedding

    def tokenize(self, document: str, disable_warning=False):
        return self.tokenizer.encode(document)

    def chunk(self, document) -> (list[str], int):
        chunked_text = hybrid_token_splitter(document,
                                             self,
                                             chunk_size_tokens=self.chunk_size,
                                             chunk_overlap_tokens=self.overlap_size)
        return chunked_text, len(chunked_text)

    def get_embed_params(self):
        lookup_params = {
            'name': "OPENAI_TEXT3_SM",
        }

        defaults = {
            'name': "OPENAI_TEXT3_SM",
            'size': 1536,
            'short_name': 'text3_small',
            'description': "Open AI text-embedding-3-small model"
        }
        model = Embedding1536

        return lookup_params, defaults, model
