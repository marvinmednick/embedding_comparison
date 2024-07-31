import time
import argparse
import json
start = time.process_time()
i = 0
print(f"start {i} {time.process_time() - start}",flush=True)
i = i + 1
import chromadb
print(f"chromadb {i} {time.process_time() - start}",flush=True)
i = i + 1
import ollama
print(f"ollama {i} {time.process_time() - start}",flush=True)
i = i + 1
# from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
print(f"from chromadb {i} {time.process_time() - start}",flush=True)
i = i + 1
import nltk
print(f"ntlk {i} {time.process_time() - start}",flush=True)
i = i + 1
from nltk.corpus import stopwords
print(f"stopwords {i} {time.process_time() - start}",flush=True)
i = i + 1
from spacy.lang.en import English
print(f"English {i} {time.process_time() - start}",flush=True)
i = i + 1
from spacy.lang.en.stop_words import STOP_WORDS
print(f"STOP_WORDS {i} {time.process_time() - start}",flush=True)
i = i + 1
import spacy
print(f"spacy {i} {time.process_time() - start}",flush=True)
i = i + 1
from nltk.stem import PorterStemmer, WordNetLemmatizer
print(f"nltk.stem {i} {time.process_time() - start}",flush=True)
i = i + 1
from nltk.corpus import wordnet
print(f"wordnet {i} {time.process_time() - start}",flush=True)
i = i + 1
from nltk.tokenize import word_tokenize
print(f"tokenize {i} {time.process_time() - start}",flush=True)
i = i + 1
from sentence_transformers import SentenceTransformer
print(f"Sentance {i} {time.process_time() - start}",flush=True)
i = i + 1

# Download necessary resources

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input_docs: Documents) -> Embeddings:
        embeddings = [self.generate_embedding(doc) for doc in input_docs]
        return embeddings

    def generate_embedding(self, document: str) -> list[float]:
        print(f"embedding {document[0:100]}")
        return self.model.encode(document).tolist()
        # return [0.0] * 4096


def load_nltk(resource):
    try:
        nltk.find(resource)
    except:
        nltk.download(resource)

def download_resources():
    # Download necessary resources
    load_nltk('corpora/wordnet')
    load_nltk('tokenizer/punkt')
    load_nltk('corpora/stopwords')
#    nltk.download('wordnet')
#    nltk.download('punkt')
#    nltk.download('stopwords')


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

class TextPreprocessor():

    def __init__(self):
        print("Loading Spaccy stop words")
        # Initialize spaCy and stop words
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english')).union(STOP_WORDS)

        # Initialize stemmer and lemmatizer
        # self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and remove stop words
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words and token.is_alpha]
        tokens = [self.lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
        
        # Join tokens back into a single string
        processed_text = ' '.join(tokens)
        return processed_text


def load_file_to_collection(filename, collection, id_key, doc_key, meta_config=None, maxrec=None):
    print(f"Loading {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    print(len(data), data[0].keys())
    data_keys = data[0].keys()
    if doc_key not in data_keys:
        raise Exception(f"Invalid doc key {doc_key}; key not found in filename")

    if id_key not in data_keys:
        raise Exception(f"Invalid id key {id_key}; key not found in filename")

    if meta_config is not None:
        metadata = []
        # print(f"Metadata -> {meta_config}")
        for meta_cfg in meta_config:

            print(meta_cfg, type(meta_cfg), type(meta_config))
            for data_rec in data:
                record = {}
                for key, field in meta_cfg.items():
                    record[key] = data_rec[field]
                    # print(f"Added {record} - {key} {field} {data_rec[field]}")

                # print(f"Appending {record}")
                metadata.append(record)

        # print(metadata)
    else:
        metadata = None

    field_lists = dict(zip(data_keys, map(list, zip(*[d.values() for d in data]))))

    collection.upsert(
        documents=field_lists[doc_key],
        embeddings=None,
        ids=field_lists[id_key],
        metadatas=metadata
    )


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")

    # Add required arguments
    # parser.add_argument('--load',
    #                nargs=2,
    #                metavar=('filename', 'collection'),
    #                help='Load data from a file into a collection')
    # parser.add_argument('collection', type=str, help='Name of the collection')
    # parser.add_argument('filename', type=str, help='Name of the file')
    parser.add_argument('--load', action='store_true', help='Flag to indicate if resources should be downloaded')
    parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    parser.add_argument('--temp', action='store_true', help='Flag to indicate to use temporary collection')
    parser.add_argument('--plook', action='store_true', help='Flag to indicate lookup document matches for claims')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')

    # Parse the arguments
    args = parser.parse_args()
    # if args.load:
    #     filename, collection = args.load
    #     print(f'Loading data from {filename} into collection {collection}')
    #     print(f'Additional parameter: {extra_param}')

    # Access the arguments
    # collection = args.collection
    # filename = args.filename

    # Your code logic here
    # print(f"Processing collection: {collection}")
    # print(f"Using file: {filename}")
    print("Starting model loading")

    # client = chromadb.Client()

    # Initialize your custom embedding function
    custom_embedder = CustomEmbeddingFunction('AI-Growth-Lab/PatentSBERTa')
    if args.temp:
        chroma_client = chromadb.Client()
    else:
        chroma_client = chromadb.PersistentClient(path="./chroma1.db")

    collections = {}
    # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
    collections['patents'] = chroma_client.get_or_create_collection(name="patents", embedding_function=custom_embedder, metadata={"hnsw:space": "cosine"})
    print(f"Patents currently includes {collections['patents'].count()} records")
    collections['h264spec'] = chroma_client.get_or_create_collection(name="h264_spec", embedding_function=custom_embedder, metadata={"hnsw:space": "cosine"})
    print(f"H264spect currently includes {collections['h264spec'].count()} records")

    if args.load:
        print("Loading records")
        load_file_to_collection("claims.json",
                                collections['patents'],
                                'claim_id',
                                'claim_text',
                                meta_config=[{'patent_number': 'patent_number', 'claim_number': 'claim_number'}],
                                maxrec=args.maxrec
                                )
        load_file_to_collection("h264.json", collections['h264spec'], 'section_id', 'section_text', meta_config=[{'section_title': 'section_title'}], maxrec=args.maxrec)

    print("Added records (new)")
    # tp = TextPreprocessor()
#    query_texts = [tp.preprocess_text("This is a query document about florida")]

#    print(patents)

    if args.plook:
        patents = collections['patents'].get(include=['embeddings'], limit=args.maxrec)
        print(f"Number of patents {len(patents['ids'])}")

        patent_info = zip(patents['ids'],patents['embeddings'])
        for (id, embed) in patent_info:
            print(f"Processing {id} {embed[0:3]}...")

    if args.test:
        patents = collections['patents'].get(include=[], limit=2)
        print(patents)
        id_list = patents['ids']
        print(f"id list: {id_list}")
        recs = collections['patents'].get(ids=id_list, include=["metadatas"])
        print(recs)

        spec = collections['h264spec'].get(include=[], limit=2, offset=1)
        print(spec)
        id_list = spec['ids']
        print(f"id list: {id_list}")
        recs = collections['h264spec'].get(ids=id_list, include=["metadatas"])
        print(recs)

        query_texts = [("This is a query document about florida")]
        print(f"Querying: {query_texts}")

        results = collections['patents'].query(
            query_texts=query_texts,   # Chroma will embed this for you
            n_results=1   # how many results to return
        )

        print("Patent Results")
        print(results)
        results = collections['h264spec'].query(
            query_texts=query_texts,   # Chroma will embed this for you
            n_results=1   # how many results to return
        )
        print("Spec Results")
        print(results)

    print("main complete")


if __name__ == "__main__":
    main()
    print('exiting')
