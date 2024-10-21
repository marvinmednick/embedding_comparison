import time
import argparse
import csv
import json
import re
cur = start = time.process_time()
i = 0
print(f"start {i} {time.process_time() - start}",flush=True)
i = i + 1
last = cur
import chromadb
cur = time.process_time()
print(f"chromadb {i} {cur - start:.3f}  {cur-last:.3f}",flush=True)
i = i + 1
# import ollama
# last = cur
# cur = time.process_time()
# print(f"ollama {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
# i = i + 1
# from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
last = cur
cur = time.process_time()
print(f"from chromadb {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
import nltk
last = cur
cur = time.process_time()
print(f"ntlk {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from nltk.corpus import stopwords
last = cur
cur = time.process_time()
print(f"stopwords {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from spacy.lang.en import English
last = cur
cur = time.process_time()
print(f"English {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from spacy.lang.en.stop_words import STOP_WORDS
last = cur
cur = time.process_time()
print(f"STOP_WORDS {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
import spacy
last = cur
cur = time.process_time()
print(f"spacy {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from nltk.stem import PorterStemmer, WordNetLemmatizer
last = cur
cur = time.process_time()
print(f"nltk.stem {i} {cur - start:.3f} {cur-last}",flush=True)
i = i + 1
from nltk.corpus import wordnet
last = cur
cur = time.process_time()
print(f"wordnet {i:.3f} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from nltk.tokenize import word_tokenize
last = cur
cur = time.process_time()
print(f"tokenize {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
i = i + 1
from sentence_transformers import SentenceTransformer
last = cur
cur = time.process_time()
print(f"Sentance {i} {cur - start:.3f} {cur-last:.3f}",flush=True)
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


def load_claims(filename,collection, maxrec=None):
    pattern =  re.compile("(?P<country>[A-Z]{2})(?P<number>\d+)(?P<key>[AB][12]*)_(?P<claim>\d+)")

    print(f"Loading patents from {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    # remove the key code e.g. B2 from the claim_id
    for r in data:
        r['claim_id'] = re.sub(pattern, r"\g<country>\g<number>_\g<claim>", r['claim_id'])

    load_data_to_collection(data,
                    collection,
                    'claim_id',
                    'claim_text',
                    meta_config=[{'patent_number': 'patent_number', 'claim_number': 'claim_number'}],
                    maxrec=maxrec
                    )

def load_h264(filename,collection, maxrec=None):
    print(f"Loading spec {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    load_data_to_collection(data, collection, 'section_id', 'section_text', meta_config=[{'section_title': 'section_title'}], maxrec=maxrec)


def load_data_to_collection(data, collection, id_key, doc_key, meta_config=None, maxrec=None):
    print(len(data), data[0].keys())
    data_keys = data[0].keys()
    if doc_key not in data_keys:
        raise Exception(f"Invalid doc key {doc_key}; key not found in filename")
\
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
    parser.add_argument('--delete', type=str, default=None, help='Delete specfied collection')
    parser.add_argument('--load', action='store_true', help='Flag to indicate if resources should be downloaded')
    parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    parser.add_argument('--temp', action='store_true', help='Flag to indicate to use temporary collection')
    parser.add_argument('--list', action='store_true', help='List chromadb collections')
    parser.add_argument('--plook', action='store_true', help='Flag to indicate lookup document matches for claims')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--results', type=int, default=5, help='Number of sections to return (default 5)')

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

    if args.list:
        colls = chroma_client.list_collections()
        print(colls)

    if args.delete:
        try:
            chroma_client.delete_collection(name=args.delete)
            print(f"Deleted collecton {args.delete}")
        except ValueError as e:
            print(f"Could not delete {args.delete} ({e})")

        exit(0)

    collections = {}
    # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
    collections['patents'] = chroma_client.get_or_create_collection(name="patents", embedding_function=custom_embedder, metadata={"hnsw:space": "cosine"})
    print(f"Patents currently includes {collections['patents'].count()} records")
    collections['h264spec'] = chroma_client.get_or_create_collection(name="h264_spec", embedding_function=custom_embedder, metadata={"hnsw:space": "cosine"})
    h264spec_len = collections['h264spec'].count()
    print(f"H264spect currently includes {h264spec_len} records")



    if args.load:
        print("Loading records")
        load_claims("claims.json", collections['patents'], args.maxrec)
        load_h264("h264.json", collections['h264spec'], args.maxrec)

    print("Added records (new)")
    # tp = TextPreprocessor()
#    query_texts = [tp.preprocess_text("This is a query document about florida")]

#    print(patents)

    if args.plook:
        h264_sections = set(collections['h264spec'].get(include=[])['ids'])
        patents = collections['patents'].get(include=['embeddings'], limit=args.maxrec)
        numpats = len(patents['ids'])
        print(f"Length of h264 {collections['h264spec'].count()}")
        print(f"Number of patents {numpats}")
        if numpats < 20:
            print(f"Patent List: {patents['ids']}")

        patent_info = list(zip(patents['ids'], patents['embeddings']))
        patent_results = []
        num_results = h264spec_len
        if args.results < num_results:
            num_results = args.results

        key_sections = {}
        with open("US_H264_patents.csv", "r", encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                print(f"Original Row {row}")
                patent = row['patent'].replace(",", "")
                claim_id = row['country'] + patent + "_" + row['claim']
                section_list = [sec for sec in list(map(str.strip, row['sections'].split(","))) if sec in h264_sections]
                # convert Ordered dict to dict
                print(f"Claim {claim_id} adding {section_list}")
                key_sections[claim_id] = list(section_list)

        with open("US_H264_patents.json", "w") as outfile:
            json.dump(key_sections, outfile, indent=6)

        # query_embeddings = []
        for (pid, embed) in patent_info:
            print(f"Processing {pid} {embed[0:3]}... Len: {len(embed)}")
            # query_embeddings.append(embed)
            results = collections['h264spec'].query(
                    query_embeddings=embed,
                    n_results=num_results,
                    include=['distances'])

            # print(results)
            combined_result = list(zip(results['ids'], results['distances']))
            # print("Combined Result", combined_result)
            # print("Combined Result by item:")
            # for item in combined_result:
            #     print("Item:", item)

            # print(f"{'ID':>20}{'Distanace':>20}")
            prec = {'claim_id': pid, 'result': [], 'key_sec_result': []}
            if pid in key_sections:
                patent_key_secs = key_sections[pid]
            else:
                patent_key_secs = []
            patent_results.append(prec)
            for (result_list, distance_list) in combined_result:
                for idx, (result_id, distance) in enumerate(zip(result_list, distance_list)):
                    # print(f"{result_id:<20}{distance:>20.8f}")

                    dist_rec = {
                       'rank' : idx,
                       'section':  result_id,
                       'distance': distance
                    }
                    prec['result'].append(dist_rec)
                                         
                    if result_id in patent_key_secs:
                        prec['key_sec_result'].append(dist_rec)

            # print("-"*40)
        with open('plook.json', 'w') as output_file:
            json.dump(patent_results, output_file, indent=6)

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
