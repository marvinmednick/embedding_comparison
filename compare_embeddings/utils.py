import re
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter

SEARCH_AREA = 0.2
SPLITTERS = [
    "\n\n",
    "\n \n",
    ":\n",
    ": ",
    "\\.\n",
    "\\. ",
    ";\n",
    "; ",
    ",\n",
    ", ",
    "\\.",
    ",",
    "\n",
    " ",
]


def find_best_split_point(astr, reverse_splitters=False):
    center = len(astr) // 2

    if reverse_splitters:
        splitters = SPLITTERS[::-1]
        nearby = int(round(len(astr) * SEARCH_AREA / 4))
    else:
        splitters = SPLITTERS
        nearby = int(round(len(astr) * SEARCH_AREA))

    for splitter in splitters:
        starts = [m.start() for m in re.finditer(splitter, astr)]
        starts.sort(key=lambda x: abs(x - center))

        if starts and abs(starts[0] - center) < nearby:
            return starts[0] + len(splitter.replace("\\.", "."))

    return center


def shorten_text(astr, max_length):
    if len(astr) < max_length:
        return astr
    else:
        str_sample = astr[: int(round(max_length / (0.5 + SEARCH_AREA)))]
        spoint = find_best_split_point(str_sample, reverse_splitters=True)
        return str_sample[:spoint]


def create_size_buckets(sizes):

    size_buckets = {size: 0 for size in sizes}
    size_buckets['size_list'] = sizes
    size_buckets['max'] = 0  # For sizes larger than the largest specified size
    return size_buckets


def increment_bucket(size_buckets, item_size):
    for size in sorted(size_buckets['size_list']):
        if item_size < size:
            size_buckets[size] += 1
            break
    else:
        size_buckets['max'] += 1
    return size_buckets


def check_nltk_resource(resource_name) -> bool:
    """
    Check to see if a resource had been downloaded into
    any of the specified directories
    """
    directories = ['tokenizers', 'taggers', 'corpora']
    for directory in directories:
        try:
            nltk.data.find(f'{directory}/{resource_name}')
            return True
        except LookupError:
            continue
    return False


def ensure_specific_nltk_resources():

    """
    Downloads specific NLTK resources if needed.
    """
    required_resources = [
        'punkt',           # for sentence tokenization
        'punkt_tab',           # for sentence tokenization
        'averaged_perceptron_tagger',  # for POS tagging
        # Add other required resources here
    ]

    for resource in required_resources:
        if not check_nltk_resource(resource):
            print(f"Downloading {resource}...")
            nltk.download(resource)


# def hybrid_token_splitter(text, chunk_size_tokens=1500, chunk_overlap_tokens=20):
#     # Step 1: Split into sentences using NLTK
#     sentences = sent_tokenize(text)
# 
#     # Step 2: Join sentences with a special separator
#     sentence_separator = " <SENT> "
#     text_with_markers = sentence_separator.join(sentences)
# 
#     # Step 3: Create token-based splitter
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=chunk_size_tokens,
#         chunk_overlap=chunk_overlap_tokens,
#         separators=["\n\n", "\n", " <SENT> ", " "]
#     )
# 
#     # Step 4: Split into chunks
#     chunks = text_splitter.create_documents([text_with_markers])
# 
#     # Step 5: Clean up the chunks
#     cleaned_chunks = [
#         chunk.page_content.replace(" <SENT> ", " ") for chunk in chunks
#     ]
# 
#     return cleaned_chunks


def hybrid_token_splitter(text, tokenizer_func, chunk_size_tokens=1500, chunk_overlap_tokens=20):
    # Step 1: Split into sentences using NLTK
    sentences = sent_tokenize(text)

    # Step 2: Join sentences with a special separator
    sentence_separator = " <SENT> "
    text_with_markers = sentence_separator.join(sentences)

    # Step 3: Create token-based splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", " <SENT> ", " "],
        length_function=lambda x: len(tokenizer_func(x))
    )

    # Step 4: Split into chunks
    chunks = text_splitter.create_documents([text_with_markers])

    # Step 5: Clean up the chunks
    cleaned_chunks = [
        chunk.page_content.replace(" <SENT> ", " ") for chunk in chunks
    ]

    return cleaned_chunks
