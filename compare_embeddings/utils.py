import re
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
from log_setup import setup_logging, get_logger
from django.apps import apps


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

setup_logging()

logger = get_logger(__name__)


def get_embed_model(size):
    model_name = f"Embedding{size}"
    try:
        return apps.get_model('polls', model_name)
    except LookupError:
        return None


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


def old_hybrid_token_splitter(text, tokenizer_func, chunk_size_tokens=1500, chunk_overlap_tokens=20):
    print("sentance")
    # Step 1: Split into sentences using NLTK
    sentences = sent_tokenize(text)

    # Step 2: Join sentences with a special separator
    sentence_separator = " <SENT> "
    text_with_markers = sentence_separator.join(sentences)

    print("text splitter")
    # Step 3: Create token-based splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", " <SENT> ", " "],
        length_function=lambda x: len(tokenizer_func(x))
    )

    print("After RCTSplit")
    # Step 4: Split into chunks
    chunks = text_splitter.create_documents([text_with_markers])
    print("After create documents")

    # Step 5: Clean up the chunks
    cleaned_chunks = [
        chunk.page_content.replace(" <SENT> ", " ") for chunk in chunks
    ]

    print("hybrid return")
    return cleaned_chunks


def split_oversized_sentence(tokenizer, sentence: str, tokens: List[str], max_tokens: int) -> List[Tuple[str, List[str]]]:
    if len(tokens) <= max_tokens:
        return [(sentence, tokens)]

    # Recursive splitting
    mid = len(sentence) // 2
    left_sentence = sentence[:mid]
    right_sentence = sentence[mid:]
    left_tokens = tokenizer.tokenize(left_sentence)
    right_tokens = tokenizer.tokenize(right_sentence)

    result = split_oversized_sentence(tokenizer, left_sentence, left_tokens, max_tokens)

    result.extend(split_oversized_sentence(tokenizer, right_sentence, right_tokens, max_tokens))

    return result


def hybrid_token_splitter(text: str, tokenizer, chunk_size_tokens: int = 460, chunk_overlap_tokens: int = 51) -> List[str]:

    # Step 1: Split into sentences and tokenize
    sentences = sent_tokenize(text)
    logger.trace("After sent tokenizer, there are %d sentences", len(sentences))
    for i, sent in enumerate(sentences):
        logger.trace("SENTENCE %d START - Len %d", i, len(sent))
        logger.trace(sent)
        logger.trace("SENTENCE %d END", i)
    tokenized_sentences: List[Tuple[str, List[str]]] = []

    for sentence in sentences:
        # at this point the size of the sentences aren't known and may be larger that limit (that is what
        # the code is looking for) so disable any warnings  during this call
        tokens = tokenizer.tokenize(sentence, disable_warning=True)

        if len(tokens) > chunk_size_tokens:
            logger.trace('Splitting sentance with %d tokens', len(tokens))
            tokenized_sentences.extend(split_oversized_sentence(tokenizer, sentence, tokens, chunk_size_tokens/2))
        else:
            tokenized_sentences.append((sentence, tokens))

    logger.trace("After check for large sentences there are  %d sentences", len(tokenized_sentences))
    for i, sent in enumerate(tokenized_sentences):
        logger.trace("TOKENIZED SENTENCE %d START - Len %d token len %d ", i, len(sent[0]), len(sent[1]))
        logger.trace(sent[0])
        logger.trace("TOKENIZED SENTENCE %d END", i)
    # Step 2: Create chunks with overlap at the beginning
    chunks: List[str] = []
    current_chunk: List[Tuple[str, List[str]]] = []
    current_chunk_tokens = 0

    for sentence, tokens in tokenized_sentences:
        token_count = len(tokens)
        if current_chunk_tokens + token_count > chunk_size_tokens and current_chunk:
            # Save current chunk
            chunks.append(" ".join(sentence for sentence, _ in current_chunk))

            # Prepare overlap for next chunk
            overlap_chunk: List[Tuple[str, List[str]]] = []
            overlap_tokens = 0
            for s, t in reversed(current_chunk):
                if overlap_tokens + len(t) <= chunk_overlap_tokens:
                    overlap_chunk.insert(0, (s, t))
                    overlap_tokens += len(t)
                else:
                    break

            # Start new chunk with overlap
            current_chunk = overlap_chunk
            current_chunk_tokens = overlap_tokens

        logger.trace("Overlap with %d", current_chunk_tokens)
        for i, chunk in enumerate(current_chunk):
            logger.trace("Overlap %d tklen: %d: %s", i, len(chunk[1]), chunk[0])
            for i, chunk in enumerate(current_chunk):
                logger.trace("Overlap %d tklen: %d: %s", i, len(chunk[1]), chunk[0])

        current_chunk.append((sentence, tokens))
        current_chunk_tokens += token_count
        logger.trace("Appending TkLen %d total: %d: %s", len(tokens), current_chunk_tokens, sentence)
        # print(f"Appending TkLen {len(tokens)} total: {current_chunk_tokens}: {sentence}")

    if current_chunk:
        chunks.append(" ".join(sentence for sentence, _ in current_chunk))
        for i, chunk in enumerate(current_chunk):
            logger.trace("Final Chunk %d tklen %d : %s", i, len(chunk[1]),  chunk[0])

    return chunks


def comma_separated_list(arg):
    arglist = [item.strip() for item in arg.split(',')]
    return arglist
