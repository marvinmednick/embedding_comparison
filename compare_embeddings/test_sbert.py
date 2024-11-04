import os
import django
from long_test_text import long_test_text

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from sbert_embedding import SbertPatentEmbedding  # noqa: E402


embedder = SbertPatentEmbedding()


def test_basic_tokenization():
    text_words = len(long_test_text.split())  # This will print the number of words in the text
    assert text_words == 3123

    embedder = SbertPatentEmbedding()

    tokens = embedder.tokenize(long_test_text)
    # print(f"Found {len(tokens)} in text")
    # print(f"token 0: {tokens[0]} 1: {tokens[1]} 100: {tokens[100]}  501: {tokens[501]}")
    # check a random sampling of tokens to previously known values for this text block
    # if these fails it would indicate the text or the tokenization process has changed
    assert tokens[0] == 0
    assert tokens[1] == 2978
    assert tokens[100] == 1016
    assert tokens[501] == 5465


def test_chunk(request):
    print(request)
    show_data = request.config.getoption("--show_data")
    print("Show data:",  show_data)
    chunks, num_chunks = embedder.chunk(long_test_text)
    if (show_data):
        # This prints the chunk info data structure if something needs to change
        # run pytest with the -s option and --show_data
        print("chunk_info = {")
        for i, chunk in enumerate(chunks):
            print(f"    {i}:  {{\'chunk_len\': {len(chunk)}, \'chunk_tokens\': {len(embedder.tokenize(chunk))}}},")
        print("}")

    chunk_info = {
        0:  {'chunk_len': 2658, 'chunk_tokens': 449},
        1:  {'chunk_len': 2692, 'chunk_tokens': 457},
        2:  {'chunk_len': 2808, 'chunk_tokens': 473},
        3:  {'chunk_len': 2882, 'chunk_tokens': 471},
        4:  {'chunk_len': 2847, 'chunk_tokens': 474},
        5:  {'chunk_len': 2772, 'chunk_tokens': 467},
        6:  {'chunk_len': 2736, 'chunk_tokens': 460},
        7:  {'chunk_len': 2840, 'chunk_tokens': 477},
        8:  {'chunk_len': 2832, 'chunk_tokens': 482},
        9:  {'chunk_len': 1686, 'chunk_tokens': 293},
    }
    assert num_chunks == 10
    assert num_chunks == len(chunk_info)
    for key, value in chunk_info.items():
        assert len(chunks[key]) == value['chunk_len']
        assert len(embedder.tokenize(chunks[key])) == value['chunk_tokens']
