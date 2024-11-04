import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from openai_embedding import OpenAIEmbedding  # noqa: E402


embedder = OpenAIEmbedding()

long_text_file = 'test/time_machine_hg_wells.txt'
with open(long_text_file, 'r') as input:
    long_test_text = input.read()
    print(f"Read {len(long_test_text)}")


def test_basic_tokenization(request):
    show_data = request.config.getoption("--show_data")
    text_words = len(long_test_text.split())  # This will print the number of words in the text
    tokens = embedder.tokenize(long_test_text)

    if show_data:
        print()
        print(f"\tassert text_words == {text_words}")
        print(f"\tassert tokens[0] == {tokens[0]}")
        print(f"\tassert tokens[1] == {tokens[1]}")
        print(f"\tassert tokens[100] == {tokens[100]}")
        print(f"\tassert tokens[501] == {tokens[501]}")
        return

    assert text_words == 32386
    assert tokens[0] == 791
    assert tokens[1] == 4212
    assert tokens[100] == 1694
    assert tokens[501] == 315


def test_chunk(request):
    show_data = request.config.getoption("--show_data")
    chunks, num_chunks = embedder.chunk(long_test_text)
    if (show_data):
        # This prints the chunk info data structure if something needs to change
        # run pytest with the -s option and --show_data
        print()
        print("\tchunk_info = {")
        for i, chunk in enumerate(chunks):
            print(f"\t\t{i}:  {{\'chunk_len\': {len(chunk)}, \'chunk_tokens\': {len(embedder.tokenize(chunk))}}},")
        print("\t}")
        return

    chunk_info = {
        0:  {'chunk_len': 32381, 'chunk_tokens': 7359},
        1:  {'chunk_len': 33965, 'chunk_tokens': 7371},
        2:  {'chunk_len': 33277, 'chunk_tokens': 7365},
        3:  {'chunk_len': 33365, 'chunk_tokens': 7390},
        4:  {'chunk_len': 32426, 'chunk_tokens': 7369},
        5:  {'chunk_len': 31508, 'chunk_tokens': 7167},
    }

    assert num_chunks == len(chunk_info)
    for key, value in chunk_info.items():
        assert len(chunks[key]) == value['chunk_len']
        assert len(embedder.tokenize(chunks[key])) == value['chunk_tokens']
