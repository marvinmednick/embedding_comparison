import os
import sys
import django
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import create_size_buckets, increment_bucket
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ClaimEmbedding
from polls.models import DocSection, ModifiedDocSection, SectionEmbedding, ModifiedSectionChunk
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding768


class SbertPatentEmbedding():
    def __init__(self):
        self.model_name = 'AI-Growth-Lab/PatentSBERTa'
        self.model = SentenceTransformer(self.model_name)
        print(f"Using {self.model_name}.  Window size is {self.model.max_seq_length}")

    def __call__(self, input_docs) -> list[list[float]]:
        embeddings = [self.generate_embedding(doc) for doc in input_docs]
        return embeddings

    def generate_embedding(self, document: str) -> list[float]:
        return self.model.encode(document).tolist()

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
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)


def hybrid_token_splitter(text, chunk_size_tokens=1500, chunk_overlap_tokens=20):
    # Step 1: Split into sentences using NLTK
    sentences = sent_tokenize(text)
    
    # Step 2: Join sentences with a special separator
    sentence_separator = " <SENT> "
    text_with_markers = sentence_separator.join(sentences)
    
    # Step 3: Create token-based splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", " <SENT> ", " "]
    )
    
    # Step 4: Split into chunks
    chunks = text_splitter.create_documents([text_with_markers])
    
    # Step 5: Clean up the chunks
    cleaned_chunks = [
        chunk.page_content.replace(" <SENT> ", " ") for chunk in chunks
    ]
    
    return cleaned_chunks


def chunk_doc(embedder, modtype, maxrec=None, token_check=False):

    lookup_params, defaults, model = embedder.get_embed_params()

    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        sections = ModifiedDocSection.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        sections = ModifiedDocSection.objects.filter(modification_type=modification_type)

    num_sections = sections.count()
    with tqdm(total=num_sections, desc="Processing Sections", unit="section") as pbar:
        for index, sec in enumerate(sections):

            # Delete Embeddings that are for modified_text entry and embedding type (?)
            existing_embeddings = model.objects.filter(source_id=sec.id, embed_source='document')
            existing_embeddings.delete()

            # Delete ModifiedSectionChunk records for this modified section
            existing_chunks = ModifiedSectionChunk.objects.filter(modified_section__id=sec.id)
            existing_chunks.delete()

            chunked_text = hybrid_token_splitter(sec.modified_text, chunk_size_tokens=500, chunk_overlap_tokens=40)
            total_chunks = len(chunked_text)

            lookup_params = {
                'source': sec,
                'embed_type': embedding_type
            }
            defaults = {
                'total_chunks': total_chunks
            }
            sec_embed_ref, _created = SectionEmbedding.objects.update_or_create(**lookup_params, defaults=defaults)

            chunk_embed_list = []
            embed_record = {
                'embed_source': 'document',
                'embed_id': sec_embed_ref.id,
                'source_id': sec_embed_ref.source.id,
                'orig_source_id': sec_embed_ref.source.section.id,
                'embed_type_name': embedding_type.name,
                'mod_type_name': sec_embed_ref.source.modification_type.name,
                'embed_type_shortname': embedding_type.short_name,
                'embedding_vector': None,
                'chunk_number': None,
                'total_chunks': total_chunks,
            }
            for chunk_num, chunk in enumerate(chunked_text, 1):

                # if there is more than one chunk, then create separate chunk records
                # otherwise the entire text is the chunk and the source text is the mod_text field in the modfiedsection record
                if total_chunks > 1:
                    chunk_info = {
                        'section_embedding': sec_embed_ref,
                        'modified_section': sec,
                        'chunk_number': chunk_num,
                        'chunk_text': chunk
                    }
                    ModifiedSectionChunk.objects.create(**chunk_info)

                chunk_embedding = embedder.generate_embedding(chunk)

                embed_record['chunk_number'] = chunk_num
                embed_record['embedding_vector'] = chunk_embedding

                chunk_embed_list.append(chunk_embedding)

                model.objects.create(**embed_record)

            mean_embedding = np.mean([chunk_embedding for chunk_embedding in chunk_embed_list], axis=0)
            full_text_embedding = embedder.generate_embedding(sec.modified_text)

            embed_record['chunk_number'] = 98
            embed_record['embedding_vector'] = full_text_embedding

            model.objects.create(**embed_record)

            embed_record['chunk_number'] = 99
            embed_record['embedding_vector'] = mean_embedding

            model.objects.create(**embed_record)
            pbar.update(1)


def embed_doc(embedder, modtype, maxrec=None, token_check=False):

    ensure_specific_nltk_resources()

    lookup_params, defaults, model = embedder.get_embed_params()

    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        sections = ModifiedDocSection.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        sections = ModifiedDocSection.objects.filter(modification_type=modification_type)

    token_lengths = []

    # Create buckes for counting the size distribution of the token lengths
    specified_sizes = [16, 256, 512, 1024, 2048, 4096, 8192, 16384]
    size_buckets = create_size_buckets(specified_sizes)

    num_sections = sections.count()
    with tqdm(total=num_sections, desc="Processing Sections", unit="section") as pbar:
        for index, sec in enumerate(sections):
            tokens = embedder.tokenize(sec.modified_text)
            token_length = len(tokens['input_ids'])

            tl_rec = {
                'id': sec.id,
                'section_id': sec.section.section_id,
                'token_length': token_length,
                'text_length': sec.modified_text
            }
            token_lengths.append(tl_rec)

            increment_bucket(size_buckets, token_length)

            if not token_check:
                sec_embed_ref, _created = SectionEmbedding.objects.update_or_create(source=sec, embed_type=embedding_type)

                text_embedding = embedder.generate_embedding(sec.modified_text)
                params = {
                    'embed_source': 'document',
                    'embed_id': sec_embed_ref.id,
                }
                defaults = {
                    'source_id': sec_embed_ref.source.id,
                    'orig_source_id': sec_embed_ref.source.section.id,
                    'embed_type_name': embedding_type.name,
                    'mod_type_name': sec_embed_ref.source.modification_type.name,
                    'embed_type_shortname': embedding_type.short_name,
                    'embedding_vector': text_embedding,
                }
                model.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)

    for ln in token_lengths:
        print(f"Section {ln['section_id']} ({ln['id']})   len: {ln['token_length']}")

    print("Token sizes by range:")
    for key in size_buckets['size_list']:
        print(f"{key}: {size_buckets[key]}")

    print(f"max: {size_buckets['max']}")


def embed_patent_claims(embedder, modtype, maxrec=None, token_check=False):

    lookup_params, defaults, model = embedder.get_embed_params()
    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)

    num_claims = claims.count()
    with tqdm(total=num_claims, desc="Processing Claims", unit="claim") as pbar:
        for index, claim in enumerate(claims):
            claim_embed_ref, _created = ClaimEmbedding.objects.update_or_create(source=claim, embed_type=embedding_type)
            text_embedding = embedder.generate_embedding(claim.modified_text)
            params = {
                'embed_source': 'claim',
                'embed_id': claim_embed_ref.id,
            }
            defaults = {
                'source_id': claim_embed_ref.source.id,
                'orig_source_id': claim_embed_ref.source.claim.id,
                'embed_type_name': embedding_type.name,
                'mod_type_name': claim_embed_ref.source.modification_type.name,
                'embed_type_shortname': 'psbert',
                'embedding_vector': text_embedding,
            }
            model.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)


def main():

    mod_types = ModificationType.objects.values_list('name', flat=True)

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    parser.add_argument('content', choices=['claims', 'doc', 'docchunk'], help='The name of the file to load.')
    parser.add_argument('modtype', choices=mod_types, help='The text modificaiton to apply.')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--tokencheck', '-tc', action='store_true', help='Perform a token count check only')

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    embedder = SbertPatentEmbedding()

    if args.content == 'claims':
        embed_patent_claims(embedder, args.modtype, args.maxrec, args.tokencheck)
    elif args.content == 'doc':
        embed_doc(embedder, args.modtype, args.maxrec, args.tokencheck)
    elif args.content == 'docchunk':
        chunk_doc(embedder, args.modtype, args.maxrec, args.tokencheck)


if __name__ == "__main__":
    main()
