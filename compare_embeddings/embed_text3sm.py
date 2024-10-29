import os
import sys
import django
import argparse
from tqdm import tqdm
import tiktoken
from openai import AzureOpenAI
from dotenv import load_dotenv
from utils import create_size_buckets, increment_bucket

load_dotenv()  # take environment variables from .env.


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ClaimEmbedding
from polls.models import DocSection, ModifiedDocSection, SectionEmbedding
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding1536

class OpenAIEmbedding():

    def __init__(self, model="text-embedding-3-small"):
        self.client = AzureOpenAI(
          api_key=os.getenv("AZURE_OPENAI_API_KEY"),
          api_version="2024-02-01",
          azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.input_window_size = 8192

    def generate_embedding(self, text): 
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        token_length = len(self.tokenize(text))
        if token_length <= self.input_window_size:
            return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        else:
            return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
            num_chunks = token_length/8000   # less than 8192 so that there is some margin, since we are splitting the original text and not the
            chars_per_chunk = len(text)/num_chunks
            start = 0
            end = chars_per_chunk
            partial_embeddings = []
            for idx in range(0, num_chunks):
                p_embed = self.client.embeddings.create(input=[text[start:end]], model=self.model).data[0].embedding
                partial_embeddings.append(p_embed)
                end += chars_per_chunk

    def tokenize(self, document: str):
        return self.tokenizer.encode(document)

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


def embed_doc(embedder, modtype, maxrec=None, token_check=False):

    lookup_params, defaults, model = embedder.get_embed_params()

    embedding_type, created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)
    if created:
        print(f"Created new Embedding Type {defaults['name']}")

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
    created_count = 0
    with tqdm(total=num_sections, desc="Processing Sections", unit="section") as pbar:
        for index, sec in enumerate(sections):
            token_length = len(embedder.tokenize(sec.modified_text))

            tl_rec = {
                'id': sec.id,
                'section_id': sec.section.section_id,
                'token_length': token_length,
                'text_length': len(sec.modified_text)
            }
            token_lengths.append(tl_rec)

            increment_bucket(size_buckets, token_length)

            if not token_check:
                sec_embed_ref, created = SectionEmbedding.objects.update_or_create(source=sec, embed_type=embedding_type)
                if created:
                    created_count += 1

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

    print(f"Created {created_count} new embeddings")

    for ln in token_lengths:
        if ln['token_length'] > 8192:
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
    parser.add_argument('content', choices=['claims', 'doc'], help='The name of the file to load.')
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

    embedder = OpenAIEmbedding(model="text-embedding-3-small")

    if args.content == 'claims':
        embed_patent_claims(embedder, args.modtype, args.maxrec, args.tokencheck)
    elif args.content == 'doc':
        embed_doc(embedder, args.modtype, args.maxrec, args.tokencheck)


if __name__ == "__main__":
    main()
