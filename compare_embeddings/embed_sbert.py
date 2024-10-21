import os
import sys
import django
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ClaimEmbedding
from polls.models import DocSection, SectionForEmbedding, SectionEmbedding
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding768


class SbertPatentEmbedding():
    def __init__(self):
        self.model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

    def __call__(self, input_docs) -> list[list[float]]:
        embeddings = [self.generate_embedding(doc) for doc in input_docs]
        return embeddings

    def generate_embedding(self, document: str) -> list[float]:
        return self.model.encode(document).tolist()


def embed_doc(modtype, maxrec=None):

    lookup_params = {
        'name': "PATENT_SBERT",
    }

    defaults = {
        'name': "PATENT_SBERT",
        'size': 768,
        'short_name': 'psbert',
        'description': "Sbert embedding that has been tuned for patents sentanceTransformer model AB1I-Growth-Lab/PatentSBERTa"
    }
    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        sections = SectionForEmbedding.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        sections = SectionForEmbedding.objects.filter(modification_type=modification_type)

    embedder = SbertPatentEmbedding()

    num_sections = sections.count()
    with tqdm(total=num_sections, desc="Processing Sections", unit="section") as pbar:
        for index, sec in enumerate(sections):
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
                'embed_type_shortname': 'psbert',
                'embedding_vector': text_embedding,
            }
            Embedding768.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)



def embed_patent_claims(modtype, maxrec=None):

    params = {
        'name': "PATENT_SBERT",
    }

    defaults = {
        'name': "PATENT_SBERT",
        'size': 768,
        'short_name': 'psbert',
        'description': "Sbert embedding that has been tuned for patents sentanceTransformer model AB1I-Growth-Lab/PatentSBERTa"
    }
    embedding_type, _created = EmbeddingType.objects.get_or_create(**params)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)

    embedder = SbertPatentEmbedding()

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
            Embedding768.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)


def main():

    mod_types = ModificationType.objects.values_list('name', flat=True)

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    parser.add_argument('content', choices=['claims', 'doc'], help='The name of the file to load.')
    parser.add_argument('modtype', choices=mod_types, help='The text modificaiton to apply.')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    if args.content == 'claims':
        embed_patent_claims(args.modtype, args.maxrec)
    elif args.content == 'doc':
        embed_doc(args.modtype, args.maxrec)


if __name__ == "__main__":
    main()
