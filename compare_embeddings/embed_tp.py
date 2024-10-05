#!/usr/bin/env python

import os
import sys
import django
import argparse
from topic_model import TopicModelEmbedding

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ClaimEmbedding
from polls.models import DocSection, SectionForEmbedding, SectionEmbedding
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding32
from utils import find_best_split_point, shorten_text
from tqdm import tqdm


def embed_doc(embedder, modtype, maxrec=None):

    modification_type = ModificationType.objects.get(name=modtype)

    if maxrec is not None:
        sections = SectionForEmbedding.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        sections = SectionForEmbedding.objects.filter(modification_type=modification_type)

    lookup_params = {
        'name': f"TOPIC_MODEL_{embedder.domain.upper()} - {embedder.topic}",
    }

    defaults = {
        'size': 32,
        'short_name': f"tpm_{embedder.model_short_name}_{embedder.topic}",
        'description': f"{embedder.domain} topic model: ({embedder.topic}) {embedder.model_name}",
    }

    embedding_type, created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

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
                'embed_type_short_name': embedding_type.short_name,
                'mod_type_name': sec_embed_ref.source.modification_type.name,
                'embedding_vector': text_embedding,
            }
            Embedding32.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)

    return


def embed_patent_claims(embedder, modtype, maxrec=None):

    params = {
        'name': f"TOPIC_MODEL_{embedder.domain.upper()} - {embedder.topic}",
    }

    defaults = {
        'size': 32,
        'short_name': f"tpm_{embedder.model_short_name}_{embedder.topic}",
        'description': f"{embedder.domain} topic model: ({embedder.topic}) {embedder.model_name}",
    }
    embedding_type, _created = EmbeddingType.objects.get_or_create(**params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        claims = ClaimForEmbedding.objects.filter(modification_type=modification_type)

    num_claims = claims.count()
    with tqdm(total=num_claims, desc="Processing Claims", unit="claim") as pbar:
        for index, claim in enumerate(claims):
            claim_embed_ref, _created = ClaimEmbedding.objects.update_or_create(source=claim, embed_type=embedding_type)
            #print(f"Processing {claim} - {claim_embed_ref.id} Created: {_created}")
            text_embedding = embedder.generate_embedding(claim.modified_text)
            params = {
                'embed_source': 'claim',
                'embed_id': claim_embed_ref.id,
            }
            defaults = {
                'source_id': claim_embed_ref.source.id,
                'orig_source_id': claim_embed_ref.source.claim.id,
                'embed_type_name': embedding_type.name,
                'embed_type_shortname': embedding_type.short_name,
                'mod_type_name': claim_embed_ref.source.modification_type.name,
                'embedding_vector': text_embedding,
            }
            #print("Params", params)
            #print("Defaults", defaults)
            embed_ref, _created = Embedding32.objects.update_or_create(defaults=defaults, **params)
            #print(f"Embed ref {embed_ref}  Created: {_created}")
            pbar.update(1)

    return


def main():

    mod_types = ModificationType.objects.values_list('name', flat=True)

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    parser.add_argument('content', choices=['claims', 'doc'], help='The name of the file to load.')
    parser.add_argument('modtype', choices=mod_types, help='The text modificaiton to apply.')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--model', type=str, default="Digital Video Technology", help='Top Level Topic Model to use')

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    # topic model requires a document to setup the model (and in particular
    # select a sub model based the entire document -- will use the entire document
    # for this selection even if max rec is set
    modification_type = ModificationType.objects.get(name=args.modtype)
    print("Gathering Sections")
    sections = [sec.modified_text for sec in SectionForEmbedding.objects.filter(modification_type=modification_type)]
    print("Creating Embbeder Model")
    embedder = TopicModelEmbedding(args.model, sections)

    if args.content == 'claims':
        embed_patent_claims(embedder, args.modtype, args.maxrec)
    elif args.content == 'doc':
        embed_doc(embedder, args.modtype, args.maxrec)


if __name__ == "__main__":
    main()
    print('exiting')
