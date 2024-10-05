#!/usr/bin/env python

import os
import sys
import django
from argparse import ArgumentParser, RawTextHelpFormatter
from pgvector.django.functions import CosineDistance
from django.db.models import F, Value, CharField

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Document, DocSection, Patent, PatentClaim, ClaimElement
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding768, Embedding32


def compare_claim(claim_id, mod_type, embedding_type):

    patent_claim = PatentClaim.objects.get(claim_id=claim_id)
    embed_type = EmbeddingType.objects.get(short_name=embedding_type)
    mod_type = ModificationType.objects.get(name=mod_type).name

    if embed_type.size == 768:
        embedding_model = Embedding768

    elif embed_type.size == 32:
        embedding_model = Embedding32
    else:
        raise ValueError("Invalid embedding size: {embed_type.size}")

    # print(f"Size {embedding_model.objects.all().count()} - claim id = {claim_id} patent_claim_id: {patent_claim.id} <{mod_type}>")

    # get the claim type
    # claim = PatentClaim.objects.get(claim_id)
    # get the embedding for this claim
    # print(f"Searching for {claim_id} with {mod_type} and {embed_type.name} {embed_type.size}")
    obj = embedding = embedding_model.objects.get(orig_source_id=patent_claim.id, embed_source='claim', mod_type_name=mod_type, embed_type_name=embed_type.name)
    # print(obj.mod_type_name, obj.embed_type_name)
    claim_embedding_vector = obj.embedding_vector

    #print(f"Vector is {embedding.embedding_vector[0:20]} Vector size {len(embedding.embedding_vector)}")

    document_embeddings = embedding_model.objects.filter(embed_source='document', mod_type_name=mod_type, embed_type_name=embed_type.name)
    #print(f"Number of document embeddings {document_embeddings.count()}")

    # for document in document_embeddings:
    #     print(f"Document ID: {document.id}, {document.embed_source} Embed_id: {document.embed_id} Source id: {document.source_id} Orig {document.orig_source_id}")

    annotated_queryset = document_embeddings.annotate(
        cosine_distance=CosineDistance(F('embedding_vector'), claim_embedding_vector),
        claim_id=Value(claim_id, output_field=CharField())
    )

    # all_fields = annotated_queryset.first().__dict__

    # for field, value in all_fields.items():
    #     if not field.startswith('_'):  # Exclude private fields
    #         print(f"{field}: {value}")

    return annotated_queryset.order_by('cosine_distance')


def compare_range(start, stop, mod_type, embed_type, claim_top_k=10,overall_top_k=100):
    print(f"Comparing patents with Id {start} to {stop}")

    claims = PatentClaim.objects.filter(id__range=(start, stop))
    combined_result = []
    for claim in claims:
        # print(f"Checking Claim {claim.claim_id}")
        result = compare_claim(claim, mod_type, embed_type)[0:claim_top_k]
#         all_fields = result.first().__dict__
# 
#         for field, value in all_fields.items():
#             if not field.startswith('_'):  # Exclude private fields
#                 print(f"{field}: {value}")
# 
        combined_result.extend(result)
        combined_result = sorted(combined_result, key=lambda x: x.cosine_distance)[0:overall_top_k]
        # for i, r in enumerate(result):
        #     print(i, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.cosine_distance)

    return combined_result


def main():

    mod_types = ModificationType.objects.values_list('name', flat=True)
    embed_types = EmbeddingType.objects.values_list('short_name', flat=True)
    options = EmbeddingType.objects.values_list('short_name', 'name')
    # print("OPTIONS", options)
    embed_type_help_string = "Embedding Type to Use\n" + "\n".join(f"  {opt[0]:<20}  {opt[1]}" for opt in options)

    # Create the argument parser
    parser = ArgumentParser(description="Process a collection and filename.", formatter_class=RawTextHelpFormatter)
    parser.add_argument('embed_type', choices=embed_types, help=embed_type_help_string)
    parser.add_argument('--claim', type=str, help='claim id of patent claim to find distances')
    parser.add_argument('--modtype', default="Unmodified", choices=mod_types, help='Modification type to use')
    parser.add_argument('-r', '--range', nargs=2, type=int, metavar=('START', 'STOP'),
                        help='Specify start and stop claim indices')
    # parser.add_argument('docname', type=str, nargs='?', help='Optional document name. Will default to filename if not specfied')
    parser.add_argument('--maxrec', type=int, default=10, help='maximum records to display')

    # parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    # parser.add_argument('--results', type=int, default=5, help='Number of sections to return (default 5)')jjjkk

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    if args.claim:
        result = compare_claim(args.claim, args.modtype, args.embed_type)
        for document in result[0:args.maxrec]:
            print(f"Document ID: {document.id}, {document.embed_source} Embed_id: {document.embed_id} Source id: {document.source_id} Orig {document.orig_source_id} Cosine Distance: {document.cosine_distance}")

    if args.range:
        result = compare_range(start=args.range[0], stop=args.range[1], mod_type=args.modtype, embed_type=args.embed_type)
        print("COMBINED")
        for i, r in enumerate(result):
            claim = PatentClaim.objects.get(claim_id=r.claim_id)
            print(i, r.claim_id, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.cosine_distance, claim.claim_id, claim.related_claim)


if __name__ == "__main__":
    main()
    print('exiting')
