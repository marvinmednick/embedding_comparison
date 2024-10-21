#!/usr/bin/env python

import os
import sys
import django
import random
import json
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter
from pgvector.django.functions import CosineDistance
from django.db.models import F, Value, CharField, BooleanField, Q, Subquery, OuterRef, When, Case


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import DocSection, PatentClaim, ClaimRelatedSection
from polls.models import ModificationType, EmbeddingType
from polls.models import Embedding768, Embedding32


def compare_claim(claim_id, mod_type, embedding_type, section_list=None, maxrec=10):

    patent_claim = PatentClaim.objects.get(claim_id=claim_id)
    embed_type = EmbeddingType.objects.get(short_name=embedding_type)
    mod_type = ModificationType.objects.get(name=mod_type).name

    if embed_type.size == 768:
        embedding_model = Embedding768

    elif embed_type.size == 32:
        embedding_model = Embedding32
    else:
        raise ValueError("Invalid embedding size: {embed_type.size}")

    obj = embedding_model.objects.get(orig_source_id=patent_claim.id,
                                      embed_source='claim',
                                      mod_type_name=mod_type,
                                      embed_type_name=embed_type.name)

    # print(obj.mod_type_name, obj.embed_type_name)
    claim_embedding_vector = obj.embedding_vector

    # print(f"Vector is {embedding.embedding_vector[0:20]} Vector size {len(embedding.embedding_vector)}")

    embedding_Q = Q(embed_source='document') & Q(mod_type_name=mod_type) & Q(embed_type_name=embed_type.name)

    if section_list:
        orig_source_id_list = DocSection.objects.filter(section_id__in=section_list).values_list('id', flat=True)
        embedding_Q &= Q(orig_source_id__in=orig_source_id_list)

    document_embeddings = embedding_model.objects.filter(embedding_Q)

    # print(f"Number of document embeddings {document_embeddings.count()}")

    # for document in document_embeddings:
    #     print(f"Document ID: {document.id}, {document.embed_source} Embed_id: {document.embed_id} Source id: {document.source_id} Orig {document.orig_source_id}")

    related_sections = list(ClaimRelatedSection.objects.filter(claim__claim_id=claim_id).values_list('related_sections', flat=True).first() or [])

    rankings = dict.fromkeys(related_sections, None)



    annotated_queryset = document_embeddings.annotate(
        cosine_distance=CosineDistance(F('embedding_vector'), claim_embedding_vector),
        claim_id=Value(claim_id, output_field=CharField()),
        section_id=Subquery(DocSection.objects.filter(id=OuterRef('orig_source_id')).values('section_id')[:1]),
        known_related_section=Case(When(section_id__in=related_sections, then=Value(True)),
                                   default=Value(False),
                                   output_field=BooleanField())
    )

    print(f"{'Rank':<7} {f'Embed {embed_type.size} ID':15} {'Section ID':15} {'Cosine Distance':20}{'Related':5}")

    result = annotated_queryset.order_by('cosine_distance')
    for rank, document in enumerate(result[0:maxrec], 1):
        if document.known_related_section:
            rankings[document.section_id] = rank
        print(f"{rank:7} {document.id:<15} {document.section_id:<15} {document.cosine_distance:<20} {document.known_related_section}")

    # Sort the items, putting None values at the end
    ranking_info = sorted(rankings.items(), key=lambda x: (x[1] is None, x[1]))

    # Print the sorted items
    for key, value in ranking_info:
        if value is None:
            print(f"{key}: not found")
        else:
            print(f"{key}: {value}")

    return result


#################################################################
# Compare Range

def compare_range(start, stop, mod_type, embed_type, claim_top_k=10, overall_top_k=100):
    print(f"Comparing patents with Id {start} to {stop}")

    claims = PatentClaim.objects.filter(id__range=(start, stop))
    combined_result = []
    num_claims = len(claims)
    with tqdm(total=num_claims, desc="Processing Claims", unit="claim") as pbar:

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
            pbar.update(1)
    return combined_result


def compare_rand(num_claims, embed_type, mod_type="Unmodified", claim_top_k=10, overall_top_k=100):
    queryset = list(PatentClaim.objects.all())
    random_items = random.sample(queryset, num_claims)
    combined_result = []

    with tqdm(total=len(random_items), desc="Processing Claims", unit="claim") as pbar:
        for claim in random_items:
            result = compare_claim(claim, mod_type, embed_type)[0:claim_top_k]
            combined_result.extend(result)
            combined_result = sorted(combined_result, key=lambda x: x.cosine_distance)[0:overall_top_k]

            pbar.update(1)

    return combined_result


def comma_separated_list(arg):
    arglist =  [item.strip() for item in arg.split(',')]
    return arglist


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
    parser.add_argument('--rand', type=int, metavar=('NUMBER'), help='evaluation the distance of N random claims')
    parser.add_argument('--maxrec', type=int, default=10, help='maximum records to display')
    parser.add_argument('--claim_k', '-ck', type=int, default=10, help='maximum records to display')
    parser.add_argument('--all_k', '-ak', type=int, default=100, help='maximum records to display')
    parser.add_argument('--sections', '-sec', type=str, help="Filename of json file with a list of sections to check")
    parser.add_argument('--docsecs', type=comma_separated_list, help="List of comma-separated section numbers")

    # parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    # parser.add_argument('--results', type=int, default=5, help='Number of sections to return (default 5)')jjjkk

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    section_list = None
    if args.docsecs:
        section_list = args.docsecs

    elif args.sections:
        with open(args.sections, 'r') as f:
            section_list = json.load(f)

    if args.claim:
        result = compare_claim(args.claim, args.modtype, args.embed_type, section_list=section_list, maxrec=args.maxrec)

    if args.range:
        result = compare_range(start=args.range[0], stop=args.range[1], mod_type=args.modtype, embed_type=args.embed_type, claim_top_k=args.claim_k,
                               overall_top_k=args.all_k)
        print("COMBINED")
        for i, r in enumerate(result):
            claim = PatentClaim.objects.get(claim_id=r.claim_id)
            orig_source = DocSection.objects.get(pk=r.orig_source_id).section_id
            print(i, r.claim_id, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.cosine_distance, claim.claim_id, claim.related_claim, orig_source)

    if args.rand:
        print(f"Random {args.rand} {type(args.rand)}")
        result = compare_rand(args.rand, embed_type=args.embed_type, mod_type=args.modtype, claim_top_k=args.claim_k, overall_top_k=args.all_k)

        print("RANDOM COMBINED")
        for i, r in enumerate(result):
            claim = PatentClaim.objects.get(claim_id=r.claim_id)
            orig_source = DocSection.objects.get(pk=r.orig_source_id).section_id
            print(i, r.claim_id, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.cosine_distance, claim.claim_id, claim.related_claim, orig_source)


if __name__ == "__main__":
    main()
    print('exiting')
