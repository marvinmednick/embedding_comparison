#!/usr/bin/env python

import os
import sys
import django
import random
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter
from pgvector.django.functions import CosineDistance, FloatField
from django.db.models import F, Value, CharField, BooleanField, Q, Subquery, OuterRef, When, Case
import ndcg
from utils import comma_separated_list, get_embed_model


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Section, PatentClaim, ClaimRelatedSection   # noqa: E402
from polls.models import ModificationType, EmbeddingType    # noqa: E402


# Function to print all fields and their values for a given instance
def print_all_fields(instance):
    for field in instance._meta.get_fields():
        # Check if the field is a regular field (not a related field)
        if not field.is_relation:
            field_name = field.name
            field_value = getattr(instance, field_name, None)
            print(f"{field_name}: {field_value}")


def hellinger_distance(p, q):
    #   # Ensure the vectors are numpy arrays
    #    print(f"P {p}")
    #    print(f"Q {q}")
    p = np.array(p)
    q = np.array(q)
    
    # Calculate the Hellinger distance
    try:
        sqrt_p = np.sqrt(p)
    except Exception as err:
        print(f"Error {err} on sqrt of {p}")
    sqrt_q = np.sqrt(q)
#    print("SQRT P", sqrt_p, " SQRT Q2", sqrt_q)
    calc1 = sqrt_p - sqrt_q
    calc2 = calc1**2
    sum2 = np.sum(calc2)
#    print("P-Q: ", calc1)
#    print("Calc2: ", calc2)
#    print("sum2: ", sum2)
    div2 = sum2 / 2
    result = np.sqrt(div2)
#    print("Div2: ", div2)
#    print("result: ", result)
    return result
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2) / 2)


class ClaimComparison():

    def __init__(self, modtype, embedding_type, section_list=None, maxrec=10, print_detail=False, related_claims=False, use_average=False, use_best=False):
        self.section_list = section_list
        self.use_average = use_average
        self.maxrec = maxrec
        self.mod_type = ModificationType.objects.get(name=modtype).name
        self.embed_type = EmbeddingType.objects.get(short_name=embedding_type)
        self.print_detail = print_detail
        self.related_claims_only = related_claims
        self.use_best = use_best

    def compare_claim(self, claim_id):

        patent_claim = PatentClaim.objects.get(claim_id=claim_id)

        if (embedding_model := get_embed_model(self.embed_type.size)) is None:
            raise ValueError(f"Invalid embedding size: {self.embed_type.size}")

        claim_Q = (
                Q(orig_source_id=patent_claim.id) &
                Q(embed_source='claim') &
                Q(mod_type_name=self.mod_type) &
                Q(embed_type_name=self.embed_type.name)
            )

        if self.use_average:
            claim_Q &= Q(tags__name='average')

        obj = embedding_model.objects.get(claim_Q)

        claim_embedding_vector = obj.embedding_vector

        embedding_Q = Q(embed_source='document') & Q(mod_type_name=self.mod_type) & Q(embed_type_name=self.embed_type.name)

        if self.use_average:
            embedding_Q &= Q(tags__name='average')

        if self.section_list:
            orig_source_id_list = Section.objects.filter(section_id__in=self.section_list).values_list('id', flat=True)
            embedding_Q &= Q(orig_source_id__in=orig_source_id_list)

        document_embeddings = embedding_model.objects.filter(embedding_Q)

        all_related_sections = list(ClaimRelatedSection.objects.filter(claim__claim_id=claim_id).values_list('related_sections', flat=True).first() or [])

        # filter down related sections to only the ones we are looking at
        related_sections = [x for x in all_related_sections if x in self.section_list]

        empty_rec = {
            'rank':  None,
            'cosine_distance': None,
            'distance':  None
        }
        found_ranking = dict.fromkeys(related_sections, False) 
        related_section_rankings = []

        annotated_queryset = document_embeddings.annotate(
            cosine_distance=CosineDistance(F('embedding_vector'), claim_embedding_vector),
            distance=Value(0, FloatField()),
            claim_id=Value(claim_id, output_field=CharField()),
            section_id=Subquery(Section.objects.filter(id=OuterRef('orig_source_id')).values('section_id')[:1]),
            known_related_section=Case(When(section_id__in=related_sections, then=Value(True)),
                                       default=Value(False),
                                       output_field=BooleanField())
        )

        # Annotate your queryset with the custom function result
        for i, obj in enumerate(annotated_queryset):
            # Use the custom function to compute the result
            dist = hellinger_distance(obj.embedding_vector, claim_embedding_vector)
            obj.distance = dist

        result = sorted(list(annotated_queryset.values()), key=lambda x: x['cosine_distance'])

        # if the use best flag is set, take only the top result from multiple chunks and rebuild the list
        if self.use_best:
            already_found = set()
            result = [item for item in result if item['section_id'] not in already_found and not already_found.add(item['section_id'])]

        relevance_list = [1 if d['known_related_section'] else 0 for d in result if 'known_related_section' in d]

        if self.print_detail:
            print(f"{'Rank':<7} {f'Embed {self.embed_type.size} ID':15} {'Section ID':15} {'Cosine Distance':20} {'Distance':20} {'Related':5}")

        for rank, document in enumerate(result[0:self.maxrec], 1):
            chunk_info = f"{document['chunk_number']}/{document['total_chunks']}"
            if document['known_related_section']:
                ranking_rec = {
                    'section_id': document['section_id'],
                    'rank':  rank,
                    'distance':  document['distance'],
                    'cosine_distance':  document['cosine_distance'],
                    'id':  document['id'],
                    'chunk_info':  chunk_info
                }
                related_section_rankings.append(ranking_rec)
                found_ranking[document['section_id']] = True
            if self.print_detail:
                print(f"{rank:7} {document['id']:<15} {document['section_id']:<15} {document['cosine_distance']:<20} {document['distance']:<20} {document['known_related_section']} {chunk_info}")

        if len(related_section_rankings) > 0:
            print(f"Ranking for found related sections for {claim_id}")
            for r in related_section_rankings:
                print(f"{r['section_id']:<20} {r['rank']:<5} Cosine Dist. {r['cosine_distance']:<15.10} {r['id']} {r['chunk_info']}")

            sections_not_found = []
            for key, value in found_ranking.items():
                if not value:
                    sections_not_found.append(key)

            if sections_not_found:
                print(f"Not found: {', '.join(sections_not_found)}")

        else:
            print(f"No related sections defined for {claim_id}  Closest item: {result[0].distance} {result[0].cosine_distance}")

        print(f"NDCG {ndcg.calculate_ndcg_from_list(relevance_list)}")

        return result

    #################################################################
    # Compare Range

    def compare_range(self, start, stop, claim_top_k=10, overall_top_k=100):


        if self.related_claims_only:
            claims_list = list(ClaimRelatedSection.objects.values_list('claim__claim_id', flat=True))
            print(f"Comparing related patent claims  index {start} to {stop}")
        else:
            claims_list = list(PatentClaim.objects.values_list('claim_id', flat=True))
            print(f"Comparing all claims  index {start} to {stop}")

        if start > len(claims_list):
            print(f" !! can't process -- list only has {len(claims_list)} entries") 
            return []

        if stop > len(claims_list):
            stop = len(claims_list)
            print(f" !! list only has {len(claims_list)} entries -- comparing from {start} to {stop}")
            
        claims = claims_list[start:stop]
        combined_result = []
        num_claims = len(claims)
        with tqdm(total=num_claims, desc="Processing Claims", unit="claim") as pbar:

            for claim in claims:
                result = self.compare_claim(claim)[0:claim_top_k]
                combined_result.extend(result)
                combined_result = sorted(combined_result, key=lambda x: x.distance)[0:overall_top_k]
                pbar.update(1)

        return combined_result

    ############################################
    # Compare Rand -- radomly select a set of N claims to review

    def compare_rand(self, num_claims, claim_top_k=10, overall_top_k=100):
        if self.related_claims_only:
            claims_list = list(ClaimRelatedSection.objects.values_list('claim__claim_id', flat=True))
        else:
            claims_list = list(PatentClaim.objects.values_list('claim_id', flat=True))

        random_items = random.sample(claims_list, num_claims)
        combined_result = []

        with tqdm(total=len(random_items), desc="Processing Claims", unit="claim") as pbar:
            for claim in random_items:
                result = self.compare_claim(claim)[0:claim_top_k]
                combined_result.extend(result)
                combined_result = sorted(combined_result, key=lambda x: x.distance)[0:overall_top_k]

                pbar.update(1)

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
    parser.add_argument('--rand', type=int, metavar=('NUMBER'), help='evaluation the distance of N random claims')
    parser.add_argument('--maxrec', type=int, default=10, help='maximum records to display')
    parser.add_argument('--claim_k', '-ck', type=int, default=10, help='maximum records to display')
    parser.add_argument('--all_k', '-ak', type=int, default=100, help='maximum records to display')
    parser.add_argument('--sections', '-sec', type=str, help="Filename of json file with a list of sections to check")
    parser.add_argument('--docsecs', type=comma_separated_list, help="List of comma-separated section numbers")
    parser.add_argument('--detail', action='store_true', help='Print details of claims comparisons')
    parser.add_argument('--average', action='store_true', help='Only compare embeddings which are the average of the chunks avaiable')
    parser.add_argument('--use-best', action='store_true', help='For sections that have multiple chunks, use only the best distance')

    # parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    # parser.add_argument('--results', type=int, default=5, help='Number of sections to return (default 5)')jjjkk

    # Parse the arguments
    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    section_list = None
    if args.docsecs:
        section_list = args.docsecs

    elif args.sections:
        with open(args.sections, 'r') as f:
            section_list = json.load(f)

    compare = ClaimComparison(section_list=section_list,
                              maxrec=args.maxrec,
                              modtype=args.modtype,
                              embedding_type=args.embed_type,
                              print_detail=args.detail,
                              use_average=args.average,
                              use_best=args.use_best,
                              )

    if args.claim:
        _ = compare.compare_claim(args.claim)

    if args.range:
        _ = compare.compare_range(start=args.range[0], stop=args.range[1], claim_top_k=args.claim_k, overall_top_k=args.all_k)
#         print("COMBINED")
#         for i, r in enumerate(result):
#             claim = PatentClaim.objects.get(claim_id=r.claim_id)
#             orig_source = Section.objects.get(pk=r.orig_source_id).section_id
#             print(i, r.claim_id, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.distance, claim.claim_id, claim.related_claim, orig_source)

    if args.rand:
        _ = compare.compare_rand(args.rand, claim_top_k=args.claim_k, overall_top_k=args.all_k)

        # print("RANDOM COMBINED")
        # for i, r in enumerate(result):
        #     claim = PatentClaim.objects.get(claim_id=r.claim_id)
        #     orig_source = Section.objects.get(pk=r.orig_source_id).section_id
        #     print(i, r.claim_id, r.id, r.embed_id, r.embed_source, r.orig_source_id, r.distance, claim.claim_id, claim.related_claim, orig_source)


if __name__ == "__main__":
    main()
    print('exiting')
