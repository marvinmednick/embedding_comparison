#!/usr/bin/env python

import os
import sys
import re
import django
import argparse
import json
# from django.utils import timezone
from pprint import pprint

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Document, DocSection, Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ModificationType, ClaimRelatedSection
from polls.models import SectionForEmbedding


def load_related_sections(filename, maxrec=None):

    table_match =  re.compile("Table\s+(?P<id>[0-9A-Z]+-\d+)")
    figure_match = re.compile("Figure\s+(?P<id>[0-9A-Z]+-\d+)")

    print(f"Loading claims and related sections from {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    for record in data:
        claim_id = record['claim_id']
        table_list = []
        figure_list = []
        section_list = []
        for sec in record['sections']:
            if tbl_match := table_match.match(sec):
                table_list.append(tbl_match['id'])

            elif fig_match := figure_match.match(sec):
                figure_list.append(fig_match['id'])

            else:
                section_list.append(sec)

        try:
            claim_ref = PatentClaim.objects.get(claim_id=claim_id)
            # print(f"Claim: {claim_id} ({claim_ref.id})  Secs: {section_list}  Tables: {table_list}  Figures: {figure_list}")
            ClaimRelatedSection.objects.update_or_create(
                claim=claim_ref,
                related_sections=section_list,
                related_tables=table_list,
                related_figures=figure_list
            )
        except:
            print(f"Cound not find claim in DB Claim {claim_id}")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")

    parser.add_argument('filename', type=str, help='The name of the file to load.')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    print(f"Args: {args}")

    load_related_sections(args.filename, args.maxrec)


if __name__ == "__main__":
    main()
    print('exiting')

