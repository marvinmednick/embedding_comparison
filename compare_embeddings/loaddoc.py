import os 
import sys
import re
import django 
import argparse
import csv
import json
from django.utils import timezone

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Question, Document, DocSection, Patent, PatentClaim, ClaimElement, ClaimForEmbedding, ModificationType
from polls.models import SectionForEmbedding

def load_claims(filename, maxrec=None, related=False, key_related=False, update=False):
    # pattern =  re.compile("(?P<country>[A-Z]{2})(?P<number>\d+)(?P<key>[AB][12]*)_(?P<claim>\d+)")
    pattern = re.compile("(?P<country>[A-Z]{2})(?P<patent_type>RE|PP|D|H|T)?(?P<number>\d+)(?P<kind_code>[A-Z]\d?)(?P<additional>.*?)_(?P<claim>\d+)")

    print(f"Loading patents from {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    # remove the kind code e.g. B2 from the claim_id
    patent_created_count = patent_updated_count = 0
    claim_created_count = claim_updated_count = 0
    element_created_count = element_updated_count = 0

    std_modtype, created = ModificationType.objects.get_or_create(name="Unmodified", description="Original text without modifications")

    for num, r in enumerate(data):
        # Extract the components

        match = pattern.search(r['claim_id'])
        if not match:
            print(f"Claim_ID {r['claim_id']} did not match the expected pattern")
        else:
            # print(f"{num} Processing Claim_ID {r['claim_id']}")
            # Capture the components
            country = match.group('country')
            number = match.group('number')
            claim = match.group('claim')
            kind_code = match.group('kind_code')
            orig_claim_id = r['claim_id']
            ptype = match.group('patent_type') or ""
            additional = match.group('additional')

            # Convert patent type to descriptive string
            patent_type_map = {
                'RE': 'Reissue',
                'PP': 'Plant',
                'D': 'Design',
                'H': 'Statutory',
                'T': 'Defensive'
            }

            print(num, orig_claim_id, country, number, claim, kind_code, ptype, additional)
            patent_type = patent_type_map.get(ptype, 'Utility')

            # Construct the new claim_id using captured components
            patent_ref = f"{country}{number}"
            full_patent_ref = f"{country}{ptype}{number}{kind_code}{additional}"
            r['claim_id'] = f"{country}{number}_{claim}"

            patent, patent_created = Patent.objects.update_or_create(
                    patent_ref=patent_ref,
                    defaults={
                            'full_patent_ref': full_patent_ref,
                            'patent_country': country,
                            'patent_number': number,
                            'patent_type': patent_type,
                            'patent_kind_code': kind_code,
                    })

            if patent_created:
                # print(f"Added {patent.patent_ref}")
                patent_created_count += 1
            else:
                print(f"Updated {patent.patent_ref}")
                patent_updated_count += 1

            claim, claim_created = PatentClaim.objects.update_or_create(
                    patent=patent,
                    claim_id=r['claim_id'],
                    defaults={
                        'claim_number': claim,
                        'text': r['claim_text'],
                        'orig_claim_id': orig_claim_id,
                        'related_claim': related,
                        'key_related_claim': key_related,
                    })

            if claim_created:
                # print(f"Added {claim.claim_id}")
                claim_created_count += 1
            else:
                # print(f"Updated {claim.claim_id}")
                claim_updated_count += 1

            for element in r['elements']:
                element, element_created = ClaimElement.objects.update_or_create(
                        claim=claim,
                        element_id=f"{claim}_{element['number']}",
                        defaults={
                            'element_number': element['number'],
                            'text': element['text'],
                        })
                if element_created:
                    # print(f"Added {element.claim.claim_id} Claim {element.element_number}")
                    element_created_count += 1
                else:
                    # print(f"Updated {element.claim.claim_id} Claim {element.element_number}")
                    element_updated_count += 1

            # build up the standard (unmodifed Records for embedding)
            # This duplciates the text since the same text is in the original PatentClaim record,
            # but this way embedding the text wihout any changes works the same as if there 
            # were changes to the ext.  (This could be changed later to have a special case in the code
            # which knows about the unmodifed version and pulls the text from the PatentClaim record
            # instead).
            lookup = {
                'claim': claim,
                'modification_type': std_modtype,
            }
            defaults = {'modified_text': claim.text}
            _c_for_e, c_for_e_created = ClaimForEmbedding.objects.update_or_create(defaults=defaults, **lookup)

    print(f"{'':>20}{'Patents':>20} {'Claims':>20} {'Elements':>20}")
    print(f"{'Created':>20} {patent_created_count:>20} {claim_created_count:>20} {element_created_count:>20}")
    print(f"{'Updated':>20} {patent_updated_count:>20} {claim_updated_count:>20} {element_updated_count:>20}")


def load_document(filename, document_name=None, maxrec=None, update=False):

    document, created = Document.objects.get_or_create(name=document_name)
    document.filename = filename
    document.save()

    if created:
        print("Document was created.")
    else:
        print("Document already exists.")

    print(f"Loading spec {filename}")
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        if maxrec is not None:
            data = data[:maxrec]

    existing_section_id = set(DocSection.objects.filter(document=document).values_list('section_id', flat=True))

    print("Adding {len(data)} records")

    std_modtype, created = ModificationType.objects.get_or_create(name="Unmodified", description="Original text without modifications")

    for section in data:

        if section['section_id'] not in existing_section_id or update:
            doc_section, created = DocSection.objects.update_or_create(
                document=document, 
                section_id=section['section_id'],
                defaults={
                    'text': section['section_text'], 
                    'section_title': section['section_title'],
                    'section_title_text': section['section_title_text']
                }
            )

            if not created:
                print(f"section {section['section_id']} already existed...  updating")
            else:
                print(f"section {section['section_id']} already exists...  skipping")

            # build up the standard modified data (i.e. unmodifed Records for embedding)
            # This duplciates the text since the same text is in the original DocSection record,
            # but this way embedding the text wihout any changes works the same as if there 
            # were changes.  (This could be changed later to have a special case in the code
            # which knows about the unmodifed version and pulls the text from the DocSection record
            # instead.
            lookup = {
                    'section': doc_section,
                    'modification_type': std_modtype,
            }
            defaults = {'modified_text': doc_section.text}
            _s_for_e, s_for_e_created = SectionForEmbedding.objects.update_or_create(defaults=defaults, **lookup)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    subparsers = parser.add_subparsers(dest='command', required=False)

    parser_loaddoc = subparsers.add_parser('loaddoc', help='Load a document.')
    parser_loaddoc.add_argument('filename', type=str, help='The name of the file to load.')
    parser_loaddoc.add_argument('docname', type=str, nargs='?', help='Optional document description for the database -- defaults to filename.')

    parser_loadclaims = subparsers.add_parser('loadclaims', help='Load a document.')
    parser_loadclaims.add_argument('filename', type=str, help='The name of the file to load.')
    parser_loadclaims.add_argument('--related',
                                   action='store_true',
                                   help='All records in this file shoudl be marked as related')
    parser_loadclaims.add_argument('--key',
                                   action='store_true',
                                   help='All records in this file should be marked as key_related')

    # parser.add_argument('--load',
    #                nargs=2,
    #                metavar=('filename', 'collection'),
    #                help='Load data from a file into a collection')
    # parser.add_argument('collection', type=str, help='Name of the collection')
    # parser.add_argument('filename', type=str, help='Name of the file to load')
    # parser.add_argument('docname', type=str, nargs='?', help='Optional document name. Will default to filename if not specfied')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--update', action='store_true', help='Update record if exists (default skip)')

    # parser.add_argument('--delete', type=str, default=None, help='Delete specfied collection')
    # parser.add_argument('--load', action='store_true', help='Flag to indicate if resources should be downloaded')
    # parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
    # parser.add_argument('--temp', action='store_true', help='Flag to indicate to use temporary collection')
    # parser.add_argument('--list', action='store_true', help='List chromadb collections')
    # parser.add_argument('--plook', action='store_true', help='Flag to indicate lookup document matches for claims')
    # parser.add_argument('--results', type=int, default=5, help='Number of sections to return (default 5)')jjjkk

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    print(f"Args: {args}")

    if args.command is None:
        parser.print_help()
        sys.exit()

    if args.command == 'loaddoc':
        input_filename = args.filename
        document_name = args.docname if args.docname else args.filename  

        print(f"Loading File {input_filename} with doc named {document_name}")

        load_document(input_filename, document_name, args.maxrec, args.update)

    if args.command == 'loadclaims':
        input_filename = args.filename
        load_claims(filename=input_filename, maxrec=args.maxrec,  related=args.related, key_related=args.key, update=args.update)


if __name__ == "__main__":
    main()
    print('exiting')

