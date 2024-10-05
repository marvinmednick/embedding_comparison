import os 
import sys
import re
import django 
import argparse
import json
from django.utils import timezone
from sentence_transformers import SentenceTransformer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import Question, Document, DocSection, Patent, PatentClaim, ClaimElement


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    subparsers = parser.add_subparsers(dest='command', required=False)

    parser_loaddoc = subparsers.add_parser('subcommand', help='Help')
    parser_loaddoc.add_argument('filename', type=str, help='The name of the file to load.')
    parser_loaddoc.add_argument('docname', type=str, nargs='?', help='Optional document description for the database -- defaults to filename.')
    parser_loaddoc.add_argument('--related',
                                action='store_true',
                                help='All records in this file should be marked as related')

    # parser.add_argument('--load',
    #                nargs=2,
    #                metavar=('filename', 'collection'),
    #                help='Load data from a file into a collection')
    # parser.add_argument('collection', type=str, help='Name of the collection')
    # parser.add_argument('filename', type=str, help='Name of the file to load')
    # parser.add_argument('docname', type=str, nargs='?', help='Optional document name. Will default to filename if not specfied')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--update', action='store_true', help='Update record if exists (default skip)')

    # parser.add_argument('--test', action='store_true', help='Flag to indicate test queries should be run')
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

if __name__ == "__main__":
    main()
    print('exiting')
