import os
import sys
import django
import argparse
import csv
import json
from pprint import pprint

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

# from polls.models import Embedding768, Embedding32


def load_associations(csv_file_path, json_file_path):
    # Read the CSV and add the data to a dictionary
    data = []
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            pprint(row)
            row['patent'] = row['patent'].replace(',', '')
            row['patent_id'] = row['country'] + row['patent']
            row['claim_id'] = row['patent_id'] + '_' + row['claim']
            row['sections'] = row['sections'].split(',')
            row['sections'] = [x.strip() for x in row['sections']]
            data.append(row)

    if json_file_path:
        # Write the data to a JSON file
        with open(json_file_path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=4)
    else:
        pprint(data)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Import a ")
    parser.add_argument('input', type=str, help='CSV file with associations')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: csv_to_json.out)')

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    load_associations(args.input, args.output)


if __name__ == "__main__":
    main()
