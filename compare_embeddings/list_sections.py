import os
# import sys
import django
# import argparse
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import DocSection


def custom_sort_key(section):
    parts = section.split('.')
    return tuple(int(part) if part.isdigit() else ord(part) - ord('A') + 10 for part in parts)


def main():

    section_list = DocSection.objects.filter(document__id=5).values_list('section_id', flat=True)
    print(f"Found {len(section_list)} records in DocSections")

    sorted_list = sorted(section_list, key=lambda x: custom_sort_key(x))

    json_file_path = 'section_list.json'
    with open(json_file_path, 'w') as jsonfile:
        json.dump(sorted_list, jsonfile, indent=4)


if __name__ == "__main__":
    main()
