import csv
import re
import argparse
import sys
import json
import unicodedata


def unicode_to_ascii(text):
    # Define replacements for specific Unicode characters
    replacements = {
        '\u2011': '-',   # Replace non-breaking hyphen with regular hyphen
        '\u2013': '-',   # EN dash
        '\u2014': '--',  # EM dash
        '\u2018': "'",   # Left single quotation mark
        '\u2019': "'",   # Right single quotation mark
        '\u201C': '"',   # Left double quotation mark
        '\u201D': '"',   # Right double quotation mark
        # Add more replacements as needed
    }
    
    # Apply specific replacements
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Normalize remaining Unicode characters
    nfkd_form = unicodedata.normalize('NFKD', text)
    
    # Remove any remaining non-ASCII characters, preserving spaces and tabs
    ascii_text = re.sub(r'[^\x00-\x7F]+', '', nfkd_form)
    
    return ascii_text


def unicode_to_ascii(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    ascii_text = re.sub(r'[^\t -~]+', '', nfkd_form)
    return ascii_text

def load_section_numbers(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def process_section_range(start, end, section_numbers):
    start_index = section_numbers.index(start)
    end_index = section_numbers.index(end)
    return section_numbers[start_index:end_index + 1]


def process_tables_ranges(items_string):
    modified_tables = []
    needs_review = False
    ranges = re.findall(r'Tables? ([0-9A-Z]\d*)-(\d+) to ([0-9A-Z]\d*)-(\d+)', items_string)
    for range_tuple in ranges:
        start_major, start_minor, end_major, end_minor = range_tuple
        if start_major != end_major:
            needs_review = True
        else:
            start = int(start_minor)
            end = int(end_minor)
            modified_tables.extend([f'Table {start_major}-{i}' for i in range(start, end + 1)])
    return modified_tables, needs_review


def process_fig_ranges(items_string):
    modified_figs = []
    needs_review = False
    ranges = re.findall(r'Figs?. ([0-9A-Z]\d*)-(\d+) to ([0-9A-Z]\d*)-(\d+)', items_string)
    for range_tuple in ranges:
        start_major, start_minor, end_major, end_minor = range_tuple
        if start_major != end_major:
            needs_review = True
        else:
            start = int(start_minor)
            end = int(end_minor)
            modified_figs.extend([f'Figure {start_major}-{i}' for i in range(start, end + 1)])
    return modified_figs, needs_review


def process_items(items_string, section_numbers):
    print("Processing Item string: ", items_string)
    items = items_string.split(', ')
    processed_items = []
    needs_review = False
    i = 0
    while i < len(items):
        item = items[i].strip()
        print(f"Item: ({item})")

        if item.startswith(('Table ', 'Tables ')) and ' to ' in item:
            tables, review = process_tables_ranges(item)
            processed_items.extend(tables)
            needs_review = needs_review or review
        elif item.startswith(('Fig. ', 'Figs. ')) and ' to ' in item:
            figs, review = process_fig_ranges(item)
            processed_items.extend(figs)
            needs_review = needs_review or review
            continue
        elif match := re.match(r'(Tables?|Fig\.|Figs\.)\s*([0-9A-Z]\d*-\d+)', item):
            print("matched 1")
            if match.group(1).startswith('Table'):
                ref_type = 'Table'
            else:
                ref_type = 'Figure'

            references = re.findall(r'([0-9A-Z]\d*-\d+)', item)
            for ref in references:
                processed_items.append(f'{ref_type} {ref.strip()}')
            i += 1
            while i < len(items) and re.match(r'^[A-Z0-9]\d*-\d+$', items[i]):
                processed_items.append(f'{ref_type} {items[i]}')
                i += 1
            continue

        elif item.startswith('Table '):
            print("Unexpected table", item, items_string)
            table_numbers = item[7:].split(', ')
            for number in table_numbers:
                processed_items.append(f'Table {number.strip()}')
            i += 1
            while i < len(items) and re.match(r'^[0-9A-Z]\d*-\d+$', items[i]):
                processed_items.append(f'Table {items[i]}')
                i += 1
            continue
        elif item.startswith(('Fig. ', 'Figs. ')):
            print(f"Unexpected Fig  Item:({item}) String: ({items_string})")
            hex_string = ''.join([f'{ord(c):04x} ' for c in item])
            print(hex_string)

            figure_numbers = item[6:].split(', ')
            for number in figure_numbers:
                processed_items.append(f'Figure {number.strip()}')
            i += 1
            while i < len(items) and re.match(r'^[A-Z0-9]\d*-\d+$', items[i]):
                processed_items.append(f'Figure {items[i]}')
                i += 1
            exit(1)
            continue
        elif item.startswith('Fig. '):
            print("Unexpected Fig 2", item, items_string)
            processed_items.append(f'Figure {item[5:]}')
        elif ' to ' in item:
            start, end = item.split(' to ')
            start = start.strip()
            end = end.strip()
            if start in section_numbers and end in section_numbers:
                processed_items.extend(process_section_range(start, end, section_numbers))
            else:
                processed_items.append(item)  # unrecognized format, keep as is
                needs_review = True
        else:
            processed_items.append(item)
        i += 1
    return processed_items, needs_review


def user_confirm(original, modified):
    print(f"Original: {original}")
    print(f"Modified: {modified}")
    while True:
        response = input("Continue processing? (y/n): ").lower().strip()
        if response in ['y', 'n']:
            return response == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")


def process_csv(input_file, output_file, section_numbers, confirm):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for num, input_row in enumerate(reader):
            print(f"line {num}")
            row = {key: unicode_to_ascii(value) for key, value in input_row.items()}
            original_sections = row['sections']
            modified_sections, needs_review = process_items(original_sections, section_numbers)
            new_sections = ', '.join(modified_sections)
            if new_sections != original_sections:
                if not confirm or (confirm and user_confirm(original_sections, new_sections)):
                    row['sections'] = new_sections
                    writer.writerow(row)
                else:
                    print("Processing stopped by user.")
                    sys.exit(0)
            else:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Process CSV file with sections')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('--output_file', '-o', default='output.csv', help='Output CSV file (default: output.csv)')
    parser.add_argument('--section_numbers_file', default='section_numbers.json', help='JSON file containing sorted section numbers')
    parser.add_argument('--confirm', action='store_true', help='User to confirm each line change')

    args = parser.parse_args()

    section_numbers = load_section_numbers(args.section_numbers_file)
    process_csv(args.input_file, args.output_file, section_numbers, args.confirm)


if __name__ == '__main__':
    main()
