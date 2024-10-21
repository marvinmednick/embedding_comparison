import csv
from unidecode import unidecode
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process CSV file with sections')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('--output_file', '-o', default='output.csv', help='Output CSV file (default: output.csv)')

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', newline='', encoding='ascii') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            ascii_row = [unidecode(cell) for cell in row]
            writer.writerow(ascii_row)


if __name__ == '__main__':
    main()
