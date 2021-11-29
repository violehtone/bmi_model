import csv
import json
import argparse


def tsv_to_json(data: str, output_file_name: str):
    """Converts a .tsv file into a json file"""
    with open(data, 'r') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        data = list(reader)

    with open(output_file_name, 'w') as json_file:
        json.dump(data, json_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_file_name', type=str, required=True)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    tsv_to_json(**args)