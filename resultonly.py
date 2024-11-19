import yaml
from PIL import Image
import argparse
import os
import json
from utils.generate import *
from utils.eiou import *

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except Exception as e:
        print(e)

def tojsonl(result0,output_jsonl_path):
    with open(output_jsonl_path, 'a') as f:
        for item in result0:
            f.write(json.dumps(item) + '\n')

def main(yaml_file):
    yaml_data = read_yaml('poisson.yaml')
    print(yaml_data)

    image_path = yaml_data['image_path']

    output_labelnew_path=yaml_data['output_labelnew_path']
    output_image_path = yaml_data['output_image_path']
    output_jsonl_path = yaml_data['output_jsonl_path']
    threshold=yaml_data['threshold'] if 'threshold' in yaml_data else 0.5

    result=readtxt(output_labelnew_path)
    print(f"result complete: {len(result)}")

    result0=resultnew(result,threshold)
    print(f"result0 complete: {len(result0)}")
    tojsonl(result0,output_jsonl_path)

    draw_bbox(image_path, output_image_path, result0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a YAML file')
    parser.add_argument('--yaml', required=True, help='Path to the YAML file to be read')
    args = parser.parse_args()
    filename = args.yaml
    main(filename)
