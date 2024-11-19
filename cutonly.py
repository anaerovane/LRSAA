import yaml
from PIL import Image
import argparse
import os
import json
from utils.poissoncut import *

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

def cromain(yaml_file):
    yaml_data = read_yaml('poisson.yaml')
    print(yaml_data)

    image_path = yaml_data['image_path']
    output_directory = yaml_data['output_directory']
    sample_radius = int(yaml_data['sample_radius'])
    output_label_dir = yaml_data['output_label_directory']
    #output_labelnew_dir=yaml_data['output_labelnew_path']
    jpg_dir=yaml_data['jpg_directory']
    #output_image_path = yaml_data['output_image_path']
    #output_jsonl_path = yaml_data['output_jsonl_path']
    threshold=yaml_data['threshold'] if 'threshold' in yaml_data else 0.5
    cropwidth=yaml_data['cropwidth'] if 'cropwidth' in yaml_data else 640
    cropheight=yaml_data['cropheight'] if 'cropheight' in yaml_data else 640

    os.makedirs(jpg_dir,exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    #os.makedirs(output_labelnew_dir, exist_ok=True)
    crop_size = (cropwidth, cropheight)
    
    Image.MAX_IMAGE_PIXELS = 10000000000  
    original_image = Image.open(image_path)
    width, height = original_image.size
    print(f"width: {width}, height: {height}")
    sample_points = poisson_disk_sampling(width, height, sample_radius)
    print("sample complete")

    sample_points = ensure_coverage(sample_points, width, height, crop_size[0])
    print("sample points complete",len(sample_points))
    
    save_cropped_images(image_path, output_directory, output_label_dir,sample_points, crop_size)
    print("cropped images saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a YAML file')
    parser.add_argument('--yaml', required=True, help='Path to the YAML file to be read')
    args = parser.parse_args()
    filename = args.yaml
    cromain(filename)