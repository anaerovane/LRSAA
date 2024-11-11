import yaml
from PIL import Image
import os
from poisson_cut import *
from yolodetect import *

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except Exception as e:
        print(e)

if __name__ == '__main__':
    yaml_data = read_yaml('poisson.yaml')
    print(yaml_data)

    image_path = yaml_data['image_path']
    output_directory = yaml_data['output_directory']
    sample_radius = int(yaml_data['sample_radius'])
    output_label_dir = yaml_data['output_label_directory']
    output_labelnew_dir=yaml_data['output_labelnew_directory']
    jpg_dir=yaml_data['jpg_directory']
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_labelnew_dir, exist_ok=True)
    crop_size = (640, 640)
    
    Image.MAX_IMAGE_PIXELS = 1000000000  
    original_image = Image.open(image_path)
    width, height = original_image.size
    print(f"width: {width}, height: {height}")
    sample_points = poisson_disk_sampling(width,height, sample_radius)
    print("sample complete")

    sample_points = ensure_coverage(sample_points, width, height, crop_size[0])
    print("sample points complete",len(sample_points))
    
    save_cropped_images(image_path, output_directory, output_label_dir,sample_points, crop_size)
    print("cropped images saved")

    label_main(output_directory,output_label_dir,output_labelnew_dir,jpg_dir)


