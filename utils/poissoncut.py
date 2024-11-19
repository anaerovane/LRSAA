import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import os
from tqdm import tqdm

def poisson_disk_sampling(width, height, r, k=30):
    cell_size = r / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [None] * (grid_width * grid_height)

    def point_to_grid_index(x, y):
        return int(y // cell_size) * grid_width + int(x // cell_size)

    def is_valid(x, y):
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        return True

    def get_random_point_around(center_x, center_y):
        angle = 2 * np.pi * np.random.rand()
        distance = r + (np.random.rand() * r)
        return center_x + distance * np.cos(angle), center_y + distance * np.sin(angle)

    def get_nearest_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    neighbor = grid[ny * grid_width + nx]
                    if neighbor is not None:
                        neighbors.append(neighbor)
        return neighbors

    first_x, first_y = np.random.rand() * width, np.random.rand() * height
    points = [(first_x, first_y)]
    active_list = [0]

    while len(active_list) > 0:
        i = active_list.pop(np.random.randint(len(active_list)))
        x, y = points[i]
        for _ in range(k):
            new_x, new_y = get_random_point_around(x, y)
            if is_valid(new_x, new_y) and all(pdist([(new_x, new_y), n]) > r for n in get_nearest_neighbors(int(new_x // cell_size), int(new_y // cell_size))):
                points.append((new_x, new_y))
                grid_index = point_to_grid_index(new_x, new_y)
                grid[grid_index] = (new_x, new_y)
                active_list.append(len(points) - 1)

    return points

def ensure_coverage(points, width, height, crop_size):
    tree = KDTree(points)
    step = crop_size // 2
    for y in range(0, height, step):
        for x in range(0, width, step):
            dist, _ = tree.query((x, y))
            if dist > step:
                points.append((x, y))
                tree = KDTree(points) 
    return points

def save_cropped_images(image_path, output_dir, output_label_dir,sample_points, size=(640, 640)):
    Image.MAX_IMAGE_PIXELS = 10000000000
    original_image = Image.open(image_path)
    width, height = original_image.size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for i, (x, y) in tqdm(enumerate(sample_points)):
        left = max(0, x - size[0] // 2)
        upper = max(0, y - size[1] // 2)
        right = min(left + size[0], width)
        lower = min(upper + size[1], height)
        cropped_image = original_image.crop((left, upper, right, lower))
        label_path=os.path.join(output_label_dir, f'crop_{i}.txt')
        with open(label_path, 'w') as f:
            f.write(f'{left}\n{upper}\n{right}\n{lower}\n')
        cropped_image.save(os.path.join(output_dir, f'crop_{i}.tif'))
        
