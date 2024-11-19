# LRSAA
Large-scale remote sensing image target recognition and automatic annotation

You can now find the preprint of the dependent paper on arxiv: 

https://arxiv.org/abs/2411.07802

### Installation

```
git clone https://github.com/anaerovane/LRSAA.git
cd LRSAA
conda create -n LRSAA python==3.11
conda activate LRSAA
pip install -r requirements.txt
```

### Usage

#### Recommend

```
cd LRSAA
python main.py --yaml poisson.yaml
```

#### Test

##### Cut-only (for test)

```
python cutonly.py --yaml poisson.yaml
```

##### Result-only (for test)

```
python resultonly.py --yaml poisson.yaml
```

#### YAML

poisson.yaml file need to be created as follows

```yaml
image_path: 'E:/mbnet/gaode/xm/18.tiff'
output_directory: 'E:/mbnet/gaode/xm/18_image'
output_label_directory: 'E:/mbnet/gaode/xm/18_label'
output_labelnew_path: 'E:/mbnet/gaode/xm/18_labelnew'  #dir or file ending with .txt are both ok
jpg_directory: 'E:/mbnet/gaode/xm/18_jpg'
output_image_path: 'E:/mbnet/gaode/xm/18new.tiff'
output_jsonl_path: 'E:/mbnet/gaode/xm/18new.jsonl'
sample_radius: 500
threshold: 0.6
cropwidth: 640
cropheight: 640

```

We conducted the tests on both Windows CPU systems and Ubuntu 22.04LTS GPU systems (with an Nvidia Tesla M40 GPU VRAM12G)

### Note

We are still making significant modifications and optimizations to the project, including the paper and code!

Please use the project code with caution until we confirm its completion!

### Contact

Email: aerovane@mail.nankai.edu.cn (Wuzheng Dong)

