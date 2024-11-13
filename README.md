# LRSAA
Large-scale remote sensing image target recognition and automatic annotation

You can now find the preprint of the paper on arxiv: 

### Installation

```
git clone https://github.com/anaerovane/LRSAA.git
cd LRSAA
conda create -n LRSAA python==3.11
conda activate LRSAA
pip install -r requirements.txt
```

### Usage

```
cd LRSAA
python main.py --yaml poisson.yaml
```

poisson.yaml file need to be created as follows

```yaml
image_path: 'E:/mbnet/sh/18.tiff'
output_directory: 'E:/mbnet/sh/18_image'
output_label_directory: 'E:/mbnet/sh/18_label'
output_labelnew_directory: 'E:/mbnet/sh/18_labelnew'
jpg_directory: 'E:/mbnet/sh/18_jpg'
output_image_path: 'E:/mbnet/sh/18new.tiff'
output_jsonl_path: 'E:/mbnet/sh/18new.jsonl'
sample_radius: 500
```

