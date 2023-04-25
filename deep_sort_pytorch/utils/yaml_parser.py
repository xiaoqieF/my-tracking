import os
import yaml

def get_yaml_data(yaml_file):
    if not os.path.isfile(yaml_file):
        raise ValueError(f'path:{yaml_file} not exist')
    with open(yaml_file, 'r') as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    return data