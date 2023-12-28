import yaml
def read_params_from_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params


yaml_params = read_params_from_yaml("/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/models.yaml")

print(type(yaml_params['decision_tree']['model_type']))