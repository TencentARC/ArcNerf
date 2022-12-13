# -*- coding: utf-8 -*-

import argparse
import json
import os

import yaml


class Obj(object):
    """object for convert dict into object"""

    def __init__(self, cfgs_dict):
        self.__dict__.update(cfgs_dict)


def dict_to_obj(nested_dict):
    """Turn nested dict into object"""
    return json.loads(json.dumps(nested_dict), object_hook=Obj)


def obj_to_dict(nested_obj):
    """Nested obj into dict"""
    dic = {}
    for k, v in nested_obj.__dict__.items():
        if isinstance(v, type(nested_obj)):
            dic[k] = obj_to_dict(v)
        else:
            dic[k] = v

    return dic


def pop_none_item(input_dict):
    """Pop the None item in dict"""
    pop_k = []
    for k, v in input_dict.items():
        if v is None:
            pop_k.append(k)
    for k in pop_k:
        input_dict.pop(k)


def remap_str_list(value):
    """Remap a string of list"""
    assert value.startswith('[') and value.endswith(']'), 'Should be in []'
    value = [remap_value((v.strip())) for v in value[1:-1].split(',')]

    return value


def remap_value(value):
    """Remap value from string"""
    if isinstance(value, dict):
        raise RuntimeError('Should not be a dict here...')

    if any([isinstance(value, t) for t in [bool, int, float, list]]):
        return value
    elif isinstance(value, str):  # Support None, T/F, int, float, list
        if value.startswith('str(') and value.endswith(')'):  # force string
            return str(value[4:-1])
        elif value.lower() == 'none':
            return None
        elif value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        # int (+-)
        elif value.isdigit():
            return int(value)
        elif value.startswith('-') and value[1:].isdigit():
            return -1 * int(value[1:])
        # float (+-)
        elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
            return float(value)
        elif value.startswith('-') and value[1:].replace('.', '', 1).isdigit() and value[1:].count('.') < 2:
            return -1 * float(value[1:])
        # scientific notation
        elif value.replace('e', '', 1).isdigit() and value.count('e') < 2:
            return float(value)
        elif value.replace('e-', '', 1).isdigit() and value.count('e-') < 2:
            return float(value)
        elif value.startswith('-') and value[1:].replace('e', '', 1).isdigit() and value[1:].count('e') < 2:
            return -1 * float(value[1:])
        elif value.startswith('-') and value[1:].replace('e-', '', 1).isdigit() and value[1:].count('e-') < 2:
            return -1 * float(value[1:])
        # string with '' ""
        elif value.startswith('\'') and value.endswith('\'') or value.startswith('\"') and value.endswith('\"'):
            return str(value[1:-1])
        # list with bracket
        elif value.startswith('[') and value.endswith(']'):
            return remap_str_list(value)
        # list without bracket but with comma
        elif len(value.split(',')) > 1:
            return remap_str_list('[{}]'.format(value))

        return value

    return value


def process_dict(dic):
    """Process dict by remapping the values of the dict"""
    for k, v in dic.items():
        if isinstance(v, dict):
            dic[k] = process_dict(v)
        else:
            dic[k] = remap_value(v)

    return dic


def load_yaml(cfgs_file):
    """Load yaml file into dict"""
    assert os.path.exists(cfgs_file), 'Configs file not exist, please check {}...'.format(cfgs_file)
    with open(cfgs_file, encoding='utf8', mode='r') as yaml_file:
        cfgs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return cfgs


def nested_set_dict(dic, keys, value):
    """Set the dict by as list of keys"""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def update_configs(cfgs, unknowns):
    """Update configs from command line arguments"""
    if unknowns is None:
        return cfgs

    # this is special param using launch
    if len(unknowns) > 0 and unknowns[0].startswith('--local_rank='):
        unknowns.pop(0)

    for idx, arg in enumerate(unknowns):
        if idx == len(unknowns) - 1:
            break
        if arg.startswith('--'):  # Only allows one following value for each key now
            keys = arg.replace('--', '').split('.')
            value = unknowns[idx + 1]
            nested_set_dict(cfgs, keys, value)

    return cfgs


def parse_configs():
    """Parse configs by reading configs. Allows updates from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True, help='Configs yaml to be read')
    args, unknowns = parser.parse_known_args()

    cfgs = load_configs(args.configs, unknowns)

    return cfgs


def load_configs(configs, unknowns=None):
    """load configs by path"""
    cfgs = load_yaml(configs)
    cfgs = update_configs(cfgs, unknowns)
    cfgs = process_dict(cfgs)
    cfgs = dict_to_obj(cfgs)

    return cfgs


def valid_key_in_cfgs(cfg_field, cfg_key):
    """Return True if cfg_key is in cfg_field and it is not None"""
    if hasattr(cfg_field, cfg_key) and getattr(cfg_field, cfg_key) is not None:
        return True
    return False


def get_value_from_cfgs_field(cfg_field, cfg_key, default_value=None):
    """Return None if cfg_key is invalid, else return the default_value"""
    return default_value if not valid_key_in_cfgs(cfg_field, cfg_key) else getattr(cfg_field, cfg_key)


def create_train_sh(rep_name, config_path, main_filepath='', save_dir='./'):
    """Create an script for reproducing the expr"""
    pycmd = 'python ' + main_filepath + ' '
    pycmd += '--name {} '.format(rep_name) + '\\\n'
    pycmd += '--config {}'.format(config_path)

    pycmd = '#!/bin/bash\n\n' + pycmd
    job_script = os.path.join(save_dir, 'job.sh')

    with open(job_script, 'w') as f:
        f.write(pycmd)


def dump_configs(cfgs, path):
    """Dump the configs into yaml file"""
    with open(path, 'w') as f:
        yaml.dump(obj_to_dict(cfgs), f)
