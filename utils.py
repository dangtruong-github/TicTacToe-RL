import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict
