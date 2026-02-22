import yaml

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return cast_config(config)

def cast_config(config):
    def to_float(key, default):
        try:
            return float(config.get(key, default))
        except (ValueError, TypeError):
            return default

    def to_int(key, default):
        try:
            return int(config.get(key, default))
        except (ValueError, TypeError):
            return default

    config['embedding_dim'] = to_int('embedding_dim', 128)
    config['margin'] = to_float('margin', 0.2)
    config['lr'] = to_float('lr', 1e-4)
    config['threshold'] = to_float('threshold', 0.7)
    config['batch_size'] = to_int('batch_size', 32)
    config['epochs'] = to_int('epochs', 20)
    config['early_stopping_patience'] = to_int('early_stopping_patience', 5)
    return config
