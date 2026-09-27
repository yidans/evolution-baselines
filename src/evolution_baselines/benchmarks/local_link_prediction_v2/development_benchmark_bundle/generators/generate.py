"""Copy a precomputed external-protocol split; never resample at evaluation time."""
import argparse
import json
import shutil
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def split_path(config, seed):
    if config['schema_version'] != 'local-link-prediction-config.v1':
        raise ValueError('wrong configuration schema')
    if seed != config['split_seed']:
        raise ValueError('instance seed differs from precomputed split seed')
    path = (BUNDLE / config['split_path']).resolve()
    path.relative_to(BUNDLE.resolve())
    return path


def generate(config, seed):
    return json.loads(split_path(config, seed).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--family-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shutil.copyfile(split_path(config, args.seed), args.output)


if __name__ == '__main__':
    main()
