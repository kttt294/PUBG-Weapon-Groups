"""Prepare the full corpus, fit the models and evaluate a held-out set of matches."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pubg_weapon.full_data import prepare_full
from pubg_weapon.io import new_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--partitions', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=400_000)
    parser.add_argument('--xgb-rounds', type=int, default=60)
    parser.add_argument('--sgd-epochs', type=int, default=2)
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    args = parser.parse_args()
    output = new_directory(args.output)
    prepared = output / 'prepared'
    manifest = prepare_full(args.source, prepared, partitions=args.partitions, chunk_size=args.chunk_size,
                            workers=args.workers)
    for split, summary in manifest['splits'].items():
        print(f'{split}: {summary["rows"]:,} dòng / {summary["matches"]:,} trận', flush=True)
    from pubg_weapon.full_training import run_full_training
    run_full_training(prepared, output / 'results', args.jobs, args.xgb_rounds,
                      args.sgd_epochs, args.bootstrap_repeats)


if __name__ == '__main__':
    main()
