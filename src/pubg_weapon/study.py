"""One notebook entry point for the complete coursework experiment."""
import argparse
import faulthandler
from datetime import datetime, timezone
from importlib.metadata import distributions
import platform
from pathlib import Path
import subprocess
import sys
import traceback

from .config import ROOT
from .io import file_sha256, new_directory, read_json, write_json


def run(source, output, config):
    faulthandler.enable(all_threads=True)
    output = new_directory(output)
    write_json(output / 'study_config.json', config)
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    (output / 'commit.txt').write_text(commit + '\n', encoding='utf-8')
    write_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
               'packages': {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']},
               'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        from .full_data import prepare_full
        from .study_data import benchmark_cohorts, summarize_train
        from .study_diagnostics import error_profiles, resource_comparison
        from .study_training import run_study_training
        from .resources import measure
        stages = {}
        subprocess.run([
            sys.executable, '-u', '-c',
            'import sys; from pubg_weapon.modern import preflight; from pubg_weapon.io import read_json; '
            'preflight(read_json(sys.argv[1]), sys.argv[2])',
            str(output.resolve() / 'study_config.json'), str(output.resolve()),
        ], check=True)
        print('Official-model GPU preflight completed in an isolated process.', flush=True)
        prepared = output / 'prepared'
        prepare_full(source, prepared, partitions=config['partitions'], chunk_size=config['chunk_rows'], workers=config['workers'])
        with measure(stages, 'train_summary', output):
            summarize_train(prepared, output / 'eda')
        with measure(stages, 'benchmark_cohorts', output):
            cohorts = benchmark_cohorts(prepared, output / 'comparison_cohorts', config)
        with measure(stages, 'worker_comparison', output):
            resource_comparison(prepared, output / 'eda')
        for seed in config['seeds']:
            full_output = output / 'full_models' / str(seed)
            run_study_training(prepared, full_output, {**config, 'seed': seed})
            with measure(stages, f'error_analysis_{seed}', output):
                error_profiles(prepared, full_output, output / 'diagnostics' / str(seed), config['jobs'])
        from .modern import run_modern
        run_modern(cohorts, output / 'modern_comparison', config)
        write_json(output / 'extra_resources.json', stages)
        from .study_results import export_results
        export_results(output)
        # The complete source split and cohort definitions are retained for exact reproduction.
        files = [p for p in output.rglob('*') if p.is_file() and prepared not in p.parents]
        write_json(output / 'artifact_checksums.json', {p.relative_to(output).as_posix(): file_sha256(p) for p in files})
        write_json(output / 'status.json', {'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'models': ['TabM (ICLR 2025)', 'TabICLv2 (ICML 2026)'],
                   'repository': 'https://github.com/kttt294/PUBG-Weapon-Groups'})
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(output / 'status.json', {'status': 'failed', 'error': traceback.format_exc()[-4000:]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'study.json')
    args = parser.parse_args()
    run(args.source, args.output, read_json(args.config))


if __name__ == '__main__':
    main()
