"""Run several fresh shared worlds; all competing policies are frozen together."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'experiments/survival_search/campaign.json'
SUITE = ROOT / 'experiments/survival_search/validation01_suite.json'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    suite = json.loads(SUITE.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    suite['queue_pid'] = os.getpid()
    suite['status'] = 'running'
    write(SUITE, suite)
    for trial in suite['trials']:
        if trial.get('status') == 'complete':
            continue
        run = Path(trial['run_dir'])
        log = Path(str(run) + '_launcher.log')
        command = [sys.executable, '-u', str(ROOT / 'run_survival_search.py'),
                   '--variants', trial['variants'], '--run-dir', str(run),
                   '--source-checkpoint', suite['source_checkpoint'], '--fast',
                   '--initial-eval-steps', '0', '--train-steps', '0',
                   '--eval-steps', '50000', '--seed', str(trial['seed']),
                   '--port', '11128', '--verify-visuals']
        campaign.update(status='validation_running', active_run=str(run),
                        launcher_log=str(log), command=command, validation_suite=str(SUITE),
                        queue_pid=os.getpid())
        trial['status'] = 'starting'
        write(SUITE, suite)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            campaign['launcher_pid'] = child.pid
            write(CAMPAIGN, campaign)
            code = child.wait()
        status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
        if code or status.get('status') != 'complete':
            trial['status'] = 'failed'
            suite['status'] = 'failed'
            campaign['status'] = 'validation_failed'
            write(SUITE, suite); write(CAMPAIGN, campaign)
            raise RuntimeError(f'Validation failed: {run}, code {code}')
        trial['status'] = 'complete'
        trial['finished_at'] = time.time()
        write(SUITE, suite)
    suite['status'] = 'complete'
    campaign['status'] = 'awaiting_validation_analysis'
    write(SUITE, suite); write(CAMPAIGN, campaign)


if __name__ == '__main__':
    main()
