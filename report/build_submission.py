"""
Builds the hand-in in submission/: the report (REPORT.md, REPORT.docx, figures) and
webshop_rl_submission.zip, from the reported Kaggle run, the earlier run and the laptop runs.

    data\\venv\\Scripts\\python.exe report\\build_submission.py

The Word version needs pandoc on the PATH, or `pip install pypandoc_binary`.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'kaggle_outputs' / 'outputs_full'          # the reported run
FIRST = ROOT / 'kaggle_outputs_run1' / 'outputs_full'   # the earlier full run (Section 6.6)
LAPTOP = ROOT / 'results_cpu'                           # validation and earlier test runs on a laptop
OUT = ROOT / 'submission'

# Left out of code/: the local environment and catalogue (data/ at the top only; baseline_models/data
# holds the demonstrations and is kept), run outputs (added separately) and generated files.
SKIP_TOP = {'data', 'submission', 'results_cpu'}
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.git', '.venv', 'venv', 'node_modules'}
SKIP_FILES = {'il_trajs_finalized_images.jsonl'}


def ignore(folder, names):
    top = Path(folder).resolve() == ROOT
    return [n for n in names if (top and n in SKIP_TOP) or n in SKIP_DIRS or n in SKIP_FILES
            or n.startswith('kaggle_outputs') or n.endswith(('.pyc', '.part', '.pt', '.pth'))]


def build_report():
    for old in ('REPORT.md', 'REPORT.docx', 'figures'):
        path = OUT / old
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'report' / 'make_report.py'), '--outputs', str(RUN), '--out_dir', str(OUT),
                    '--previous', str(FIRST), '--laptop', str(LAPTOP)], check=True)


def build_zip():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'bundle'
        shutil.copytree(ROOT, bundle / 'code', ignore=ignore)
        report = bundle / 'report'
        report.mkdir()
        for name in ('REPORT.md', 'REPORT.docx'):
            if (OUT / name).exists():
                shutil.copy(OUT / name, report / name)
        shutil.copytree(OUT / 'figures', report / 'figures')
        shutil.copytree(RUN / 'results', bundle / 'results')
        shutil.copytree(RUN / 'logs', bundle / 'logs')
        shutil.copytree(RUN / 'runs', bundle / 'training_logs', ignore=shutil.ignore_patterns('*.pt', '*.pth', '*.tmp'))
        for name in ('gpu_usage.csv', 'meta.json', 'catalog_train.json', 'catalog_test.json'):
            shutil.copy(RUN / name, bundle / name)
        shutil.copytree(RUN / 'stages', bundle / 'stages')
        if FIRST.exists():
            shutil.copytree(FIRST / 'results', bundle / 'first_run' / 'results')
            shutil.copy(FIRST / 'meta.json', bundle / 'first_run' / 'meta.json')
        if LAPTOP.exists():
            shutil.copytree(LAPTOP, bundle / 'laptop_runs')
        archive = shutil.make_archive(str(OUT / 'webshop_rl_submission'), 'zip', bundle)
    print(f'Wrote {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    build_report()
    build_zip()
