"""
Builds kaggle/webshop_rl_kaggle.ipynb, a single file that carries the project's source
code, so it only has to be uploaded to Kaggle. Run this again after changing the code:

    python kaggle/build_notebook.py
"""
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle' / 'webshop_rl_kaggle.ipynb'
MAX_NOTEBOOK_BYTES = 1_000_000  # Kaggle refuses to commit a kernel of 1 MB or more
BUNDLE_TIME = 946684800  # year 2000; the zip format cannot store dates before 1980

INCLUDE = ['baseline_models', 'web_agent_site', 'search_engine', 'tests', 'kaggle', 'report', 'run_envs',
           'README.md', 'LICENSE.md', 'requirements.txt', 'conftest.py', 'setup.sh',
           'run_dev.sh', 'run_prod.sh', 'run_web_agent_site_env.sh', 'run_web_agent_text_env.sh']
EXCLUDE_NAMES = {'__pycache__', '.pytest_cache', 'chromedriver', 'webshop_rl_kaggle.ipynb'}
# These are large, and the pipeline downloads them.
EXCLUDE_DATA = {'il_trajs_finalized_images.zip', 'il_trajs_finalized_images.jsonl', 'goal_query_predict.json',
                'items_human_ins.json', 'human_goals.json', 'goal_query_map.json'}


def bundle_source():
    def keep(info):
        name = Path(info.name).name
        if name in EXCLUDE_NAMES or name in EXCLUDE_DATA or name.endswith('.pyc'):
            return None
        info.mtime = BUNDLE_TIME  # the same source always gives the same bundle
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:xz', preset=9) as tar:
        for item in INCLUDE:
            if (ROOT / item).exists():
                tar.add(ROOT / item, arcname=item, filter=keep)
    return buffer.getvalue()


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text, hidden=False):
    metadata = {'jupyter': {'source_hidden': True}, 'collapsed': True} if hidden else {}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': metadata, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


INTRO = """
# Checkout: reinforcement learning agents that shop (IL + A2C on two T4 GPUs)

This notebook runs the whole project and writes the report from its own results:

| Stage | What happens | Where |
|---|---|---|
| Setup | Isolated Python 3.10 environment, data download, search index, tests | CPU |
| Imitation learning | Behaviour cloning on human demonstrations, one seed per GPU | GPU 0 and GPU 1 |
| Reinforcement learning | A2C: IL+RL on GPU 0, and RL from pretrained BERT on GPU 1 | GPU 0 and GPU 1 |
| Value model | Simulated purchases on training goals, then a linear fit for the value shopper | CPU |
| Evaluation | Every agent on the 500 test goals | One GPU, one agent at a time, because each holds the whole catalogue in memory |
| Report | Figures, tables, comparisons, submission zip | CPU |

## How to run

1. **Settings** (right-hand panel): *Accelerator* = **GPU T4 x2**, *Internet* = **On**.
2. First do a quick check: leave `MODE = "smoke"` below and run all cells (about 15 to 20 minutes).
3. Then set `MODE = "full"` and use **Save Version → Save & Run All (Commit)**, so that the run continues after you close the tab.
4. Get `webshop_rl_submission.zip` and the `report` folder from the notebook's **Output** tab.

Smoke-test numbers only show that the pipeline works. They are not results.

## How the GPUs are used

Each T4 runs its own experiment, so two results take the time of one. All environments share one batched GPU pass with 16-bit mixed precision, and micro-batches grow until about 80% of GPU memory is in use. GPU use is recorded and plotted in the report.
"""

CONFIG = """
# ------------------------------- Settings -------------------------------
MODE = "smoke"          # "smoke" = quick pipeline check, "full" = the real experiment

CATALOG_SIZE = 200_000  # products, always including every human-goal product; 0 = all 1.18 million
IL_EPOCHS = 10          # imitation learning epochs
RL_HOURS = 2.5          # training time for each RL agent; both run at the same time
NUM_ENVS = 16           # parallel environments per RL agent
ROLLOUT_STEPS = 8       # steps per environment between updates

# If a run was interrupted: add its output as an input of this notebook and name it here.
RESUME_FROM = ""        # e.g. "/kaggle/input/my-notebook-output/outputs_full"
"""

UNPACK = """
# Unpacks the project's source code, which is stored in this notebook.
import base64, hashlib, io, os, shutil, subprocess, sys, tarfile
from pathlib import Path

PROJECT = Path("/kaggle/working/webshop")
if PROJECT.exists():
    shutil.rmtree(PROJECT)
PROJECT.mkdir(parents=True)
data = base64.b64decode(SOURCE_BUNDLE)
assert hashlib.sha256(data).hexdigest() == SOURCE_SHA256, "The source bundle is damaged"
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
    tar.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / "kaggle"))
print("Project unpacked to", PROJECT)
"""

GPU_CHECK = """
import importlib, pipeline
importlib.reload(pipeline)

config = pipeline.Config(
    mode=MODE, catalog_size=CATALOG_SIZE, il_epochs=IL_EPOCHS, rl_hours=RL_HOURS,
    num_envs=NUM_ENVS, bptt=ROLLOUT_STEPS, gpu_ids=pipeline.detect_gpus(),
    outputs_dir=f"/kaggle/working/outputs_{MODE}",
)
if MODE == "smoke":
    pipeline.smoke_settings(config)

subprocess.run(["nvidia-smi"], check=False)
print("GPUs found:", config.gpu_ids)
assert config.gpu_ids, "No GPU found: set Accelerator to GPU T4 x2 in the settings panel."
if len(config.gpu_ids) < 2:
    print("WARNING: only one GPU. The experiments will run one after the other and take about twice as long.")

if RESUME_FROM and not Path(config.outputs_dir).exists():
    shutil.copytree(RESUME_FROM, config.outputs_dir)
    print("Resuming from", RESUME_FROM)
Path(config.outputs_dir).mkdir(parents=True, exist_ok=True)
monitor = pipeline.GpuMonitor(config).start()
"""

STAGES = [
    ('## 1. Setup', 'Installs the exact package versions, downloads the catalogue, builds the search index and runs the tests. About 15 to 25 minutes in full mode.',
     'pipeline.install(config)\npipeline.fetch_data(config)\npipeline.prepare_catalog(config)\npipeline.run_tests(config)'),
    ('## 2. Imitation learning', 'Trains the choice policy on human demonstrations. Progress is printed every two minutes, and full logs are in `outputs/logs`.',
     'pipeline.train_il(config)'),
    ('## 3. Reinforcement learning', 'A2C on both GPUs at once. Checkpoints are saved every 15 minutes, and running this cell again resumes from the last one.',
     'pipeline.train_rl(config)'),
    ('## 4. Value model', 'Simulated purchases on training goals, then a linear fit that predicts how well a product matches a request. The value shopper needs it. Runs on the CPU.',
     'pipeline.train_value(config)'),
    ('## 5. Evaluation', 'Every agent on the test goals.',
     'pipeline.evaluate(config)'),
]

FINISH = """
monitor.stop()
report_dir = pipeline.make_report(config)
archive = pipeline.package(config)

from IPython.display import Image, Markdown, display
for name in ["test_scores", "learning_curves", "reward_breakdown", "il_accuracy", "gpu_usage"]:
    figure = report_dir / "figures" / f"{name}.png"
    if figure.exists():
        display(Image(filename=str(figure)))
text = (report_dir / "REPORT.md").read_text(encoding="utf-8")
start = text.find("# 6. Results")
end = text.find("# 7. Discussion")
display(Markdown(text[start:end].replace("figures/", str(report_dir / "figures") + "/")))
print("Submission archive:", archive)
print("Report:", report_dir)
"""

FINAL_NOTE = """
## Before you submit

Complete the two sections of `report/REPORT.md` marked *To complete after the run* (Discussion and Conclusion), and fill in the author and course on its first lines. A Word version is written alongside it when pandoc is available.
"""


def main():
    bundle = bundle_source()
    digest = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode('ascii')
    lines = [encoded[i:i + 120] for i in range(0, len(encoded), 120)]
    bundle_cell = (
        '# The project source code (a compressed archive), so that this one file is all you need.\n'
        '# The readable code is in the project folder; `kaggle/build_notebook.py` regenerates this cell.\n'
        f'SOURCE_SHA256 = "{digest}"\n'
        'SOURCE_BUNDLE = """\n' + '\n'.join(lines) + '\n"""\n'
    )

    cells = [markdown(INTRO), code(CONFIG), code(bundle_cell, hidden=True), code(UNPACK), code(GPU_CHECK)]
    for title, description, source in STAGES:
        cells += [markdown(f'{title}\n\n{description}'), code(source)]
    cells += [markdown('## 6. Report and submission archive'), code(FINISH), markdown(FINAL_NOTE)]

    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': True,
                       'dataSources': [], 'language': 'python', 'sourceType': 'notebook'},
        },
        'nbformat': 4,
        'nbformat_minor': 4,
    }
    text = json.dumps(notebook, indent=1)
    if len(text.encode('utf-8')) >= MAX_NOTEBOOK_BYTES:
        raise SystemExit(f'The notebook would be {len(text) / 1e3:.0f} KB, and Kaggle only accepts less than 1 MB.')
    OUT.write_text(text, encoding='utf-8', newline='\n')
    print(f'Wrote {OUT}: {OUT.stat().st_size / 1e3:.0f} KB, source bundle {len(bundle) / 1e3:.0f} KB, sha256 {digest[:12]}')


if __name__ == '__main__':
    main()
