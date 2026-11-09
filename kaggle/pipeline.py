"""
The whole experiment, as resumable stages. The Kaggle notebook only calls these
functions, and `--local` runs the same code without a GPU for testing.

Each finished stage leaves a marker in <outputs>/stages, so running again skips it.
This file only needs the standard library; all real work happens in a separate
Python 3.10 environment with exact package versions.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIRRORS = ['YWZBrandon/webshop-data', 'Yuxuan13/webshop_dataset', 'quanwei0/webshop-minimal']
# The original Google Drive links are dead. These are the hashes that all three mirrors agree on.
EXPECTED_SHA256 = {
    'items_shuffle.json': '2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8',
    'items_ins_v2.json': '1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e',
}
UPSTREAM_DATA = 'https://raw.githubusercontent.com/princeton-nlp/WebShop/master/baseline_models/data/'
BASELINE_DATA_FILES = ['human_goals.json', 'goal_query_predict.json', 'goal_query_map.json',
                       'items_human_ins.json', 'il_trajs_finalized_images.zip']
TORCH_INDEX = 'https://download.pytorch.org/whl/cu121'
SPACY_MODEL_URL = ('https://github.com/explosion/spacy-models/releases/download/'
                   'en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl')


@dataclass
class Config:
    mode: str = 'full'                 # 'full' or 'smoke' (a quick check of the whole pipeline)
    project_dir: str = '/kaggle/working/webshop'
    outputs_dir: str = '/kaggle/working/outputs'
    data_dir: str = '/tmp/webshop_data'
    venv_dir: str = '/tmp/webshop_venv'

    catalog_size: int = 200_000        # training catalogue, keeping all human-goal products; 0 = all
    test_on_full_catalog: bool = True  # test on all 1.18 million products, as the paper does
    il_epochs: int = 10
    il_batch_size: int = 32
    il_max_minutes: float = 60
    rl_hours: float = 2.5
    value_goals: int = 2000            # training goals whose search results the value model learns from
    value_inspect: int = 14            # products the value shopper opens per goal (chosen on validation goals)
    value_pages: int = 2               # results pages it reads per search
    num_envs: int = 16
    bptt: int = 8
    learning_rate: float = 1e-5
    workers_per_run: int = 2
    eval_episodes_during_training: int = 100
    test_goals: int = 0                # 0 = all 500
    gpu_ids: list = field(default_factory=list)
    device_torch: str = 'cuda'         # 'cuda' or 'cpu'
    smoke_catalog: bool = False


def smoke_settings(config):
    config.il_epochs = 1
    config.il_batch_size = 8
    config.il_max_minutes = 5
    config.rl_hours = 3 / 60
    config.value_goals = 20
    config.num_envs = 4
    config.eval_episodes_during_training = 8
    config.test_goals = 16
    config.smoke_catalog = True
    return config


# ---------------------------------------------------------------- utilities

def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def python(config):
    folder = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    return str(Path(config.venv_dir) / folder)


def run(command, cwd=None, env=None, check=True):
    log('$ ' + ' '.join(str(c) for c in command)[:200])
    return subprocess.run([str(c) for c in command], cwd=cwd, env=env, check=check)


def stage(name):
    """Runs a stage once; its marker file makes later calls skip it."""
    def decorate(function):
        def wrapper(config, *args, **kwargs):
            marker = Path(config.outputs_dir) / 'stages' / name
            if marker.exists():
                log(f'Stage "{name}" is already done')
                return None
            start = time.time()
            log(f'=== {name} ===')
            result = function(config, *args, **kwargs)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f'{(time.time() - start) / 60:.1f} minutes\n')
            log(f'=== {name} finished in {(time.time() - start) / 60:.1f} min ===')
            return result
        return wrapper
    return decorate


def run_env(config, catalog='train'):
    """Environment variables for every project process."""
    prepared = Path(config.data_dir) / ('prepared_' + catalog)
    env = dict(os.environ)
    env.update({
        'WEBSHOP_ITEMS_PATH': str(prepared / 'items.json'),
        'WEBSHOP_ATTR_PATH': str(prepared / 'attrs.json'),
        'WEBSHOP_SQLITE_INDEX': str(prepared / 'webshop.sqlite'),
        'WEBSHOP_HUMAN_PATH': str(Path(config.data_dir) / 'items_human_ins.json'),
        'HF_HOME': str(Path(config.data_dir) / 'hf'),
        'TOKENIZERS_PARALLELISM': 'true',
        'PYTHONUNBUFFERED': '1',
        'PYTHONUTF8': '1',
    })
    return env


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    run(['curl', '-fsSL', '--retry', '5', '--retry-delay', '5', '-C', '-', '-o', tmp, url])
    os.replace(tmp, path)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def download_from_mirrors(filename, path):
    last_error = None
    for mirror in MIRRORS:
        try:
            download(f'https://huggingface.co/datasets/{mirror}/resolve/main/{filename}', path)
            expected = EXPECTED_SHA256.get(filename)
            if expected and sha256_of(path) != expected:
                Path(path).unlink()
                raise ValueError(f'{filename} from {mirror} has the wrong hash')
            return
        except (subprocess.CalledProcessError, ValueError) as error:
            last_error = error
            log(f'{mirror} failed for {filename}; trying the next mirror')
    raise RuntimeError(f'No mirror could provide {filename}: {last_error}')


# ------------------------------------------------------------------- stages

@stage('install')
def install(config):
    """Creates the isolated Python 3.10 environment with exact versions."""
    requirements = Path(config.project_dir) / 'requirements.txt'
    lines = [line for line in requirements.read_text().splitlines() if not line.startswith('torch')]
    filtered = Path(config.venv_dir + '_requirements.txt')
    filtered.write_text('\n'.join(lines) + '\n')

    if not shutil.which('uv'):
        run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    run(['uv', 'venv', '--python', '3.10', '-q', config.venv_dir, '--allow-existing'])
    py = python(config)
    run(['uv', 'pip', 'install', '-q', '--python', py, '-r', filtered])
    torch_command = ['uv', 'pip', 'install', '-q', '--python', py, 'torch==2.2.2']
    if config.device_torch == 'cuda':
        torch_command += ['--index-url', TORCH_INDEX]
    run(torch_command)
    run(['uv', 'pip', 'install', '-q', '--python', py, f'en_core_web_lg @ {SPACY_MODEL_URL}'])
    if not shutil.which('pandoc') and shutil.which('apt-get'):
        run(['apt-get', 'install', '-y', '-qq', 'pandoc'], check=False)


@stage('data')
def fetch_data(config):
    """Downloads the catalogue, the demonstrations and the pretrained tokenizer."""
    data = Path(config.data_dir)
    baseline = Path(config.project_dir) / 'baseline_models' / 'data'
    for name in BASELINE_DATA_FILES:
        target = baseline / name
        if not target.exists():
            download(UPSTREAM_DATA + name, target)
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline / 'items_human_ins.json', data / 'items_human_ins.json')

    jsonl = baseline / 'il_trajs_finalized_images.jsonl'
    if not jsonl.exists():
        shutil.unpack_archive(str(baseline / 'il_trajs_finalized_images.zip'), str(baseline))

    if config.smoke_catalog:
        items, attrs = 'items_shuffle_1000.json', 'items_ins_v2_1000.json'
    else:
        items, attrs = 'items_shuffle.json', 'items_ins_v2.json'
    download_from_mirrors(items, data / items)
    download_from_mirrors(attrs, data / attrs)

    command = [python(config), '-c',
               "from transformers import AutoTokenizer, BertModel;"
               "AutoTokenizer.from_pretrained('bert-base-uncased');"
               "BertModel.from_pretrained('bert-base-uncased')"]
    run(command, env=run_env(config))


@stage('catalog')
def prepare_catalog(config):
    """Builds the train and test catalogues with their indexes, then checks the goal order."""
    data = Path(config.data_dir)
    items = 'items_shuffle_1000.json' if config.smoke_catalog else 'items_shuffle.json'
    attrs = 'items_ins_v2_1000.json' if config.smoke_catalog else 'items_ins_v2.json'
    Path(config.outputs_dir).mkdir(parents=True, exist_ok=True)

    def build(catalog, max_products):
        command = [python(config), Path(config.project_dir) / 'search_engine' / 'prepare_catalog.py',
                   '--raw_items', data / items, '--raw_attrs', data / attrs,
                   '--human', data / 'items_human_ins.json', '--out_dir', data / f'prepared_{catalog}']
        if max_products:
            command += ['--max_products', max_products]
        run(command, env=run_env(config))
        shutil.copyfile(data / f'prepared_{catalog}' / 'catalog_summary.json',
                        Path(config.outputs_dir) / f'catalog_{catalog}.json')

    train_size = 0 if config.smoke_catalog else config.catalog_size
    build('train', train_size)
    if config.test_on_full_catalog and train_size:
        build('test', 0)
    else:
        target = data / 'prepared_test'
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(data / 'prepared_train', target)
    shutil.copyfile(data / 'prepared_test' / 'catalog_summary.json', Path(config.outputs_dir) / 'catalog_test.json')
    # The raw download is no longer needed.
    (data / items).unlink(missing_ok=True)

    if not config.smoke_catalog:
        check = Path(config.project_dir) / 'kaggle' / 'check_goal_order.py'
        run([python(config), check], cwd=Path(config.project_dir) / 'baseline_models', env=run_env(config))


@stage('tests')
def run_tests(config):
    env = run_env(config)
    env['CUDA_VISIBLE_DEVICES'] = ''
    run([python(config), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--disable-warnings', 'tests'],
        cwd=config.project_dir, env=env)


class GpuMonitor:
    """Records GPU utilization and memory to a CSV file throughout the run."""

    def __init__(self, config, interval=15):
        self.path = Path(config.outputs_dir) / 'gpu_usage.csv'
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if shutil.which('nvidia-smi'):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text('t,gpu,util,mem_used,mem_total\n')
            self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def _loop(self):
        query = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits']
        while not self.stop_event.is_set():
            try:
                output = subprocess.run(query, capture_output=True, text=True, timeout=10).stdout
                now = time.time()
                with open(self.path, 'a') as f:
                    for line in output.strip().splitlines():
                        f.write(f"{now:.0f},{','.join(part.strip() for part in line.split(','))}\n")
            except (subprocess.SubprocessError, OSError):
                pass
            self.stop_event.wait(self.interval)


def run_jobs(config, jobs, cwd, catalog='train', max_parallel=None):
    """Runs (name, command) jobs with one job per GPU at a time, showing progress."""
    gpus = (config.gpu_ids or [None])[:max_parallel]
    logs_dir = Path(config.outputs_dir) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue, running, failed = list(jobs), {}, []
    free = list(gpus)
    last_report = 0
    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            name, command = queue.pop(0)
            env = run_env(config, catalog)
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            log_file = open(logs_dir / f'{name}.log', 'a', encoding='utf-8')
            log(f'Starting {name} on GPU {gpu}')
            process = subprocess.Popen([str(c) for c in command], cwd=cwd, env=env,
                                       stdout=log_file, stderr=subprocess.STDOUT)
            running[name] = (process, gpu, log_file)
        time.sleep(5)
        for name, (process, gpu, log_file) in list(running.items()):
            code = process.poll()
            if code is not None:
                log_file.close()
                del running[name]
                free.append(gpu)
                if code != 0:
                    failed.append(name)
                    killed = ' The system killed it, most likely because memory ran out.' if code in (-9, 137) else ''
                    log(f'{name} FAILED (exit code {code}).{killed} Last lines of its log:')
                    print(tail(logs_dir / f'{name}.log', 25), flush=True)
                else:
                    log(f'{name} finished')
        if time.time() - last_report > 120:
            last_report = time.time()
            for name in running:
                log(f'{name}: {tail(logs_dir / f"{name}.log", 1)}')
    if failed:
        raise RuntimeError(f'Jobs failed: {failed}. Fix the cause and run this stage again; '
                           'training resumes from its last checkpoint.')


def tail(path, n):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20000))
            lines = f.read().decode('utf-8', 'replace').replace('\r', '\n').splitlines()
        return '\n'.join(line[:220] for line in lines[-n:])
    except OSError:
        return ''


def goal_args(config):
    return ['--human_goals', '0'] if config.smoke_catalog else ['--human_goals', '1']


@stage('train_il')
def train_il(config):
    """Imitation learning: one seed per GPU."""
    runs = Path(config.outputs_dir) / 'runs'
    seeds = [0, 1] if len(config.gpu_ids) >= 2 else [0]
    jobs = []
    for seed in seeds:
        command = [python(config), 'train_choice_il.py', '--seed', seed, '--output_dir', runs / f'il_seed{seed}',
                   '--num_train_epochs', config.il_epochs, '--per_device_train_batch_size', config.il_batch_size,
                   '--max_minutes', config.il_max_minutes]
        if config.smoke_catalog:
            command += ['--max_train_samples', 64]
        jobs.append((f'il_seed{seed}', command))
    run_jobs(config, jobs, Path(config.project_dir) / 'baseline_models')


@stage('train_rl')
def train_rl(config):
    """A2C: IL+RL on one GPU and RL from pretrained BERT on the other."""
    runs = Path(config.outputs_dir) / 'runs'

    def command(name, bert_path, seed):
        return [python(config), 'train_rl.py', '--output_dir', runs / name, '--seed', seed, '--resume', 1,
                '--bert_path', bert_path, '--max_hours', config.rl_hours, '--num_envs', config.num_envs,
                '--num_eval_envs', config.num_envs, '--num_workers', config.workers_per_run,
                '--bptt', config.bptt, '--learning_rate', config.learning_rate,
                '--eval_episodes', config.eval_episodes_during_training,
                '--log_freq', 10 if config.smoke_catalog else 50,
                '--eval_every_min', 1 if config.smoke_catalog else 15, *goal_args(config)]

    run_jobs(config, [
        ('il_rl', command('il_rl', runs / 'il_seed0' / 'model_best.pth', 0)),
        ('rl_scratch', command('rl_scratch', '', 1)),
    ], Path(config.project_dir) / 'baseline_models')


@stage('train_value')
def train_value(config):
    """The value model: simulated purchases on training goals, then a linear fit. Needs no GPU."""
    folder = Path(config.outputs_dir) / 'runs' / 'value'
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / 'train.npz'
    cwd = Path(config.project_dir) / 'baseline_models'
    run_jobs(config, [('value_collect', [python(config), 'train_value_model.py', 'collect', '--split', 'train',
                                         '--num_goals', config.value_goals, '--out', data, *goal_args(config)])],
             cwd, catalog='test')
    run([python(config), 'train_value_model.py', 'fit', '--data', data, '--alpha', 100, '--out', folder / 'value_model.json'],
        cwd=cwd, env=run_env(config, catalog='test'))


@stage('evaluate')
def evaluate(config):
    """Evaluates every agent on the test goals, sharing the GPUs."""
    runs = Path(config.outputs_dir) / 'runs'
    results = Path(config.outputs_dir) / 'results'

    def job(name, *extra):
        command = [python(config), 'test.py', '--out', results / f'{name}.json', '--num_workers',
                   config.workers_per_run, '--num_goals', config.test_goals, *goal_args(config), *extra]
        return (name, command)

    def model_jobs(prefix, weights, search):
        return [
            job(f'{prefix}_softmax', '--model_path', weights, '--softmax', 1, '--search', search),
            job(f'{prefix}_greedy', '--model_path', weights, '--softmax', 0, '--search', search),
        ]

    jobs = [job('rule_softmax', '--policy', 'rule')]
    jobs += model_jobs('il_seed0', runs / 'il_seed0' / 'model_best.pth', 'top1')
    jobs.append(job('il_seed0_instruction_softmax', '--model_path', runs / 'il_seed0' / 'model_best.pth',
                    '--search', 'instruction'))
    if (runs / 'il_seed1' / 'model_best.pth').exists():
        jobs.append(job('il_seed1_softmax', '--model_path', runs / 'il_seed1' / 'model_best.pth', '--search', 'top1'))
    jobs += model_jobs('il_rl', runs / 'il_rl' / 'model_best.pt', 'model')
    jobs += model_jobs('rl_scratch', runs / 'rl_scratch' / 'model_best.pt', 'model')
    jobs.append(job('rule_assist', '--policy', 'rule', '--assist', 1))
    jobs.append(job('il_rl_assist_greedy', '--model_path', runs / 'il_rl' / 'model_best.pt', '--softmax', 0,
                    '--search', 'model', '--assist', 1))
    jobs.append(job('careful_shopper', '--policy', 'shopper', '--inspect', 5, '--queries', 'top1,instruction'))
    jobs.append(job('value_shopper', '--policy', 'value', '--value_model', runs / 'value' / 'value_model.json',
                    '--inspect', config.value_inspect, '--pages', config.value_pages))
    jobs = [j for j in jobs if not (results / f'{j[0]}.json').exists()]  # finished before an interruption

    # Every job holds the whole catalogue in memory. Two jobs on the full catalogue failed
    # on Kaggle's T4 x2 machine while one alone finished, so these run one at a time.
    max_parallel = 1 if config.test_on_full_catalog and not config.smoke_catalog else None
    run_jobs(config, jobs, Path(config.project_dir) / 'baseline_models', catalog='test', max_parallel=max_parallel)


def write_meta(config):
    command = ("import json,sys,torch,transformers,spacy;"
               "n=torch.cuda.device_count();"
               "names=[torch.cuda.get_device_name(i) for i in range(n)];"
               "print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,"
               "transformers=transformers.__version__,spacy=spacy.__version__,"
               "gpus=(str(n)+' x '+names[0]) if n else '')))")
    output = subprocess.run([python(config), '-c', command], capture_output=True, text=True,
                            env=run_env(config), check=True).stdout
    meta = json.loads(output.strip().splitlines()[-1])
    meta['mode'] = config.mode
    meta['config'] = asdict(config)
    (Path(config.outputs_dir) / 'meta.json').write_text(json.dumps(meta, indent=2))


def make_report(config):
    write_meta(config)
    project = Path(config.project_dir)
    report_dir = Path(config.outputs_dir) / 'report'
    run([python(config), project / 'report' / 'make_report.py', '--outputs', config.outputs_dir,
         '--out_dir', report_dir], env=run_env(config))
    return report_dir


def package(config):
    """Zips everything to hand in. Model weights are large, so they stay in `runs/`, outside the zip."""
    outputs = Path(config.outputs_dir)
    project = Path(config.project_dir)
    bundle = outputs / 'submission'
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', 'il_trajs_finalized_images.jsonl', '*.part',
                                    'kaggle_outputs*', 'submission')
    shutil.copytree(project, bundle / 'code', ignore=ignore)
    for name in ('report', 'results', 'results_before_fix', 'results_second_run', 'logs'):
        if (outputs / name).exists():
            shutil.copytree(outputs / name, bundle / name)
    runs = outputs / 'runs'
    if runs.exists():
        shutil.copytree(runs, bundle / 'training_logs',
                        ignore=shutil.ignore_patterns('*.pt', '*.pth', '*.tmp'))
    for name in ('gpu_usage.csv', 'meta.json', 'catalog_train.json', 'catalog_test.json'):
        if (outputs / name).exists():
            shutil.copy(outputs / name, bundle / name)
    archive = shutil.make_archive(str(outputs / 'webshop_rl_submission'), 'zip', bundle)
    shutil.rmtree(bundle)
    log(f'Submission archive: {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)')

    # Keep only the weights worth keeping; the resume checkpoints hold an optimizer too.
    for checkpoint in runs.glob('*/ckpt_latest.pt') if runs.exists() else []:
        checkpoint.unlink()
    for final in list(runs.glob('*/model_final.*')) if runs.exists() else []:
        final.unlink()
    return archive


def run_all(config, monitor=True):
    """Every stage, in order."""
    monitor_thread = GpuMonitor(config).start() if monitor else None
    try:
        install(config)
        fetch_data(config)
        prepare_catalog(config)
        run_tests(config)
        train_il(config)
        train_rl(config)
        train_value(config)
        evaluate(config)
        make_report(config)
        return package(config)
    finally:
        if monitor_thread:
            monitor_thread.stop()


def detect_gpus():
    if not shutil.which('nvidia-smi'):
        return []
    output = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                            capture_output=True, text=True).stdout
    return [int(line) for line in output.split()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', default='smoke', choices=['smoke', 'full'])
    parser.add_argument('--project_dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--work_dir', required=True, help='where outputs, data and the environment go')
    parser.add_argument('--venv_dir', default='')
    parser.add_argument('--tiny', action='store_true', help='the smallest possible run, for testing on a CPU')
    args = parser.parse_args()
    work = Path(args.work_dir)
    settings = Config(mode=args.mode, project_dir=args.project_dir, outputs_dir=str(work / 'outputs'),
                      data_dir=str(work / 'data'), venv_dir=args.venv_dir or str(work / 'venv'),
                      gpu_ids=detect_gpus(), device_torch='cuda' if detect_gpus() else 'cpu')
    if args.mode == 'smoke':
        smoke_settings(settings)
    if args.tiny:
        settings.rl_hours = 2 / 60
        settings.num_envs = 2
        settings.eval_episodes_during_training = 2
        settings.test_goals = 3
    run_all(settings)
