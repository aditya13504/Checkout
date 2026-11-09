"""
Tests of how the Kaggle pipeline schedules evaluation jobs.
"""
import base64
import hashlib
import inspect
import io
import json
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kaggle'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'report'))

import make_report  # noqa: E402
import pipeline  # noqa: E402


NOTEBOOK = Path(__file__).resolve().parents[2] / 'kaggle' / 'webshop_rl_kaggle.ipynb'


@pytest.mark.skipif(not NOTEBOOK.exists(), reason='the notebook does not carry a copy of itself')
def test_notebook_fits_kaggle_and_its_bundle_unpacks():
    assert NOTEBOOK.stat().st_size < 1_000_000  # Kaggle refuses to commit a kernel of 1 MB or more
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    source = '\n'.join(''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code')
    data = base64.b64decode(re.search(r'SOURCE_BUNDLE = """\n(.*?)\n"""', source, re.S).group(1))
    assert hashlib.sha256(data).hexdigest() == re.search(r'SOURCE_SHA256 = "(\w+)"', source).group(1)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:  # the way the notebook opens it
        assert 'kaggle/pipeline.py' in tar.getnames()
        # Unpacked files end up in the submission zip, which cannot store dates before 1980.
        assert min(member.mtime for member in tar.getmembers()) >= time.mktime((1980, 1, 2, 0, 0, 0, 0, 0, -1))


@pytest.mark.skipif(not NOTEBOOK.exists(), reason='the notebook does not carry a copy of itself')
def test_notebook_runs_every_stage_of_the_pipeline():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    source = '\n'.join(''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code')
    stages = re.findall(r'^\s+(?:return )?(\w+)\(config\)', inspect.getsource(pipeline.run_all), re.M)
    assert 'train_value' in stages
    assert [stage for stage in stages if f'pipeline.{stage}(config)' not in source] == []


def make_config(tmp_path, **kwargs):
    return pipeline.Config(project_dir=str(tmp_path), outputs_dir=str(tmp_path / 'outputs'),
                           data_dir=str(tmp_path / 'data'), venv_dir=str(tmp_path / 'venv'), **kwargs)


def scheduled_parallelism(config, monkeypatch):
    seen = {}

    def fake_run_jobs(config, jobs, cwd, catalog='train', max_parallel=None):
        seen.update(catalog=catalog, max_parallel=max_parallel)

    monkeypatch.setattr(pipeline, 'run_jobs', fake_run_jobs)
    pipeline.evaluate(config)
    return seen


def test_full_catalogue_evaluates_one_agent_at_a_time(tmp_path, monkeypatch):
    seen = scheduled_parallelism(make_config(tmp_path, gpu_ids=[0, 1]), monkeypatch)
    assert seen == {'catalog': 'test', 'max_parallel': 1}


def test_submission_zip_keeps_logs_and_leaves_out_downloads(tmp_path):
    project = tmp_path / 'project'
    for name in ('main.py', 'kaggle_outputs/huge.pt', 'submission/old.zip'):
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_text('x')
    outputs = tmp_path / 'outputs'
    for name in ('logs/job.log', 'report/REPORT.md', 'results/agent.json'):
        (outputs / name).parent.mkdir(parents=True, exist_ok=True)
        (outputs / name).write_text('x')
    config = pipeline.Config(project_dir=str(project), outputs_dir=str(outputs))
    with zipfile.ZipFile(pipeline.package(config)) as archive:
        names = archive.namelist()
    assert {'code/main.py', 'logs/job.log', 'report/REPORT.md', 'results/agent.json'} <= set(names)
    assert not [name for name in names if 'kaggle_outputs' in name or 'old.zip' in name]


def fake_run(mode='full', **changed):
    scores = {**make_report.INTERPRETED_RUN, **changed}
    return {'meta': {'mode': mode}, 'results': {name: {'summary': {'score': score}} for name, score in scores.items()}}


def test_written_interpretation_only_appears_for_the_run_it_describes():
    assert 'To complete after the run' not in make_report.interpretation(fake_run())['DISCUSSION']
    assert 'To complete after the run' in make_report.interpretation(fake_run(il_rl_softmax=70.0))['DISCUSSION']
    assert 'To complete after the run' in make_report.interpretation(fake_run(value_shopper=70.0))['DISCUSSION']
    assert 'To complete after the run' in make_report.interpretation(fake_run(mode='smoke'))['DISCUSSION']


def test_a_run_without_the_added_agents_gets_no_interpretation():
    run = fake_run()
    del run['results']['value_shopper']
    assert 'To complete after the run' in make_report.interpretation(run)['DISCUSSION']


def test_gpu_stages_are_placed_from_the_stage_timings(tmp_path):
    for name, minutes in [('install', 1), ('data', 1), ('catalog', 2), ('tests', 2), ('train_il', 60),
                          ('train_rl', 120), ('evaluate', 30)]:
        (tmp_path / name).write_text(f'{minutes}.0 minutes\n')
    windows = make_report.stage_windows(tmp_path)
    assert windows['Imitation learning'] == pytest.approx((0.1, 1.1))
    assert windows['Reinforcement learning'] == pytest.approx((1.1, 3.1))
    assert windows['Evaluation'] == pytest.approx((3.1, 3.6))


def test_small_catalogue_still_uses_both_gpus(tmp_path, monkeypatch):
    config = make_config(tmp_path, gpu_ids=[0, 1], smoke_catalog=True)
    assert scheduled_parallelism(config, monkeypatch)['max_parallel'] is None


def test_a_killed_job_is_reported_as_out_of_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.time, 'sleep', lambda seconds: None)
    job = ('killed', [sys.executable, '-c', 'import sys; sys.exit(137)'])
    try:
        pipeline.run_jobs(make_config(tmp_path), [job], tmp_path)
    except RuntimeError as error:
        assert 'killed' in str(error)
    else:
        raise AssertionError('a failed job must raise')
    assert 'memory' in capsys.readouterr().out
