import pytest

collect_ignore = ['kaggle_outputs', 'kaggle_outputs_run1', 'submission', 'data']  # downloaded runs, hand-in files, local data


@pytest.fixture(autouse=True)
def _no_disk_catalogue(monkeypatch):
    """Tests build their own small catalogues. A WEBSHOP_PRODUCT_STORE set to run the demo
    (as run.ps1 and run.sh do) would otherwise replace them with the full catalogue."""
    import web_agent_site.engine.engine as engine
    monkeypatch.setattr(engine, 'PRODUCT_STORE_PATH', '')
