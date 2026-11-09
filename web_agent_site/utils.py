import bisect
import hashlib
import logging
import os
import random
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable


def _path_setting(env_name, default_relative):
    """A data path that can be overridden with an environment variable."""
    return os.environ.get(env_name) or join(BASE_DIR, default_relative)


# Point these at another catalogue (e.g. the full one) by setting the environment
# variables, so that no source edit is needed between setup and indexing.
DEFAULT_ATTR_PATH = _path_setting('WEBSHOP_ATTR_PATH', '../data/items_ins_v2_1000.json')
DEFAULT_FILE_PATH = _path_setting('WEBSHOP_ITEMS_PATH', '../data/items_shuffle_1000.json')
DEFAULT_REVIEW_PATH = _path_setting('WEBSHOP_REVIEW_PATH', '../data/reviews.json')
HUMAN_ATTR_PATH = _path_setting('WEBSHOP_HUMAN_PATH', '../data/items_human_ins.json')
SQLITE_INDEX_PATH = _path_setting('WEBSHOP_SQLITE_INDEX', '../search_engine/webshop.sqlite')
PRODUCT_STORE_PATH = os.environ.get('WEBSHOP_PRODUCT_STORE', '')  # the catalogue on disk; see engine/product_store.py

FEAT_CONV = _path_setting('WEBSHOP_FEAT_CONV', '../data/feat_conv.pt')
FEAT_IDS = _path_setting('WEBSHOP_FEAT_IDS', '../data/feat_ids.pt')


def random_idx(cum_weights):
    """Sample an index in proportion to its weight.

    `cum_weights` is `[0, w0, w0 + w1, ...]`, so `bisect` lands one position past
    the sampled index. The `min` guards the rare draw of exactly the total weight.
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos) - 1
    idx = min(idx, len(cum_weights) - 2)
    return idx


def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def close_logger(session_id):
    """Closes a session's log file and forgets its logger."""
    logger = logging.getLogger(session_id)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logging.root.manager.loggerDict.pop(session_id, None)


def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()
