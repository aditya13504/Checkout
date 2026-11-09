"""
Live demo: watch an agent shop in the real WebShop store, one step at a time, in the browser.

    data\\venv\\Scripts\\python.exe demo\\live_demo.py        (or double-click demo\\START_DEMO.bat)

Then open http://127.0.0.1:3100, pick an agent and a test request, and press Start. Every page shown
is the page the agent actually received from the environment, and every decision is made live by the
trained policy (or the value model) on this machine. Nothing is replayed from a file.

Needs the prepared full catalogue in data/, which run.ps1 or run.sh builds.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PREPARED = DATA / 'prepared_test'
DEMO = Path(__file__).resolve().parent
RUN = ROOT / 'kaggle_outputs' / 'outputs_full'  # the Kaggle run reported in the report
RUNS, RESULTS = RUN / 'runs', RUN / 'results'

# The environment reads these when it is imported, so they are set first.
os.environ.setdefault('WEBSHOP_ITEMS_PATH', str(PREPARED / 'items.json'))
os.environ.setdefault('WEBSHOP_ATTR_PATH', str(PREPARED / 'attrs.json'))
os.environ.setdefault('WEBSHOP_SQLITE_INDEX', str(PREPARED / 'webshop.sqlite'))
os.environ.setdefault('WEBSHOP_HUMAN_PATH', str(DATA / 'items_human_ins.json'))
os.environ.setdefault('WEBSHOP_PRODUCT_STORE', str(DATA / 'products.sqlite'))
os.environ.setdefault('HF_HOME', str(DATA / 'hf'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT / 'baseline_models'))
os.chdir(ROOT / 'baseline_models')  # the agents' data paths are relative to this folder

# (label, kind, weights, how search pages are handled, recorded test results)
AGENTS = {
    'value_shopper': ('Value shopper (learned value model)', 'value', RUNS / 'value' / 'value_model.json', None,
                      RESULTS / 'value_shopper.json'),
    'il_rl_greedy': ('IL+RL: BERT policy fine-tuned with A2C', 'model', RUNS / 'il_rl' / 'model_best.pt', 'model',
                     RESULTS / 'il_rl_greedy.json'),
    'rl_scratch_greedy': ('RL from pretrained BERT (A2C only)', 'model', RUNS / 'rl_scratch' / 'model_best.pt', 'model',
                          RESULTS / 'rl_scratch_greedy.json'),
    'il_seed0_greedy': ('IL: BERT policy copying people', 'model', RUNS / 'il_seed0' / 'model_best.pth', 'top1',
                        RESULTS / 'il_seed0_greedy.json'),
    'rule_softmax': ('Rule baseline', 'rule', None, None, RESULTS / 'rule_softmax.json'),
}
STAGES = {'scan': 'Reading the search results', 'inspect': 'Opening the most promising products',
          'buy': 'Going back to buy the best one'}


class Demo:
    def __init__(self):
        self.lock = threading.Lock()
        self.events, self.status, self.env, self.goals = [], 'Loading the store: 1,181,430 products…', None, []
        self.run_id, self.delay, self.stop = 0, 1.0, False
        self.models, self.recorded = {}, {}
        self.thread = None

    # ------------------------------------------------------------------ setup
    def load(self):
        try:
            from env import WebEnv
            args = SimpleNamespace(state_format='text_rich', click_item_name=1, num_prev_obs=0, num_prev_actions=0,
                                   ban_buy=0, go_to_item=0, go_to_search=0, harsh_reward=0, num=None, human_goals=1,
                                   get_image=0, step_limit=100, extra_search_path='./data/goal_query_predict.json',
                                   assist=0)
            t0 = time.time()
            self.env = WebEnv(args, split='test', id='demo_')
            self.goals = [g['instruction_text'] for g in self.env.env.server.goals[:500]]
            for key, (*_, path) in AGENTS.items():
                if path.exists():
                    with open(path, encoding='utf-8') as f:
                        self.recorded[key] = {r['goal_idx']: r['score'] for r in json.load(f)['results']}
            self.status = 'ready'
            print(f'Store ready in {time.time() - t0:.0f} s. Open the viewer and press Start.', flush=True)
        except Exception:
            self.status = 'The store failed to load:\n' + traceback.format_exc()
            print(self.status, flush=True)

    def model(self, key):
        if key not in self.models:
            import torch
            from agent import Agent
            from test import add_agent_defaults
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            args = add_agent_defaults(SimpleNamespace(get_image=0, amp=0, truncation='right'))
            agent = Agent(args)
            agent.load_weights(str(AGENTS[key][2]))
            self.models[key] = agent
        return self.models[key]

    # ------------------------------------------------------------- publishing
    def publish(self, run_id, event):
        with self.lock:
            if run_id == self.run_id:
                event['run'] = run_id
                self.events.append(event)

    def target(self, action):
        """What to highlight on the page: the search box, a product, a button or an option."""
        if action.startswith('search['):
            return {'kind': 'search', 'text': action[7:-1]}
        if action.startswith('click['):
            arg = action[6:-1]
            if arg.startswith('item - '):
                return {'kind': 'item', 'text': self.env.name2asin.get(arg[7:], arg[7:])}
            return {'kind': 'click', 'text': arg}
        return {'kind': 'none', 'text': ''}

    # ---------------------------------------------------------------- deciding
    def policy(self, key):
        kind = AGENTS[key][1]
        if kind == 'rule':
            from test import rule_policy

            def act(ob, info):
                action = rule_policy([ob], [info])[0]
                why = ('Searches with the whole request, then buys the first result without choosing options.')
                return action, {'kind': 'rule', 'why': why}
            return act
        if kind == 'value':
            from value_model import ValueShopper, ValueWeights
            # The settings of the reported evaluation (kaggle/pipeline.py, evaluate).
            shopper = ValueShopper(ValueWeights.load(str(AGENTS[key][2])), inspect=14, pages=2, read_details=True,
                                   step_limit=100)

            def act(ob, info):
                action = shopper([ob], [info])[0]
                trip = shopper.trips[0]
                scored = sorted((c for c in trip.found.values() if c.value is not None), key=lambda c: -c.value)
                shortlist = sorted(trip.found.values(), key=lambda c: -c.shortlist_value)[:6]
                return action, {
                    'kind': 'value', 'stage': STAGES.get(trip.stage, trip.stage), 'seen': len(trip.found),
                    'left': len(trip.queue), 'opening': trip.opening.title if trip.opening else None,
                    'scored': [[c.title, round(c.value, 1)] for c in scored[:8]],
                    'shortlist': [[c.title, round(c.shortlist_value, 1)] for c in shortlist] if trip.stage != 'scan' else [],
                    'best': trip.best.title if trip.best else None,
                }
            return act

        import torch
        agent, search = self.model(key), AGENTS[key][3]

        def act(ob, info):
            valid = info['valid']
            if len(valid) == 1:
                return valid[0], {'kind': 'forced', 'why': 'Only one action is possible here.'}
            if valid[0].startswith('search[') and search == 'top1':
                return valid[0], {'kind': 'forced', 'why': 'Uses the first search query predicted for this request.'}
            states, ids = agent.build_states([ob], [info]), agent.encode_valids([valid])
            with torch.no_grad():
                log_probs, _, values = agent.network.rl_forward(states, ids, value=True, act=True)
            probs = log_probs.float().exp().tolist()
            best = max(range(len(probs)), key=probs.__getitem__)  # greedy, as in the reported runs
            ranked = sorted(zip(valid, probs), key=lambda x: -x[1])[:8]
            return valid[best], {'kind': 'policy', 'probs': [[a, round(p, 4)] for a, p in ranked],
                                 'n': len(valid), 'value': round(float(values.float().reshape(-1)[0]), 2)}
        return act

    # ----------------------------------------------------------------- running
    def run(self, run_id, key, goal_idx):
        try:
            label = AGENTS[key][0]
            self.publish(run_id, {'type': 'start', 'agent': label, 'key': key, 'goal': goal_idx,
                                  'instruction': self.goals[goal_idx],
                                  'recorded': self.recorded.get(key, {}).get(goal_idx)})
            act = self.policy(key)
            env = self.env
            ob, info = env.reset(goal_idx)
            step = 0
            while not self.stop:
                info.update(env_id=0, goal_idx=goal_idx)
                action, think = act(ob, info)
                step += 1
                self.publish(run_id, {'type': 'step', 'n': step, 'action': action, 'target': self.target(action),
                                      'url': env.env.state['url'], 'html': env.env.state['html'], 'think': think})
                self.wait()
                ob, _, done, info = env.step(action)
                if done:
                    verbose = info.get('verbose', {})
                    self.publish(run_id, {
                        'type': 'done', 'score': round(float(info['score']) * 10, 1), 'steps': info.get('steps', step),
                        'bought': bool(info.get('bought')), 'html': env.env.state['html'],
                        'parts': {k: round(float(verbose.get(k, 0.0)), 3) for k in ('r_type', 'r_att', 'r_option', 'r_price')}})
                    return
        except Exception:
            self.publish(run_id, {'type': 'error', 'message': traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    def wait(self):
        end = time.time() + self.delay
        while time.time() < end and not self.stop:
            time.sleep(0.03)

    def start(self, key, goal_idx):
        if self.thread and self.thread.is_alive():
            self.stop = True
            self.thread.join()
        with self.lock:
            self.run_id += 1
            self.events = []
        self.stop = False
        self.thread = threading.Thread(target=self.run, args=(self.run_id, key, goal_idx), daemon=True)
        self.thread.start()


DEMO_STATE = Demo()


STATIC_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
                '.json': 'application/json', '.md': 'text/plain; charset=utf-8', '.png': 'image/png',
                '.svg': 'image/svg+xml', '.pdf': 'application/pdf'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind='application/json', code=200):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            return self.send((DEMO / 'viewer.html').read_bytes(), 'text/html; charset=utf-8')
        if url.path.startswith('/static/'):
            path = (ROOT / 'web_agent_site' / url.path.lstrip('/')).resolve()
            if path.is_file() and (ROOT / 'web_agent_site' / 'static') in path.parents:
                kind = 'text/css' if path.suffix == '.css' else 'application/octet-stream'
                return self.send(path.read_bytes(), kind)
            return self.send('{}', code=404)
        if url.path in ('/showcase', '/showcase/'):
            return self.send((ROOT / 'showcase' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        # The results page and the report it links to, read-only and limited to their own folders.
        for folder in ('showcase', 'submission'):
            if url.path.startswith(f'/{folder}/'):
                base = (ROOT / folder).resolve()
                path = (ROOT / unquote(url.path).lstrip('/')).resolve()
                if path.is_file() and base in path.parents:
                    return self.send(path.read_bytes(), STATIC_TYPES.get(path.suffix, 'application/octet-stream'))
                return self.send('{}', code=404)
        if url.path == '/api/setup':
            d = DEMO_STATE
            return self.send(json.dumps({
                'status': d.status, 'goals': d.goals if d.status == 'ready' else [],
                'agents': [{'key': k, 'label': v[0]} for k, v in AGENTS.items()],
                'recorded': {k: {str(g): s for g, s in r.items()} for k, r in d.recorded.items()}}))
        if url.path == '/api/events':
            since = int(parse_qs(url.query).get('since', ['0'])[0])
            run = int(parse_qs(url.query).get('run', ['0'])[0])
            with DEMO_STATE.lock:
                current = DEMO_STATE.run_id
                events = DEMO_STATE.events[since:] if run == current else DEMO_STATE.events
            return self.send(json.dumps({'run': current, 'events': events, 'reset': run != current}))
        return self.send('{}', code=404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length) or b'{}')
        url = urlparse(self.path)
        if url.path == '/api/start':
            if DEMO_STATE.status != 'ready':
                return self.send(json.dumps({'error': 'The store is still loading.'}), code=409)
            key, goal = body.get('agent'), int(body.get('goal', 0))
            if key not in AGENTS or not 0 <= goal < len(DEMO_STATE.goals):
                return self.send(json.dumps({'error': 'Unknown agent or request.'}), code=400)
            DEMO_STATE.delay = float(body.get('delay', DEMO_STATE.delay))
            DEMO_STATE.start(key, goal)
            return self.send(json.dumps({'run': DEMO_STATE.run_id}))
        if url.path == '/api/speed':
            DEMO_STATE.delay = max(0.0, min(5.0, float(body.get('delay', 1.0))))
            return self.send('{}')
        if url.path == '/api/stop':
            DEMO_STATE.stop = True
            return self.send('{}')
        return self.send('{}', code=404)


def main():
    parser = argparse.ArgumentParser(description='WebShop live agent demo')
    parser.add_argument('--port', type=int, default=3100)
    parser.add_argument('--no_browser', action='store_true')
    args = parser.parse_args()
    for path in (PREPARED / 'webshop.sqlite', DATA / 'products.sqlite'):
        if not path.exists():
            sys.exit(f'{path} is missing. Run run.ps1 (Windows) or run.sh (macOS) first.')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=DEMO_STATE.load, daemon=True).start()
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Live demo at {url}  (loading the store takes a minute or two; press Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
