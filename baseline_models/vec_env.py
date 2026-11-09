"""
Runs many WebEnvs across worker processes, so a whole batch of environments can be
stepped between GPU forward passes.

The catalogue is loaded once and shared with the workers by fork (copy-on-write).
Where fork doesn't exist (Windows) each worker loads its own copy.
"""
import gc
import multiprocessing as mp
import random
import traceback

from env import WebEnv, goal_range


def make_server(args):
    """Loads the catalogue, goals and search engine."""
    return WebEnv(args, split='train', id='loader_').env.server


def step_env(env, action):
    """Steps an env. A failure ends only that episode, not hours of training."""
    try:
        return env.step(action)
    except Exception as e:
        print(f'Env error on action {action[:80]!r}: {e}\n{traceback.format_exc()}', flush=True)
        return '', 0.0, True, {'valid': ['finish'], 'goal': '', 'score': 0.0, 'verbose': {}, 'env_error': str(e)}


def worker_main(conn, args, env_ids, seed, server):
    try:
        import torch
        torch.set_num_threads(1)
        if server is None:
            server = make_server(args)
        envs = {i: WebEnv(args, split='train', server=server, id=f'env{i}_') for i in env_ids}
        random.seed(seed)
        conn.send(('ready', len(server.goals)))
        while True:
            command, payload = conn.recv()
            if command == 'reset':
                conn.send(('ok', [(i, *envs[i].reset(idx)) for i, idx in payload]))
            elif command == 'step':
                conn.send(('ok', [(i, *step_env(envs[i], action)) for i, action in payload]))
            elif command == 'close':
                break
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


class VecWebEnv:
    """`n_envs` environments addressed by id, on `n_workers` processes (0 = in-process)."""

    def __init__(self, args, n_envs, n_workers=0, seed=0):
        self.n_envs = n_envs
        self.n_workers = min(n_workers, n_envs)
        if self.n_workers == 0:
            server = make_server(args)
            self.n_goals = len(server.goals)
            self.envs = {i: WebEnv(args, split='train', server=server, id=f'env{i}_') for i in range(n_envs)}
            random.seed(seed)
            return

        use_fork = 'fork' in mp.get_all_start_methods()
        server = make_server(args) if use_fork else None
        if use_fork:
            # Keeps Python's garbage collector from copying the shared catalogue pages.
            gc.collect()
            gc.freeze()
        context = mp.get_context('fork' if use_fork else 'spawn')
        self.connections, self.processes, self.owner = [], [], {}
        for w in range(self.n_workers):
            env_ids = list(range(n_envs))[w::self.n_workers]
            for i in env_ids:
                self.owner[i] = w
            parent, child = context.Pipe()
            process = context.Process(target=worker_main,
                                      args=(child, args, env_ids, seed * 1000 + w, server),
                                      daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self.n_goals = [self._receive(conn) for conn in self.connections][0]

    @staticmethod
    def _receive(conn):
        status, payload = conn.recv()
        if status == 'error':
            raise RuntimeError(f'Env worker failed:\n{payload}')
        return payload

    def goal_range(self, split):
        return goal_range(self.n_goals, split)

    def _call(self, command, requests):
        by_worker = {}
        for request in requests:
            by_worker.setdefault(self.owner[request[0]], []).append(request)
        for w, payload in by_worker.items():
            self.connections[w].send((command, payload))
        results = {}
        for w in by_worker:
            for result in self._receive(self.connections[w]):
                results[result[0]] = result[1:]
        return results

    def reset(self, requests):
        """`requests` is [(env_id, goal_idx), ...]; returns {env_id: (ob, info)}."""
        if self.n_workers == 0:
            return {i: self.envs[i].reset(idx) for i, idx in requests}
        return self._call('reset', requests)

    def step(self, requests):
        """`requests` is [(env_id, action), ...]; returns {env_id: (ob, r, done, info)}."""
        if self.n_workers == 0:
            return {i: step_env(self.envs[i], action) for i, action in requests}
        return self._call('step', requests)

    def close(self):
        if self.n_workers == 0:
            return
        for conn in self.connections:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def run_episodes(vec, env_ids, goal_idxs, act_batch):
    """Plays each goal once, using `env_ids` in parallel.

    `act_batch(obs_list, infos)` returns one action per observation. Returns one
    result per goal.
    """
    pending = list(goal_idxs)
    results, active = [], {}

    def start(free_ids):
        requests = []
        for env_id in free_ids:
            if pending:
                requests.append((env_id, pending.pop(0)))
        for env_id, idx in requests:
            active[env_id] = {'goal_idx': idx}
        if requests:
            for env_id, (ob, info) in vec.reset(requests).items():
                active[env_id].update(ob=ob, info=info)

    start(env_ids)
    while active:
        ids = list(active)
        for i in ids:  # lets a policy keep notes per episode
            active[i]['info'].update(env_id=i, goal_idx=active[i]['goal_idx'])
        actions = act_batch([active[i]['ob'] for i in ids], [active[i]['info'] for i in ids])
        outcomes = vec.step(list(zip(ids, actions)))
        finished = []
        for env_id, (ob, reward, done, info) in outcomes.items():
            if done:
                verbose = info.get('verbose', {})
                results.append({
                    'goal_idx': active[env_id]['goal_idx'],
                    'score': float(info['score']) * 10,  # 0-100, as in the paper
                    'success': float(info['score']) == 10.0,
                    'steps': info.get('steps', 0),
                    'bought': bool(info.get('bought', False)),
                    'r_att': float(verbose.get('r_att', 0.0)),
                    'r_option': float(verbose.get('r_option', 0.0)),
                    'r_price': float(verbose.get('r_price', 0.0)),
                    'r_type': float(verbose.get('r_type', 0.0)),
                    'n_search': info.get('actions', {}).get('search', 0),
                    'n_items': info.get('actions', {}).get('asin', 0),
                    'n_options': info.get('actions', {}).get('options', 0),
                })
                del active[env_id]
                finished.append(env_id)
            else:
                active[env_id].update(ob=ob, info=info)
        start(finished)
    results.sort(key=lambda r: r['goal_idx'])
    return results
