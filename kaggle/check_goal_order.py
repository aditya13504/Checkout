"""
Checks that the environment's goal order matches data/human_goals.json.

Imitation learning assigns demonstrations to the train/eval/test splits by their
position in that file, and the environment does so by its own goal order. If the two
disagree, test goals leak into training, so a mismatch stops the run here.

Run from baseline_models/, with the WEBSHOP_* paths set.
"""
import json
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, join(dirname(abspath(__file__)), '..'))
sys.path.insert(0, join(dirname(abspath(__file__)), '..', 'baseline_models'))

from train_choice_il import process_goal  # noqa: E402
from web_agent_site.envs.web_agent_text_env import SimServer  # noqa: E402
from web_agent_site.utils import DEFAULT_FILE_PATH  # noqa: E402


def main():
    with open('./data/human_goals.json', encoding='utf-8') as f:
        expected = json.load(f)
    server = SimServer('http://x', DEFAULT_FILE_PATH, human_goals=1)
    actual = [process_goal(goal['instruction_text']) for goal in server.goals]
    if len(actual) != len(expected):
        sys.exit(f'{len(actual)} goals in the environment, {len(expected)} expected')
    mismatches = sum(a != b for a, b in zip(actual, expected))
    in_test = sum(a != b for a, b in zip(actual[:500], expected[:500]))
    print(f'{len(actual)} goals; {mismatches} differ from human_goals.json ({in_test} in the test split)', flush=True)
    if mismatches:
        print('First differences:', flush=True)
        shown = 0
        for i, (a, b) in enumerate(zip(actual, expected)):
            if a != b and shown < 5:
                print(f'  {i}: {a[:90]!r} vs {b[:90]!r}', flush=True)
                shown += 1
        sys.exit('The goal order does not match human_goals.json')
    print('Goal order matches human_goals.json', flush=True)


if __name__ == '__main__':
    main()
