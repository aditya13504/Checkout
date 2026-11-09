"""A selected option must be shown as selected, including options that contain a quote."""
import json
from types import SimpleNamespace

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


def observe(options, labels):
    html = '<html><body>' + ''.join(f'<label>{label}</label>' for label in labels) + '</body></html>'
    env = WebAgentTextEnv.__new__(WebAgentTextEnv)
    env.browser = SimpleNamespace(current_url='http://x/item_page/s/B01/kw/1/' + json.dumps(options, ensure_ascii=False),
                                  page_source=html)
    env.instruction_text, env._parsed_html = '', None
    return env.convert_html_to_text(html)


def test_options_with_quotes_show_as_clicked():
    page = observe({'size': '20"x20"'}, ['12"x12"', '20"x20"'])
    assert '[clicked button] 20"x20" [clicked button_]' in page
    assert '[button] 12"x12" [button_]' in page


def test_plain_and_non_ascii_options_still_show_correctly():
    page = observe({'size': 'large', 'color': 'crème'}, ['large', 'x-large', 'crème'])
    assert '[clicked button] large [clicked button_]' in page
    assert '[button] x-large [button_]' in page
    assert '[clicked button] crème [clicked button_]' in page
