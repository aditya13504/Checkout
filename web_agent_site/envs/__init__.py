from gym.envs.registration import register

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

try:
    from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
except ImportError:
    WebAgentSiteEnv = None  # Selenium is only needed for the browser-driven env

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)
