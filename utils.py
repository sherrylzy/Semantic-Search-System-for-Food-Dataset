import yaml
import os

def load_config(config_path='config.yaml'):
    """Loads configuration from the YAML file."""
    print("Loading configuration...")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_api_credentials(config):
    """Sets API credentials as environment variables for LiteLLM."""
    os.environ["OPENAI_API_KEY"] = config['api']['key']
    os.environ["OPENAI_API_BASE"] = config['api']['base']