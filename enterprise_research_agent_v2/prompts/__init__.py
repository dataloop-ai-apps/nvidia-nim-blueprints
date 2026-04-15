"""Jinja2 prompt templates for AIQ v2 agents."""

import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape([]),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_prompt(template_name: str) -> str:
    """Load a raw Jinja2 template string by filename (e.g. 'orchestrator.j2')."""
    return _env.loader.get_source(_env, template_name)[0]


def render_prompt(template_name: str, **kwargs) -> str:
    """Render a Jinja2 prompt template with the given variables."""
    tmpl = _env.get_template(template_name)
    return tmpl.render(**kwargs)
