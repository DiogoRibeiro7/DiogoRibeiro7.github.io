import re

class Post(dict):
    """Simple container for front matter metadata and content."""
    def __init__(self, content='', **metadata):
        super().__init__(metadata)
        self.content = content

def _parse_yaml(yaml_str):
    data = {}
    for line in yaml_str.splitlines():
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
    return data

def _to_yaml(data):
    lines = [f"{k}: {v}" for k, v in data.items()]
    return '\n'.join(lines)

def dumps(post):
    fm = _to_yaml(dict(post))
    return f"---\n{fm}\n---\n{post.content}"

def load(fp):
    if hasattr(fp, 'read'):
        text = fp.read()
    else:
        with open(fp, 'r', encoding='utf-8') as f:
            text = f.read()
    match = re.match(r'^---\n(.*?)\n---\n?(.*)', text, re.DOTALL)
    if match:
        fm_yaml, body = match.group(1), match.group(2)
        metadata = _parse_yaml(fm_yaml)
    else:
        body = text
        metadata = {}
    return Post(content=body, **metadata)
