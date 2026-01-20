
from pathlib import Path

def ensure_dir(p: str|Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
