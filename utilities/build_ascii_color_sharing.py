"""Explicitly build the optional, small native terminal colour-sharing helper."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascii_color_sharing import SOURCE, library_path


def build():
    target = library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.so', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(compiler + ['-O3', '-std=c99', '-shared', '-fPIC',
            '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
            str(SOURCE), '-o', str(temporary), '-lm'], check=True)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


if __name__ == '__main__':
    print(build())
