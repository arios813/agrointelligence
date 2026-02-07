"""Create a release archive for the project without requiring git.

This script packages the repository root into `releases/agrointellegence-0.1.0.tar.gz`,
excluding large output folders like `outputs/` and `models/`.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "releases"
RELEASES.mkdir(exist_ok=True)
ARCHIVE_NAME = RELEASES / "agrointellegence-0.1.0"

def exclude_func(tarinfo):
    # Exclude outputs and models directories
    p = Path(tarinfo.name)
    parts = p.parts
    if len(parts) > 0 and parts[0] in {"outputs", "models", "releases", "venv"}:
        return None
    return tarinfo

def make_archive():
    print(f"Creating archive {ARCHIVE_NAME}.tar.gz")
    shutil.make_archive(str(ARCHIVE_NAME), 'gztar', root_dir=str(ROOT), base_dir='.', logger=None)
    print("Archive created.")

if __name__ == '__main__':
    make_archive()
