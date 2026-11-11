"""Prepare the vendored N3 model beside its meshes, as download_data.sh does for RP1.

    python tools/data/n3_assets.py
"""

from pathlib import Path
import shutil


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    src, dst = repo / "assets/n3_29dof_mjlab", repo / "data/n3_29dof_mjlab"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "n3_29dof.xml", dst / "n3_29dof.xml")
    shutil.copytree(src / "meshes", dst / "meshes", dirs_exist_ok=True)
    print(f"N3: {dst / 'n3_29dof.xml'} ({len(list((dst / 'meshes').glob('*.STL')))} meshes)")


if __name__ == "__main__":
    main()
