from pathlib import Path
import subprocess
import tempfile
from IPython.display import Image, display


def show_panels(paths):
    for name in paths:
        path = Path(name).resolve()
        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "preview"
            subprocess.run([
                "pdftoppm", "-png", "-singlefile", "-scale-to", "1600",
                str(path), str(preview),
            ], check=True, capture_output=True)
            display(Image(data=preview.with_suffix(".png").read_bytes()))
