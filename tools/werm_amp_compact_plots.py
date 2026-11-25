"""Figure export helpers for the paper AMP panels."""
from pathlib import Path
import hashlib


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _export(fig, output_dir, name, artifacts):
    """Use canonical tight bounds; return the raster alongside a vector PDF."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
    paths = {}
    for extension in ('pdf', 'png'):
        path = output_dir / f'{name}.{extension}'
        fig.savefig(path, bbox_inches=bounds, dpi=300)
        paths[extension] = dict(path=path.name, sha256=_sha256(path))
    artifacts.append(dict(name=name, canvas_inches=fig.get_size_inches().tolist(),
        exported_inches=[bounds.width, bounds.height], files=paths,
        axes=[dict(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=ax.get_title(),
                   xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())) for ax in fig.axes],
        legends=sum(ax.get_legend() is not None for ax in fig.axes)))
    plt.close(fig)
    return output_dir / f'{name}.png'


