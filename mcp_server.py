from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from LineExtraction.extractor import Extractor
from LineExtraction.split import Split
from helper import get_folder
from util import cache_folder


ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(cache_folder)
WORK_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("digital-image")


def _resolve_existing_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"File does not exist: {candidate}")
    return candidate


def _rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@mcp.tool()
def extract_curve_points(
    image_path: str,
    x_start: float = 0,
    x_end: float = 100,
    y_start: float = 0,
    y_end: float = 100,
    min_point_distance: float = 5,
    outlier_distance: float = 20,
) -> dict[str, Any]:
    """Extract data points from a single curve image and create CSV plus interpolation plot."""
    source = _resolve_existing_path(image_path)
    extractor = Extractor(
        str(source),
        x_start,
        x_end,
        y_start,
        y_end,
        min_point_distance,
        outlier_distance,
    )
    plot_path = extractor.interpolate()
    return {
        "source": _rel(source),
        "csv_path": _rel(extractor.csv_path),
        "plot_path": _rel(plot_path),
        "point_count": len(extractor.cordinate),
        "sample_points": extractor.cordinate[:10],
    }


@mcp.tool()
def split_horizontal_images(
    image_paths: list[str],
    threshold: int = 60,
    distance_ratio: float = 0.015,
) -> dict[str, Any]:
    """Horizontally split all input images and return visualization plus generated image paths."""
    if not image_paths:
        raise ValueError("image_paths must contain at least one image path")

    results = []
    for image_path in image_paths:
        source = _resolve_existing_path(image_path)
        splitter = Split(str(source), thresh=threshold, distance=distance_ratio)
        results.append(
            {
                "source": _rel(source),
                "visualize_path": _rel(splitter.visualize),
                "output_dir": _rel(splitter.output),
                "split_images": [_rel(path) for path in get_folder(splitter.output)],
            }
        )

    return {"count": len(results), "results": results}


if __name__ == "__main__":
    mcp.run()
