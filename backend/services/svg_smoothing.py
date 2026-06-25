import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from svgpathtools import parse_path, Path as SvgPath


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def smooth_svg_curves(
    svg_path: Path,
    enabled: bool,
    smoothing_passes: int,
    sample_step: float,
) -> dict:
    """
    Post-process SVG paths to reduce wobbly Bezier joins.
    """

    if not enabled or smoothing_passes <= 0:
        return {"smoothed_paths": 0}

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    smoothed_count = 0

    for path_element in iter_path_elements(root):
        d_attribute = path_element.get("d")
        if not d_attribute:
            continue

        try:
            smoothed_d = smooth_path_string(
                d_attribute,
                smoothing_passes=smoothing_passes,
                sample_step=sample_step,
            )
        except Exception:
            continue

        if smoothed_d:
            path_element.set("d", smoothed_d)
            smoothed_count += 1

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)

    return {"smoothed_paths": smoothed_count}


# ---------------------------------------------------------------------------
# Path-level smoothing
# ---------------------------------------------------------------------------
def smooth_path_string(d_attribute: str, smoothing_passes: int, sample_step: float) -> str:
    path = parse_path(d_attribute)
    if len(path) == 0:
        return d_attribute

    sub_paths = split_path_into_subpaths(path)

    smoothed_parts = []

    for sub in sub_paths:
        sub_d = smooth_subpath(sub, smoothing_passes, sample_step)
        if sub_d:
            smoothed_parts.append(sub_d)

    return " ".join(smoothed_parts).strip()


def split_path_into_subpaths(path: SvgPath):
    sub_paths = []
    current = SvgPath()

    for segment in path:
        if len(current) == 0:
            current.append(segment)
            continue

        last_end = current[-1].end
        if abs(last_end - segment.start) > 1e-6:
            sub_paths.append(current)
            current = SvgPath()

        current.append(segment)

    if len(current) > 0:
        sub_paths.append(current)

    return sub_paths


def smooth_subpath(sub_path: SvgPath, smoothing_passes: int, sample_step: float) -> str:
    total_length = sub_path.length(error=1e-2)
    if total_length <= 0:
        return ""

    num_samples = max(16, int(total_length / max(0.1, sample_step)))
    num_samples = min(num_samples, 1500)

    t_values = np.linspace(0, 1, num_samples)
    points_complex = np.array([sub_path.point(t) for t in t_values])
    points = np.column_stack([points_complex.real, points_complex.imag])

    is_closed = np.allclose(points[0], points[-1], atol=0.5)

    smoothed_points = apply_moving_average(points, smoothing_passes, is_closed)

    return points_to_bezier_path(smoothed_points, is_closed)


# ---------------------------------------------------------------------------
# Moving-average smoothing
# ---------------------------------------------------------------------------
def apply_moving_average(points: np.ndarray, passes: int, is_closed: bool) -> np.ndarray:
    smoothed = points.copy()

    for _ in range(passes):
        new_points = smoothed.copy()
        last_index = len(smoothed) - 1

        for i in range(len(smoothed)):
            if not is_closed and (i == 0 or i == last_index):
                continue

            prev_index = (i - 1) % len(smoothed) if is_closed else max(0, i - 1)
            next_index = (i + 1) % len(smoothed) if is_closed else min(last_index, i + 1)

            new_points[i] = (
                smoothed[prev_index] * 0.25
                + smoothed[i] * 0.5
                + smoothed[next_index] * 0.25
            )

        smoothed = new_points

    return smoothed


# ---------------------------------------------------------------------------
# Convert smoothed point list back to a Bezier path string
# ---------------------------------------------------------------------------
def points_to_bezier_path(points: np.ndarray, is_closed: bool) -> str:
    """
    Build a simple cubic Bezier path from a sequence of points.
    Uses neighbour-tangent estimation for smooth, joinable curves.
    """

    if len(points) < 2:
        return ""

    n = len(points)

    def tangent_at(index: int) -> np.ndarray:
        if is_closed:
            prev = points[(index - 1) % n]
            next_ = points[(index + 1) % n]
        else:
            prev = points[max(0, index - 1)]
            next_ = points[min(n - 1, index + 1)]
        return (next_ - prev) * 0.5

    parts = []
    parts.append("M{0:.2f} {1:.2f}".format(float(points[0][0]), float(points[0][1])))

    end_index = n if is_closed else n - 1

    for index in range(end_index):
        p0 = points[index]
        p3 = points[(index + 1) % n]

        tangent_start = tangent_at(index)
        tangent_end = tangent_at((index + 1) % n)

        control1 = p0 + tangent_start / 3.0
        control2 = p3 - tangent_end / 3.0

        parts.append(
            "C{0:.2f} {1:.2f},{2:.2f} {3:.2f},{4:.2f} {5:.2f}".format(
                float(control1[0]),
                float(control1[1]),
                float(control2[0]),
                float(control2[1]),
                float(p3[0]),
                float(p3[1]),
            )
        )

    if is_closed:
        parts.append("Z")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iter_path_elements(root: ET.Element):
    for element in root.iter():
        tag = element.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        if tag == "path":
            yield element
