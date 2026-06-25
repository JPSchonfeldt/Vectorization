import re
from pathlib import Path
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def cleanup_svg_file(
    svg_path: Path,
    round_coordinates: bool,
    coordinate_decimals: int,
    remove_tiny_paths: bool,
    min_path_size: int,
    merge_same_color_paths: bool,
) -> dict:
    """
    Post-process a VTracer-generated SVG to make it smaller and cleaner.

    Applies up to four optional cleanup passes:
      1. Round path coordinates to fewer decimals
      2. Remove very small (speckle) paths
      3. Merge adjacent paths of identical fill colour
      4. Strip XML metadata and comments (always on)
    """

    original_size_bytes = svg_path.stat().st_size

    # Avoid namespace mangling on output by registering the default SVG ns.
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Always strip comments and unused metadata
    strip_metadata_and_comments(root)

    if round_coordinates:
        round_path_coordinates(root, decimals=coordinate_decimals)

    removed_path_count = 0
    if remove_tiny_paths:
        removed_path_count = remove_small_paths(root, min_size=min_path_size)

    merged_path_count = 0
    if merge_same_color_paths:
        merged_path_count = merge_paths_by_fill_colour(root)

    # Save back
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)

    new_size_bytes = svg_path.stat().st_size
    saved_bytes = original_size_bytes - new_size_bytes
    saved_percent = (saved_bytes / original_size_bytes * 100.0) if original_size_bytes > 0 else 0.0

    return {
        "original_size_bytes": original_size_bytes,
        "new_size_bytes": new_size_bytes,
        "saved_bytes": saved_bytes,
        "saved_percent": round(saved_percent, 1),
        "removed_path_count": removed_path_count,
        "merged_path_count": merged_path_count,
    }


# ---------------------------------------------------------------------------
# Step 1 — Round path coordinates
# ---------------------------------------------------------------------------
def round_path_coordinates(root: ET.Element, decimals: int) -> None:
    """
    Round every floating-point number in every <path d="..."> attribute.
    """

    pattern = re.compile(r"-?\d+\.\d+")
    format_string = "{:." + str(decimals) + "f}"

    def replace_number(match: re.Match) -> str:
        rounded = float(match.group(0))
        # Format and strip trailing zeros/decimal point for compactness
        text = format_string.format(rounded)
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    for path_element in iter_elements_by_local_name(root, "path"):
        d_attribute = path_element.get("d")
        if not d_attribute:
            continue
        path_element.set("d", pattern.sub(replace_number, d_attribute))


# ---------------------------------------------------------------------------
# Step 2 — Remove tiny paths
# ---------------------------------------------------------------------------
def remove_small_paths(root: ET.Element, min_size: int) -> int:
    """
    Drop paths whose approximate bounding-box area is below min_size^2.

    Approximation uses the numeric extents of the path's 'd' attribute.
    Fast, conservative, and good enough for VTracer output.
    """

    if min_size <= 0:
        return 0

    threshold = min_size * min_size
    number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
    removed_count = 0

    # Search recursively for path elements and their parents.
    parents = {child: parent for parent in root.iter() for child in parent}

    paths_to_remove = []

    for path_element in iter_elements_by_local_name(root, "path"):
        d_attribute = path_element.get("d", "")
        if not d_attribute:
            continue

        numbers = [float(value) for value in number_pattern.findall(d_attribute)]
        if len(numbers) < 4:
            continue

        # Treat alternating values as (x, y) pairs to approximate extents
        xs = numbers[0::2]
        ys = numbers[1::2]

        bounding_width = max(xs) - min(xs)
        bounding_height = max(ys) - min(ys)
        bounding_area = bounding_width * bounding_height

        if bounding_area < threshold:
            paths_to_remove.append(path_element)

    for path_element in paths_to_remove:
        parent_element = parents.get(path_element)
        if parent_element is not None:
            parent_element.remove(path_element)
            removed_count += 1

    return removed_count


# ---------------------------------------------------------------------------
# Step 3 — Merge same-colour paths
# ---------------------------------------------------------------------------
def merge_paths_by_fill_colour(root: ET.Element) -> int:
    """
    Combine adjacent paths that share the exact same fill colour.

    This reduces SVG path count significantly for output produced by
    VTracer's stacked colour layers.
    """

    parents = {child: parent for parent in root.iter() for child in parent}

    paths_by_parent = {}

    for path_element in iter_elements_by_local_name(root, "path"):
        parent_element = parents.get(path_element)
        if parent_element is None:
            continue
        paths_by_parent.setdefault(parent_element, []).append(path_element)

    merged_count = 0

    for parent_element, path_list in paths_by_parent.items():
        # Walk in order; merge runs of consecutive same-colour paths.
        current_run = []

        def flush_run():
            nonlocal merged_count
            if len(current_run) <= 1:
                return
            # Merge into the first path of the run.
            base_path = current_run[0]
            combined_d = base_path.get("d", "")
            for follower in current_run[1:]:
                follower_d = follower.get("d", "")
                if combined_d and follower_d:
                    combined_d = combined_d + " " + follower_d
                elif follower_d:
                    combined_d = follower_d
                parent_element.remove(follower)
                merged_count += 1
            base_path.set("d", combined_d)

        last_fill = None

        for path_element in path_list:
            fill = path_element.get("fill", "")
            if fill == last_fill and fill:
                current_run.append(path_element)
            else:
                flush_run()
                current_run = [path_element]
                last_fill = fill

        flush_run()

    return merged_count


# ---------------------------------------------------------------------------
# Step 4 — Strip metadata and comments
# ---------------------------------------------------------------------------
def strip_metadata_and_comments(root: ET.Element) -> None:
    """
    Remove <metadata>, <desc>, <title>, comments, and DTD-like cruft.
    """

    tags_to_remove = {"metadata", "desc", "title"}

    parents = {child: parent for parent in root.iter() for child in parent}

    elements_to_remove = []

    for element in root.iter():
        local_name = strip_namespace(element.tag)
        if local_name in tags_to_remove:
            elements_to_remove.append(element)

    for element in elements_to_remove:
        parent_element = parents.get(element)
        if parent_element is not None:
            parent_element.remove(element)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iter_elements_by_local_name(root: ET.Element, local_name: str):
    """
    Yield elements whose tag's local name matches local_name,
    ignoring XML namespaces (which VTracer always emits).
    """

    for element in root.iter():
        if strip_namespace(element.tag) == local_name:
            yield element


def strip_namespace(tag: str) -> str:
    """
    Convert '{http://www.w3.org/2000/svg}path' into 'path'.
    """

    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag