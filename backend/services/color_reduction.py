import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans


def reduce_image_colors_image(image: Image.Image, target_colors: int) -> Image.Image:
    """
    Reduce image colours while trying to preserve visually different hue groups.

    Standard quantization may pick several similar shades of the most common colour.
    This function uses HSV analysis to spread the palette across different hue ranges
    where possible, so the result feels closer to the original visual variety.
    """

    image = image.convert("RGB")

    # Use a smaller copy for palette analysis to keep this fast.
    analysis_image = image.copy()
    analysis_image.thumbnail((700, 700))

    analysis_pixels = np.array(analysis_image).reshape(-1, 3)
    analysis_pixels_float = analysis_pixels.astype(np.float32) / 255.0

    hsv_pixels = rgb_to_hsv_numpy(analysis_pixels_float)

    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]

    palette = []

    # Reserve a slot for near-white if the image has any meaningful white area.
    white_mask = (saturation < 0.12) & (value > 0.88)

    if np.mean(white_mask) > 0.05:
        palette.append([255, 255, 255])

    # Reserve a slot for near-black if the image has any meaningful dark area.
    dark_mask = value < 0.12

    if np.mean(dark_mask) > 0.02 and len(palette) < target_colors:
        dark_pixels = analysis_pixels[dark_mask]

        if len(dark_pixels) > 0:
            palette.append(np.median(dark_pixels, axis=0).astype(int).tolist())

    # Chromatic (colourful) pixels only.
    colour_mask = (saturation >= 0.16) & (value >= 0.12)

    colour_pixels = analysis_pixels[colour_mask]
    colour_hues = hue[colour_mask]
    colour_saturation = saturation[colour_mask]
    colour_value = value[colour_mask]

    if len(colour_pixels) > 0:
        hue_bin_count = max(24, target_colors * 4)

        hue_bins = np.clip(
            np.floor(colour_hues * hue_bin_count).astype(int),
            0,
            hue_bin_count - 1
        )

        candidate_colours = []

        for bin_index in range(hue_bin_count):
            bin_mask = hue_bins == bin_index

            if not np.any(bin_mask):
                continue

            bin_pixels = colour_pixels[bin_mask]
            bin_saturation = colour_saturation[bin_mask]
            bin_value = colour_value[bin_mask]

            score = (
                len(bin_pixels)
                * (0.5 + float(np.mean(bin_saturation)))
                * (0.5 + float(np.mean(bin_value)))
            )

            candidate_colours.append({
                "bin": bin_index,
                "score": score,
                "colour": np.median(bin_pixels, axis=0).astype(int).tolist(),
            })

        candidate_colours.sort(key=lambda item: item["score"], reverse=True)

        selected_bins = []

        # First pass: prefer hue bins that are not adjacent to already chosen ones.
        for candidate in candidate_colours:
            if len(palette) >= target_colors:
                break

            candidate_bin = candidate["bin"]

            too_close = any(
                min(
                    abs(candidate_bin - selected_bin),
                    hue_bin_count - abs(candidate_bin - selected_bin)
                ) <= 1
                for selected_bin in selected_bins
            )

            if too_close:
                continue

            palette.append(candidate["colour"])
            selected_bins.append(candidate_bin)

        # Second pass: fill remaining slots if we still do not have enough.
        for candidate in candidate_colours:
            if len(palette) >= target_colors:
                break

            if candidate["colour"] not in palette:
                palette.append(candidate["colour"])

    # Fallback: ask Pillow to fill any remaining slots using median cut.
    if len(palette) < target_colors:
        fallback_quantized = image.quantize(
            colors=target_colors,
            method=Image.Quantize.MEDIANCUT,
        )

        fallback_palette = fallback_quantized.getpalette()

        for i in range(0, min(len(fallback_palette), target_colors * 3), 3):
            colour = [
                fallback_palette[i],
                fallback_palette[i + 1],
                fallback_palette[i + 2],
            ]

            if len(palette) >= target_colors:
                break

            if colour not in palette:
                palette.append(colour)

    palette = palette[:target_colors]

    if not palette:
        raise Exception("Could not create a colour palette from the image.")

    return remap_image_to_palette(image, palette), palette


def rgb_to_hsv_numpy(rgb_array: np.ndarray) -> np.ndarray:
    """
    Vectorised RGB to HSV conversion.
    Input shape:  N x 3 with values in 0..1
    Output shape: N x 3 with values in 0..1
    """

    r = rgb_array[:, 0]
    g = rgb_array[:, 1]
    b = rgb_array[:, 2]

    max_colour = np.max(rgb_array, axis=1)
    min_colour = np.min(rgb_array, axis=1)
    delta = max_colour - min_colour

    hue = np.zeros_like(max_colour)

    red_mask = (max_colour == r) & (delta != 0)
    green_mask = (max_colour == g) & (delta != 0)
    blue_mask = (max_colour == b) & (delta != 0)

    hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = ((b[green_mask] - r[green_mask]) / delta[green_mask]) + 2
    hue[blue_mask] = ((r[blue_mask] - g[blue_mask]) / delta[blue_mask]) + 4

    hue /= 6.0

    saturation = np.zeros_like(max_colour)
    non_zero = max_colour != 0
    saturation[non_zero] = delta[non_zero] / max_colour[non_zero]

    return np.stack([hue, saturation, max_colour], axis=1)


def remap_image_to_palette(image: Image.Image, palette: list) -> Image.Image:
    """
    Replace each pixel with the nearest colour in the palette.
    Uses chunking to keep memory use reasonable on large images.
    """

    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)

    height, width, _ = image_array.shape

    flat_pixels = image_array.reshape(-1, 3).astype(np.int32)
    palette_array = np.array(palette, dtype=np.int32)

    output_pixels = np.empty_like(flat_pixels, dtype=np.uint8)

    chunk_size = 200_000

    for start in range(0, len(flat_pixels), chunk_size):
        chunk = flat_pixels[start:start + chunk_size]

        dists = np.sum(
            (chunk[:, None, :] - palette_array[None, :, :]) ** 2,
            axis=2,
        )

        nearest = np.argmin(dists, axis=1)

        output_pixels[start:start + chunk_size] = palette_array[nearest].astype(np.uint8)

    return Image.fromarray(output_pixels.reshape(height, width, 3), mode="RGB")

# ---------------------------------------------------------------------------
# LAB K-means colour reduction (Phase 1A)
# ---------------------------------------------------------------------------
def reduce_image_colors_lab_kmeans(
    image: Image.Image,
    target_colors: int,
) -> Image.Image:
    """
    Reduce image colours using K-means clustering in LAB colour space.

    Why LAB?
    LAB is designed around human perception, so two LAB-close colours
    actually look similar to us. RGB distance does not reflect that â€”
    it can group visually different colours together and split visually
    similar ones.

    Why K-means?
    K-means picks N representative colours that best summarise the image,
    which is much more "natural" than median cut or hue spreading.

    Result: cleaner, more believable colour groupings, especially for
    logos, illustrations and flat artwork.
    """

    image = image.convert("RGB")

    # Use a smaller copy for clustering to keep speed reasonable.
    analysis_image = image.copy()
    analysis_image.thumbnail((600, 600))

    analysis_array = np.array(analysis_image)
    height, width, _ = analysis_array.shape

    # Convert to LAB
    rgb_normalised = analysis_array.astype(np.float32) / 255.0
    lab_pixels = skcolor.rgb2lab(rgb_normalised).reshape(-1, 3)

    # Reserve obvious near-white and near-black colours so they survive
    # the K-means step. This helps keep clean backgrounds and outlines.
    rgb_flat = analysis_array.reshape(-1, 3)

    near_white_mask = np.all(rgb_flat > 235, axis=1)
    near_black_mask = np.all(rgb_flat < 25, axis=1)

    reserved_colours = []

    if np.mean(near_white_mask) > 0.04:
        reserved_colours.append([255, 255, 255])

    if np.mean(near_black_mask) > 0.02 and len(reserved_colours) < target_colors:
        reserved_colours.append([0, 0, 0])

    # Calculate how many K-means clusters to fit, accounting for reserved slots.
    clusters_to_find = max(1, target_colors - len(reserved_colours))

    # Drop reserved-coloured pixels from clustering input so they do not
    # dominate the K-means centroids.
    keep_mask = ~(near_white_mask | near_black_mask)
    clustering_pixels = lab_pixels[keep_mask]

    if len(clustering_pixels) < clusters_to_find:
        clustering_pixels = lab_pixels  # fallback

    # Subsample to keep K-means quick on big images.
    max_clustering_samples = 80_000
    if len(clustering_pixels) > max_clustering_samples:
        sample_indices = np.random.default_rng(42).choice(
            len(clustering_pixels),
            size=max_clustering_samples,
            replace=False,
        )
        clustering_pixels = clustering_pixels[sample_indices]

    kmeans = KMeans(
        n_clusters=clusters_to_find,
        n_init=4,
        random_state=42,
    )
    kmeans.fit(clustering_pixels)

    # Convert cluster centres (LAB) back to RGB
    cluster_centres_lab = kmeans.cluster_centers_.reshape(1, -1, 3)
    cluster_centres_rgb = skcolor.lab2rgb(cluster_centres_lab).reshape(-1, 3)
    cluster_centres_rgb = np.clip(cluster_centres_rgb * 255.0, 0, 255).astype(int).tolist()

    palette = list(reserved_colours) + cluster_centres_rgb
    palette = palette[:target_colors]

    if not palette:
        raise Exception("Could not create a LAB K-means palette from the image.")

    return remap_image_to_palette(image, palette), palette

