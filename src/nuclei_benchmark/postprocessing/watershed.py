from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed


def threshold_foreground_map(
    foreground_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert a foreground probability map to a binary mask."""
    if foreground_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D foreground map, got shape: {foreground_map.shape}"
        )

    return np.asarray(foreground_map >= threshold, dtype=bool)


def remove_small_connected_components(
    binary_mask: np.ndarray,
    min_size: int = 16,
) -> np.ndarray:
    """Remove small connected components from a binary mask."""
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape: {binary_mask.shape}")

    labeled_mask = label(binary_mask)
    component_ids, counts = np.unique(labeled_mask, return_counts=True)

    cleaned_mask = np.zeros_like(binary_mask, dtype=bool)

    for component_id, count in zip(component_ids, counts):
        if component_id == 0:
            continue
        if count >= min_size:
            cleaned_mask[labeled_mask == component_id] = True

    return cleaned_mask


def compute_distance_map(binary_mask: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance transform of a binary mask."""
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape: {binary_mask.shape}")

    return ndi.distance_transform_edt(binary_mask)


def find_watershed_markers(
    distance_map: np.ndarray,
    binary_mask: np.ndarray,
    min_distance: int = 9,
) -> np.ndarray:
    """Find marker locations for watershed from a distance map."""
    if distance_map.shape != binary_mask.shape:
        raise ValueError(
            "distance_map and binary_mask must have the same shape. "
            f"Got {distance_map.shape} vs {binary_mask.shape}."
        )

    peak_coords = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=False,
    )

    marker_mask = np.zeros_like(binary_mask, dtype=bool)
    if len(peak_coords) > 0:
        marker_mask[tuple(peak_coords.T)] = True

    return label(marker_mask)


def apply_watershed(
    distance_map: np.ndarray,
    binary_mask: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """Apply marker-controlled watershed to obtain instance labels."""
    if distance_map.shape != binary_mask.shape:
        raise ValueError(
            "distance_map and binary_mask must have the same shape. "
            f"Got {distance_map.shape} vs {binary_mask.shape}."
        )

    if markers.shape != binary_mask.shape:
        raise ValueError(
            "markers and binary_mask must have the same shape. "
            f"Got {markers.shape} vs {binary_mask.shape}."
        )

    instance_mask = watershed(
        -distance_map,
        markers=markers,
        mask=binary_mask,
    )

    return np.asarray(instance_mask, dtype=np.uint16)


def foreground_to_instances(
    foreground_map: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 16,
    min_distance: int = 5,
) -> np.ndarray:
    """Convert a foreground map into an instance segmentation mask using watershed."""
    binary_mask = threshold_foreground_map(
        foreground_map=foreground_map,
        threshold=threshold,
    )

    binary_mask = remove_small_connected_components(
        binary_mask=binary_mask,
        min_size=min_size,
    )

    distance_map = compute_distance_map(binary_mask=binary_mask)

    markers = find_watershed_markers(
        distance_map=distance_map,
        binary_mask=binary_mask,
        min_distance=min_distance,
    )

    return apply_watershed(
        distance_map=distance_map,
        binary_mask=binary_mask,
        markers=markers,
    )