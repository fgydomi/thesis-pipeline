from __future__ import annotations

import numpy as np


def to_binary_foreground(mask: np.ndarray) -> np.ndarray:
    """Convert an instance or semantic mask to a binary foreground mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape: {mask.shape}")

    return np.asarray(mask > 0, dtype=bool)


def binary_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute foreground Dice score from two masks."""
    gt_fg = to_binary_foreground(gt_mask)
    pred_fg = to_binary_foreground(pred_mask)

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    gt_sum = gt_fg.sum()
    pred_sum = pred_fg.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0

    denominator = gt_sum + pred_sum
    if denominator == 0:
        return 0.0

    return float((2.0 * intersection) / denominator)


def binary_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute foreground IoU score from two masks."""
    gt_fg = to_binary_foreground(gt_mask)
    pred_fg = to_binary_foreground(pred_mask)

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def _nonzero_instance_ids(mask: np.ndarray) -> np.ndarray:
    """Return sorted non-background instance IDs."""
    instance_ids = np.unique(mask)
    return instance_ids[instance_ids > 0]


def aggregated_jaccard_index(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute Aggregated Jaccard Index (AJI) for instance segmentation.

    This implementation uses one-to-one greedy matching between GT and predicted
    instances based on pairwise IoU over overlapping pairs.

    Background must be labeled as 0. Foreground instances must be positive integers.
    """
    if gt_mask.ndim != 2 or pred_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D masks, got gt shape={gt_mask.shape}, pred shape={pred_mask.shape}"
        )

    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Shape mismatch: gt shape={gt_mask.shape}, pred shape={pred_mask.shape}"
        )

    gt_ids = _nonzero_instance_ids(gt_mask)
    pred_ids = _nonzero_instance_ids(pred_mask)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    gt_areas = {int(gt_id): int((gt_mask == gt_id).sum()) for gt_id in gt_ids}
    pred_areas = {int(pred_id): int((pred_mask == pred_id).sum()) for pred_id in pred_ids}

    candidate_pairs: list[tuple[float, int, int, int, int]] = []

    for gt_id in gt_ids:
        gt_region = gt_mask == gt_id

        overlapping_pred_ids = np.unique(pred_mask[gt_region])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        for pred_id in overlapping_pred_ids:
            pred_region = pred_mask == pred_id

            intersection = int(np.logical_and(gt_region, pred_region).sum())
            if intersection == 0:
                continue

            union = gt_areas[int(gt_id)] + pred_areas[int(pred_id)] - intersection
            iou = intersection / union

            candidate_pairs.append(
                (
                    float(iou),
                    int(gt_id),
                    int(pred_id),
                    intersection,
                    union,
                )
            )

    candidate_pairs.sort(reverse=True, key=lambda item: item[0])

    matched_gt_ids: set[int] = set()
    matched_pred_ids: set[int] = set()

    intersection_sum = 0
    union_sum = 0

    for _, gt_id, pred_id, intersection, union in candidate_pairs:
        if gt_id in matched_gt_ids or pred_id in matched_pred_ids:
            continue

        matched_gt_ids.add(gt_id)
        matched_pred_ids.add(pred_id)
        intersection_sum += intersection
        union_sum += union

    unmatched_gt_area = sum(
        gt_areas[gt_id] for gt_id in gt_areas if gt_id not in matched_gt_ids
    )
    unmatched_pred_area = sum(
        pred_areas[pred_id] for pred_id in pred_areas if pred_id not in matched_pred_ids
    )

    denominator = union_sum + unmatched_gt_area + unmatched_pred_area
    if denominator == 0:
        return 1.0

    return float(intersection_sum / denominator)