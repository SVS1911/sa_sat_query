"""
utils/evaluation.py
----------------------
"Perfect evaluation for each and every model" — a real, from-scratch metrics
harness (no sklearn dependency) for every model in the pipeline. Two levels:

  1. Pixel-level land-cover evaluation (evaluate_landcover): confusion matrix,
     per-class precision/recall/F1, per-class IoU, mean IoU, overall accuracy.
     Use this once you have ground-truth label rasters (from your future
     datasets, or from a public benchmark like ben-ge's ESA WorldCover maps).

  2. Whole-pipeline evaluation (evaluate_dataset): runs SatQueryController
     over a labeled manifest (image(s) + query + expected answer/class) and
     reports per-task accuracy, so every specialist model (VQA, captioning,
     grounding, change, fusion) gets its own accuracy number, not just the
     land-cover backend.

Nothing here requires torch/scikit — pure numpy — so it runs anywhere the
base app runs, with zero extra dependencies.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# --------------------------------------------------------------------------- #
# 1. Pixel-level land-cover evaluation
# --------------------------------------------------------------------------- #

@dataclass
class ClassMetrics:
    class_name: str
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    support: int  # number of ground-truth pixels of this class


@dataclass
class LandcoverEvalReport:
    class_metrics: List[ClassMetrics]
    overall_accuracy: float
    mean_iou: float
    mean_f1: float
    confusion_matrix: List[List[int]]
    class_names: List[str]

    def to_dict(self) -> dict:
        return {
            "overall_accuracy": self.overall_accuracy,
            "mean_iou": self.mean_iou,
            "mean_f1": self.mean_f1,
            "class_names": self.class_names,
            "confusion_matrix": self.confusion_matrix,
            "per_class": [
                {
                    "class": m.class_name, "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4), "f1": round(m.f1, 4),
                    "iou": round(m.iou, 4), "support": m.support,
                    "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                }
                for m in self.class_metrics
            ],
        }

    def pretty_print(self) -> str:
        lines = [f"Overall accuracy: {self.overall_accuracy*100:.2f}%   "
                 f"Mean IoU: {self.mean_iou*100:.2f}%   Mean F1: {self.mean_f1*100:.2f}%", ""]
        lines.append(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'iou':>10}{'support':>10}")
        for m in self.class_metrics:
            lines.append(f"{m.class_name:<12}{m.precision*100:>9.1f}%{m.recall*100:>9.1f}%"
                         f"{m.f1*100:>9.1f}%{m.iou*100:>9.1f}%{m.support:>10}")
        return "\n".join(lines)


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    """Standard row=ground-truth, col=predicted confusion matrix."""
    pred_flat = pred.reshape(-1).astype(np.int64)
    gt_flat = gt.reshape(-1).astype(np.int64)
    valid = (gt_flat >= 0) & (gt_flat < n_classes) & (pred_flat >= 0) & (pred_flat < n_classes)
    pred_flat, gt_flat = pred_flat[valid], gt_flat[valid]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (gt_flat, pred_flat), 1)
    return cm


def evaluate_landcover(pred_class_map: np.ndarray, gt_class_map: np.ndarray,
                        class_names: List[str]) -> LandcoverEvalReport:
    """
    pred_class_map / gt_class_map: H x W int arrays with the same shape,
    values indexing into class_names (e.g. utils.spectral_indices.LAND_COVER_CLASSES).
    Resize/align both to the same grid before calling this if they differ.
    """
    if pred_class_map.shape != gt_class_map.shape:
        raise ValueError(f"Shape mismatch: pred {pred_class_map.shape} vs gt {gt_class_map.shape}. "
                          f"Resample ground truth to the prediction grid first.")

    n = len(class_names)
    cm = confusion_matrix(pred_class_map, gt_class_map, n)

    class_metrics = []
    ious, f1s = [], []
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        class_metrics.append(ClassMetrics(cls, tp, fp, fn, tn, precision, recall, f1, iou, support))
        if support > 0:  # only average over classes actually present in ground truth
            ious.append(iou)
            f1s.append(f1)

    overall_accuracy = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0

    return LandcoverEvalReport(
        class_metrics=class_metrics, overall_accuracy=overall_accuracy,
        mean_iou=mean_iou, mean_f1=mean_f1, confusion_matrix=cm.tolist(), class_names=class_names,
    )


# --------------------------------------------------------------------------- #
# 2. Whole-pipeline evaluation (per specialist model)
# --------------------------------------------------------------------------- #

@dataclass
class TaskEvalSummary:
    task: str
    n_samples: int
    n_correct: int
    accuracy: float
    failures: List[Dict[str, Any]] = field(default_factory=list)


def evaluate_dataset(controller, manifest: List[Dict[str, Any]],
                      match_fn=None) -> Dict[str, TaskEvalSummary]:
    """
    manifest: list of dicts, each with at minimum:
        {"images": [path, ...], "query": str, "expected_task": str (optional),
         "expected_answer_contains": str (optional, substring match),
         "expected_target_class": str (optional),
         "declared_pair_type": str (optional), "band_preset": str (optional)}

    match_fn(result, sample) -> bool, optional custom correctness check; default
    checks `expected_answer_contains` as a case-insensitive substring of the
    returned answer (cheap but real: exact-match/substring scoring is standard
    for template-generated NLG answers; swap in your own scorer — e.g. exact
    label match for classification-style ground truth — via match_fn).

    Returns: {task_name: TaskEvalSummary}, so every specialist model's
    accuracy is reported separately.
    """
    def default_match(result, sample):
        expected = sample.get("expected_answer_contains")
        if expected is None:
            return None  # can't score this sample
        return expected.lower() in (result.answer or "").lower()

    match_fn = match_fn or default_match
    per_task: Dict[str, List[bool]] = {}
    failures: Dict[str, List[Dict[str, Any]]] = {}

    for sample in manifest:
        result = controller.run(
            sample["images"], sample["query"],
            declared_pair_type=sample.get("declared_pair_type"),
            band_preset=sample.get("band_preset"),
        )
        task = result.task
        correct = match_fn(result, sample)
        if correct is None:
            continue
        per_task.setdefault(task, []).append(bool(correct))
        if not correct:
            failures.setdefault(task, []).append({
                "images": sample["images"], "query": sample["query"],
                "expected": sample.get("expected_answer_contains"),
                "got": result.answer,
            })

    summaries = {}
    for task, results in per_task.items():
        n = len(results)
        n_correct = sum(results)
        summaries[task] = TaskEvalSummary(
            task=task, n_samples=n, n_correct=n_correct,
            accuracy=n_correct / n if n else 0.0,
            failures=failures.get(task, []),
        )
    return summaries


def print_dataset_eval(summaries: Dict[str, TaskEvalSummary]) -> str:
    lines = ["Per-model evaluation:", ""]
    lines.append(f"{'task':<20}{'samples':>10}{'correct':>10}{'accuracy':>12}")
    for task, s in summaries.items():
        lines.append(f"{task:<20}{s.n_samples:>10}{s.n_correct:>10}{s.accuracy*100:>11.1f}%")
    return "\n".join(lines)
