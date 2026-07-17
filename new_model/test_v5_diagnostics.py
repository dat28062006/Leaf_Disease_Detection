"""Regression tests for compact, source-only V5 diagnostic summaries."""

from __future__ import annotations

import json
import unittest

import torch

try:
    from .curriculum_train_v5 import _disease_class_diagnostics
    from .source_dg_losses import ClassificationMetricsAccumulator
except ImportError:  # Direct execution from ``new_model``.
    from curriculum_train_v5 import _disease_class_diagnostics  # type: ignore
    from source_dg_losses import ClassificationMetricsAccumulator  # type: ignore


class DiseaseClassDiagnosticsTests(unittest.TestCase):
    def test_reports_weak_class_and_dominant_confusion_as_json(self) -> None:
        metrics = ClassificationMetricsAccumulator(3)
        # True class 1 is repeatedly confused with class 2.
        metrics.update(
            torch.tensor([0, 2, 2, 2, 1]),
            torch.tensor([0, 1, 1, 1, 1]),
        )

        report = _disease_class_diagnostics(
            metrics,
            class_names=["PlantA___Healthy", "PlantA___Spot", "PlantB___Rust"],
            class_to_plant=[0, 0, 1],
            train_class_counts=[100, 20, 80],
            top_k=3,
            min_support=1,
        )

        self.assertEqual(report["weakest_classes"][0]["class_index"], 1)
        top = report["top_confusions"][0]
        self.assertEqual(top["true_class_index"], 1)
        self.assertEqual(top["predicted_class_index"], 2)
        self.assertFalse(top["same_plant"])
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
