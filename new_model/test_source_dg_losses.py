"""Focused CPU tests for the source-only EMA/contrastive additions.

Run with a PyTorch environment from the project root:

    python -m unittest new_model.test_source_dg_losses
"""

from __future__ import annotations

import unittest

import torch

try:
    from .source_dg_losses import (
        ConfidenceGatedTeacherLoss,
        CrossBatchSupervisedContrastiveLoss,
    )
except ImportError:  # Direct execution from ``new_model``.
    from source_dg_losses import (  # type: ignore
        ConfidenceGatedTeacherLoss,
        CrossBatchSupervisedContrastiveLoss,
    )


class CrossBatchSupervisedContrastiveLossTests(unittest.TestCase):
    def test_paired_teacher_keys_train_anchors_but_not_keys(self) -> None:
        objective = CrossBatchSupervisedContrastiveLoss(
            feature_dim=2, queue_size=3, temperature=0.2
        )
        anchors = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        keys = torch.tensor(
            [[0.6, 0.8], [0.8, 0.6]], requires_grad=True
        )

        loss = objective(anchors, torch.tensor([0, 1]), key_features=keys)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(anchors.grad)
        self.assertGreater(float(anchors.grad.abs().sum()), 0.0)
        self.assertIsNone(keys.grad)

    def test_fifo_wrap_and_state_round_trip(self) -> None:
        objective = CrossBatchSupervisedContrastiveLoss(
            feature_dim=2, queue_size=3, temperature=0.2
        )
        objective.enqueue(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])
        )
        objective.enqueue(
            torch.tensor([[-1.0, 0.0], [0.0, -1.0]]), torch.tensor([2, 3])
        )

        self.assertEqual(objective.stored_features, 3)
        self.assertEqual(int(objective.queue_pointer.item()), 1)
        torch.testing.assert_close(
            objective.queue_labels, torch.tensor([3, 1, 2])
        )

        restored = CrossBatchSupervisedContrastiveLoss(
            feature_dim=2, queue_size=3, temperature=0.2
        )
        restored.load_state_dict(objective.state_dict())
        self.assertEqual(restored.stored_features, 3)
        torch.testing.assert_close(restored.queue_features, objective.queue_features)
        torch.testing.assert_close(restored.queue_labels, objective.queue_labels)
        torch.testing.assert_close(restored.queue_count, objective.queue_count)
        torch.testing.assert_close(restored.queue_pointer, objective.queue_pointer)


class ConfidenceGatedTeacherLossTests(unittest.TestCase):
    def test_all_rejected_gate_returns_differentiable_zero(self) -> None:
        objective = ConfidenceGatedTeacherLoss(
            temperature=1.5, confidence_threshold=1.0, require_label_agreement=True
        )
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.tensor([[8.0, -2.0, -2.0], [-2.0, 8.0, -2.0]])

        loss, coverage = objective(student, teacher, torch.tensor([0, 1]))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(coverage), 0.0)
        loss.backward()

        self.assertIsNotNone(student.grad)
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_confident_label_agreeing_teacher_supervises_student(self) -> None:
        objective = ConfidenceGatedTeacherLoss(
            temperature=1.5, confidence_threshold=0.7, require_label_agreement=True
        )
        student = torch.zeros(2, 3, requires_grad=True)
        teacher = torch.tensor([[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]])

        loss, coverage = objective(student, teacher, torch.tensor([0, 1]))
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(coverage), 1.0)
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
