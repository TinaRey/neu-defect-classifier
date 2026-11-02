import torch

from defect.evaluate import confusion_matrix, evaluate_model, format_report


def test_confusion_matrix_counts():
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = torch.tensor([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(preds, targets, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


class FixedModel(torch.nn.Module):
    """Ignores the input and predicts class 0 for the first image of the batch, class 1 for the rest."""
    def forward(self, x):
        logits = torch.zeros(len(x), 2)
        logits[0, 0] = 1.0
        logits[1:, 1] = 1.0
        return logits


def test_metrics_hand_computed():
    # targets [0,0,1,1], model predicts [0,1,1,1] -> class 0: P=1 R=0.5; class 1: P=2/3 R=1
    loader = [(torch.zeros(4, 1, 4, 4), torch.tensor([0, 0, 1, 1]))]
    report = evaluate_model(FixedModel(), loader, ["a", "b"], "cpu")
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"]["precision"] == 1.0 and report["per_class"]["a"]["recall"] == 0.5
    assert abs(report["per_class"]["b"]["precision"] - 2 / 3) < 1e-6 and report["per_class"]["b"]["recall"] == 1.0
    assert "confusion matrix" in format_report(report)
