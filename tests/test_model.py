import pytest
import torch

from defect.model import SimpleCNN


@pytest.mark.parametrize("batch,size", [(1, 200), (4, 200), (3, 128)])
def test_output_shape(batch, size):
    out = SimpleCNN(num_classes=6)(torch.randn(batch, 1, size, size))
    assert out.shape == (batch, 6)


def test_num_classes_is_configurable():
    assert SimpleCNN(num_classes=3)(torch.randn(2, 1, 200, 200)).shape == (2, 3)


def test_backward_gives_gradients_to_all_parameters():
    model = SimpleCNN()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 1, 200, 200)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
