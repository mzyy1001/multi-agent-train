"""Tests for communication channels."""

import torch
from src.comm.ssr import SSRChannel
from src.comm.discrete import DiscreteChannel
from src.comm.continuous import ContinuousChannel
from src.comm.none import NoChannel


def test_ssr_channel():
    ch = SSRChannel(input_dim=128, dim=8)
    z = torch.randn(4, 128)
    m = ch(z)
    assert m.shape == (4, 8)
    assert ch.message_dim() == 8
    # Test gradient flow
    m.sum().backward()
    assert ch.encoder[0].weight.grad is not None


def test_discrete_channel():
    ch = DiscreteChannel(input_dim=128, num_symbols=8, tau=1.0)
    z = torch.randn(4, 128)

    ch.train()
    m_train = ch(z)
    assert m_train.shape == (4, 8)
    # Soft one-hot during training
    assert not torch.all(m_train == m_train.round())

    ch.eval()
    m_eval = ch(z)
    assert m_eval.shape == (4, 8)
    # Hard one-hot during eval
    assert torch.all(m_eval.sum(dim=-1).isclose(torch.ones(4)))


def test_continuous_channel():
    ch = ContinuousChannel(input_dim=128, dim=8)
    z = torch.randn(4, 128)
    m = ch(z)
    assert m.shape == (4, 8)
    assert ch.message_dim() == 8


def test_none_channel():
    ch = NoChannel(dim=8)
    z = torch.randn(4, 128)
    m = ch(z)
    assert m.shape == (4, 8)
    assert torch.all(m == 0)
    assert ch.message_dim() == 8


def test_gradient_flow_through_ssr():
    """Verify gradients flow from downstream loss through SSR to upstream params."""
    proj = torch.nn.Linear(64, 128)
    ch = SSRChannel(128, 8)
    head = torch.nn.Linear(8, 1)

    x = torch.randn(2, 64)
    z = proj(x)
    m = ch(z)
    loss = head(m).sum()
    loss.backward()

    assert proj.weight.grad is not None
    assert ch.encoder[0].weight.grad is not None
    assert head.weight.grad is not None


if __name__ == "__main__":
    test_ssr_channel()
    test_discrete_channel()
    test_continuous_channel()
    test_none_channel()
    test_gradient_flow_through_ssr()
    print("All comm tests passed!")
