"""Small CPU tests for the ablation's initial conditions and differentiability."""

import unittest

import torch

from model import make_model


class ModelTests(unittest.TestCase):
    def test_initial_outputs_and_common_parameters_match(self):
        torch.manual_seed(42)
        plain = make_model(False).eval()
        torch.manual_seed(42)
        gated = make_model(True).eval()
        x = torch.randn(3, 3, 32, 32)
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[name]), name)
        with torch.no_grad():
            a, b = plain(x), gated(x)
        self.assertEqual(a.shape, (3, 10))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_gate_can_learn(self):
        torch.manual_seed(23)
        model = make_model(True)
        logits = model(torch.randn(4, 3, 32, 32))
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(4))
        loss.backward()
        grad = model.gate.expand.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
