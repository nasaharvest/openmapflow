from unittest import TestCase

import torch
from tsai.models.TransformerModel import TransformerModel


class TestTorchScriptability(TestCase):
    def test_torch_scriptability_lstm(self):
        model = torch.nn.LSTM(3, 1)
        sm = torch.jit.script(model)
        self.assertIsInstance(sm, torch.jit.ScriptModule)

    def test_torch_scriptability_torch_module(self):
        sm = torch.jit.script(torch.nn.Module())
        self.assertIsInstance(sm, torch.jit.ScriptModule)

    def test_torch_scriptability_transformer(self):
        model = TransformerModel(c_in=3, c_out=1)
        sm = torch.jit.script(model)
        self.assertIsInstance(sm, torch.jit.ScriptModule)
