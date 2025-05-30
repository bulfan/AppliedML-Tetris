import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.model import QNetwork
class TestQNetwork(unittest.TestCase):
    """Test cases for the QNetwork class"""
    def setUp(self):
        """Set up test fixtures"""
        self.state_size = 220        
        self.action_size = 4
        self.hidden_layers = [512, 256, 128]
        self.dropout_rate = 0.2
        self.network = QNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        )
    def test_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.network.state_size, self.state_size)
        self.assertEqual(self.network.action_size, self.action_size)
        self.assertIsInstance(self.network.network, nn.Sequential)
    def test_network_architecture(self):
        """Test network architecture is correct"""
        linear_layers = []
        relu_layers = []
        dropout_layers = []
        for module in self.network.network:
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
            elif isinstance(module, nn.ReLU):
                relu_layers.append(module)
            elif isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        expected_linear_layers = len(self.hidden_layers) + 1
        self.assertEqual(len(linear_layers), expected_linear_layers)
        self.assertEqual(len(relu_layers), len(self.hidden_layers))
        self.assertEqual(len(dropout_layers), len(self.hidden_layers))
        first_layer = linear_layers[0]
        last_layer = linear_layers[-1]
        self.assertEqual(first_layer.in_features, self.state_size)
        self.assertEqual(last_layer.out_features, self.action_size)
        for i, layer in enumerate(linear_layers[:-1]):
            expected_out_features = self.hidden_layers[i]
            self.assertEqual(layer.out_features, expected_out_features)
            if i > 0:
                expected_in_features = self.hidden_layers[i-1]
                self.assertEqual(layer.in_features, expected_in_features)
    def test_forward_pass(self):
        """Test forward pass through the network"""
        batch_size = 32
        input_tensor = torch.randn(batch_size, self.state_size)
        output = self.network(input_tensor)
        self.assertEqual(output.shape, (batch_size, self.action_size))
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample"""
        input_tensor = torch.randn(1, self.state_size)
        output = self.network(input_tensor)
        self.assertEqual(output.shape, (1, self.action_size))
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes"""
        batch_sizes = [1, 8, 16, 32, 64]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.state_size)
            output = self.network(input_tensor)
            self.assertEqual(output.shape, (batch_size, self.action_size))
            self.assertTrue(torch.all(torch.isfinite(output)))
    def test_weight_initialization(self):
        """Test that weights are properly initialized"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                self.assertFalse(torch.all(module.weight == 0))
                self.assertTrue(torch.all(module.bias == 0.01))
                weight_std = torch.std(module.weight)
                self.assertGreater(weight_std, 0.01)
                self.assertLess(weight_std, 1.0)
    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        input_tensor = torch.randn(16, self.state_size, requires_grad=True)
        output = self.network(input_tensor)
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.any(input_tensor.grad != 0))
        for param in self.network.parameters():
            self.assertIsNotNone(param.grad)
            if param.numel() > 1:                self.assertTrue(torch.any(param.grad != 0))
    def test_training_vs_eval_mode(self):
        """Test behavior difference between training and evaluation modes"""
        input_tensor = torch.randn(16, self.state_size)
        self.network.train()
        output_train = self.network(input_tensor)
        self.network.eval()
        output_eval = self.network(input_tensor)
        self.assertEqual(output_train.shape, output_eval.shape)
        self.assertTrue(torch.all(torch.isfinite(output_train)))
        self.assertTrue(torch.all(torch.isfinite(output_eval)))
    def test_dropout_effect(self):
        """Test that dropout has an effect during training"""
        input_tensor = torch.randn(32, self.state_size)
        self.network.train()
        outputs = []
        for _ in range(5):
            output = self.network(input_tensor)
            outputs.append(output.detach().clone())
        all_same = True
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
                all_same = False
                break
        self.assertFalse(all_same)
    def test_custom_architecture(self):
        """Test network with custom architecture"""
        custom_hidden = [256, 128, 64]
        custom_dropout = 0.3
        custom_network = QNetwork(
            state_size=100,
            action_size=6,
            hidden_layers=custom_hidden,
            dropout_rate=custom_dropout
        )
        input_tensor = torch.randn(8, 100)
        output = custom_network(input_tensor)
        self.assertEqual(output.shape, (8, 6))
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_no_hidden_layers(self):
        """Test network with no hidden layers (direct input to output)"""
        simple_network = QNetwork(
            state_size=10,
            action_size=4,
            hidden_layers=[],
            dropout_rate=0.0
        )
        input_tensor = torch.randn(5, 10)
        output = simple_network(input_tensor)
        self.assertEqual(output.shape, (5, 4))
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_large_network(self):
        """Test network with large architecture"""
        large_network = QNetwork(
            state_size=1000,
            action_size=10,
            hidden_layers=[1024, 512, 256, 128],
            dropout_rate=0.5
        )
        input_tensor = torch.randn(4, 1000)
        output = large_network(input_tensor)
        self.assertEqual(output.shape, (4, 10))
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_parameter_count(self):
        """Test that parameter count is reasonable"""
        total_params = sum(p.numel() for p in self.network.parameters())
        self.assertGreater(total_params, 1000)        
        self.assertLess(total_params, 10000000)        
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)
    def test_device_compatibility(self):
        """Test network works on different devices"""
        input_tensor = torch.randn(8, self.state_size)
        self.network.cpu()
        input_cpu = input_tensor.cpu()
        output_cpu = self.network(input_cpu)
        self.assertEqual(output_cpu.device.type, 'cpu')
        self.assertEqual(output_cpu.shape, (8, self.action_size))
        if torch.cuda.is_available():
            self.network.cuda()
            input_gpu = input_tensor.cuda()
            output_gpu = self.network(input_gpu)
            self.assertEqual(output_gpu.device.type, 'cuda')
            self.assertEqual(output_gpu.shape, (8, self.action_size))
class TestQNetworkEdgeCases(unittest.TestCase):
    """Test edge cases for QNetwork"""
    def test_zero_dropout(self):
        """Test network with zero dropout"""
        network = QNetwork(
            state_size=100,
            action_size=4,
            hidden_layers=[64, 32],
            dropout_rate=0.0
        )
        input_tensor = torch.randn(16, 100)
        network.train()
        output1 = network(input_tensor)
        output2 = network(input_tensor)
        torch.testing.assert_close(output1, output2)
    def test_very_small_network(self):
        """Test very small network"""
        tiny_network = QNetwork(
            state_size=2,
            action_size=2,
            hidden_layers=[4],
            dropout_rate=0.1
        )
        input_tensor = torch.randn(1, 2)
        output = tiny_network(input_tensor)
        self.assertEqual(output.shape, (1, 2))
        self.assertTrue(torch.all(torch.isfinite(output)))
    def test_single_hidden_layer(self):
        """Test network with single hidden layer"""
        single_layer_network = QNetwork(
            state_size=50,
            action_size=4,
            hidden_layers=[32],
            dropout_rate=0.2
        )
        input_tensor = torch.randn(8, 50)
        output = single_layer_network(input_tensor)
        self.assertEqual(output.shape, (8, 4))
        self.assertTrue(torch.all(torch.isfinite(output)))
if __name__ == '__main__':
    unittest.main() 