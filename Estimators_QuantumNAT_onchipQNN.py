import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add PennyLane import at the top
try:
    import pennylane as qml
    from pennylane import numpy as pnp

    PENNYLANE_AVAILABLE = True
except ImportError:
    print("Warning: PennyLane not available. Quantum components will not work.")
    PENNYLANE_AVAILABLE = False


    # Create dummy qml for fallback
    class DummyQML:
        def device(self, *args, **kwargs):
            return None

        def QNode(self, *args, **kwargs):
            return lambda x: x

        def qnn(self):
            class DummyTorchLayer(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return x

            return DummyTorchLayer


    qml = DummyQML()


# the DCE network for Pilot_num 128
class DCE_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.kernel_size = 3
        self.padding = 1
        layers = []
        # the first layer
        layers.append(nn.Conv2d(in_channels=2, out_channels=self.features, kernel_size=self.kernel_size, stride=1,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.BatchNorm2d(self.features))
        layers.append(nn.ReLU(inplace=True))

        # the second and the third layer
        for i in range(2):
            layers.append(
                nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(self.features))
            layers.append(nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(*layers)

        # the linear layer
        self.FC = nn.Linear(self.features * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(x.shape[0], self.features * 16 * 8)
        # print(x.shape)
        x = self.FC(x)
        return x


# the scenario classifier for Pilot_num 128
class SC_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                               bias=False)
        self.FC = nn.Linear(32 * 4 * 2, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(x.shape[0], 32 * 4 * 2)
        # print(x.shape)
        x = self.FC(x)
        return F.log_softmax(x, dim=1)


# ============================
# Quantum Scenario Classifier
# ============================
class QSC_P128(nn.Module):
    def __init__(self, n_qubits=6, n_layers=3, n_classes=3, use_quantumnat=True, use_gradient_pruning=True):
        super(QSC_P128, self).__init__()

        self.use_quantum = True
        self.num_qubits = n_qubits
        self.n_classes = n_classes
        self.use_quantumnat = use_quantumnat
        self.use_gradient_pruning = use_gradient_pruning

        # QuantumNAT parameters - more conservative values
        self.noise_level = 0.01 if use_quantumnat else 0.0  # Reduced from 0.02
        self.gradient_threshold = 0.1  # Increased from 0.05 for more conservative pruning

        # 1. Define quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # 2. KEEP ORIGINAL QUANTUM CIRCUIT - stability is key!
        def quantum_circuit(inputs, weights):
            # Original embedding - proven to work
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            # ORIGINAL ANSATZ - don't change what works
            for layer in range(n_layers):
                # Original rotations
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)

                # Original entanglement - proven pattern
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            # Original measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # 3. Keep original weight shapes
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}

        # Create the QNode
        qnode = qml.QNode(quantum_circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 4. KEEP ORIGINAL PREPROCESSING - don't overcomplicate
        self.preprocess = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 8, 4)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 4, 2)
            nn.Flatten(),
            nn.Linear(32 * 4 * 2, n_qubits),  # Original: 256 -> n_qubits
            nn.Tanh()  # Scale to [-1, 1] range
        )

        # 5. KEEP ORIGINAL CLASSIFIER - proven to work
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        if not self.use_quantum:
            x = self.classical_fallback(x)
            return F.log_softmax(x, dim=1)

        # Classical preprocessing
        x = self.preprocess(x)

        # QuantumNAT: SIMPLIFIED noise injection
        if self.training and self.use_quantumnat and self.noise_level > 0:
            # Simple noise injection without parameter saving/restoring
            with torch.no_grad():
                # Get current quantum parameters
                quantum_params = list(self.qlayer.parameters())

                # Store original values
                original_values = [param.data.clone() for param in quantum_params]

                # Apply noise
                for param in quantum_params:
                    noise = self.noise_level * torch.randn_like(param)
                    param.data.add_(noise)

            # Forward pass with noisy parameters
            x_quantum = self.qlayer(x)

            # Restore original parameters immediately
            with torch.no_grad():
                for param, original in zip(quantum_params, original_values):
                    param.data.copy_(original)
        else:
            # Standard forward pass
            x_quantum = self.qlayer(x)

        # Final classification
        x = self.classifier(x_quantum)
        return F.log_softmax(x, dim=1)

    def apply_gradient_pruning(self):
        """On-chip QNN: Conservative gradient pruning"""
        if not self.use_gradient_pruning:
            return

        total_params = 0
        pruned_params = 0

        for name, param in self.named_parameters():
            if param.grad is not None:
                total_params += param.grad.numel()

                # Conservative pruning: only prune very small gradients
                mask = torch.abs(param.grad) > self.gradient_threshold
                pruned_count = (~mask).sum().item()
                pruned_params += pruned_count

                param.grad = param.grad * mask.float()

        # Optional: Log pruning ratio for monitoring
        if total_params > 0 and pruned_params > 0:
            pruning_ratio = pruned_params / total_params
            if pruning_ratio > 0.1:  # Only log if significant pruning occurred
                print(f"Gradient pruning: {pruning_ratio:.1%} gradients pruned")







# the feature extractor for Pilot_num 128
class Conv_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.kernel_size = 3
        self.padding = 1
        layers = []
        # the first layer
        layers.append(
            nn.Conv2d(in_channels=2, out_channels=self.features, kernel_size=self.kernel_size, padding=self.padding,
                      bias=False))
        layers.append(nn.BatchNorm2d(self.features))
        layers.append(nn.ReLU(inplace=True))

        # the second and the third layer
        for i in range(2):
            # the second layer
            layers.append(nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.features))
            layers.append(nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(x.shape[0], self.features * 16 * 8)

        return x


# the feature mapper for Pilot_num 128
class FC_P128(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(32 * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.FC(x)
        return x


def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x_hat - x) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        return nmse