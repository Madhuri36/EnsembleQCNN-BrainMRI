import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

# --- Backbones with Backward Compatibility ---
def get_vgg19():
    try:
        return models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    except AttributeError:
        return models.vgg19(pretrained=True)

def get_resnet50():
    try:
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except AttributeError:
        return models.resnet50(pretrained=True)

def get_densenet121():
    try:
        return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    except AttributeError:
        return models.densenet121(pretrained=True)


# --- 1. Classical Custom CNN Baseline ---
class ClassicalCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
            
            nn.AdaptiveAvgPool2d((1, 1)) # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# --- 2. VGG CNN (Pre-trained classical backbone) ---
class VGGCNN(nn.Module):
    def __init__(self, num_classes=3, mode='fine_tune'):
        super().__init__()
        vgg = get_vgg19()
        
        if mode == 'freeze':
            for param in vgg.parameters():
                param.requires_grad = False
        elif mode == 'fine_tune':
            for param in vgg.parameters():
                param.requires_grad = False
            # Unfreeze block 5 of VGG-19 features (layers 28 to 36)
            for param in vgg.features[28:].parameters():
                param.requires_grad = True
        else:  # 'unfreeze'
            pass  # Keep all parameters trainable
            
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        
        # Replace VGG classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


# --- 3. ResNet CNN (Pre-trained classical backbone) ---
class ResNetCNN(nn.Module):
    def __init__(self, num_classes=3, mode='fine_tune'):
        super().__init__()
        resnet = get_resnet50()
        
        if mode == 'freeze':
            for param in resnet.parameters():
                param.requires_grad = False
        elif mode == 'fine_tune':
            for param in resnet.parameters():
                param.requires_grad = False
            # Unfreeze residual block group 4 (layer4)
            for param in resnet.layer4.parameters():
                param.requires_grad = True
        else:  # 'unfreeze'
            pass
            
        # Remove fc and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


# --- 4. DenseNet CNN (Base Paper Classical Benchmark) ---
class DenseNetCNN(nn.Module):
    def __init__(self, num_classes=3, mode='fine_tune'):
        super().__init__()
        densenet = get_densenet121()
        
        if mode == 'freeze':
            for param in densenet.parameters():
                param.requires_grad = False
        elif mode == 'fine_tune':
            for param in densenet.parameters():
                param.requires_grad = False
            # Unfreeze denseblock4 and norm5
            for param in densenet.features.denseblock4.parameters():
                param.requires_grad = True
            for param in densenet.features.norm5.parameters():
                param.requires_grad = True
        else:  # 'unfreeze'
            pass
            
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Rescale image to 128x128 as in the base paper
        x_rescaled = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = self.features(x_rescaled)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


# --- 5. Ensemble CNN (Classical ResNet + VGG Fused via Logit Ensembling) ---
class EnsembleCNN(nn.Module):
    def __init__(self, num_classes=3, mode='fine_tune'):
        super().__init__()
        resnet = get_resnet50()
        vgg = get_vgg19()
        
        if mode == 'freeze':
            for param in resnet.parameters():
                param.requires_grad = False
            for param in vgg.parameters():
                param.requires_grad = False
        elif mode == 'fine_tune':
            for param in resnet.parameters():
                param.requires_grad = False
            for param in vgg.parameters():
                param.requires_grad = False
            # Unfreeze resnet layer4 and vgg features[28:]
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            for param in vgg.features[28:].parameters():
                param.requires_grad = True
        else:  # 'unfreeze'
            pass
            
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.vgg_features = vgg.features
        self.vgg_pool = vgg.avgpool
        self.vgg_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # ResNet path
        r_feat = self.resnet_pool(self.resnet_features(x))
        r_logits = self.resnet_fc(r_feat)
        
        # VGG path
        v_feat = self.vgg_pool(self.vgg_features(x))
        v_logits = self.vgg_fc(v_feat)
        
        # Weighted average of logits: ResNet is stronger, so we give it slightly more weight (0.6 vs 0.4)
        logits = 0.6 * r_logits + 0.4 * v_logits
        return logits


# --- PennyLane QNode and PyTorch Layer Helpers ---
class QuantumConvolutionLayer(nn.Module):
    def __init__(self, n_data_qubits=4, n_ancillas=2):
        super().__init__()
        self.n_data_qubits = n_data_qubits
        self.n_ancillas = n_ancillas
        
        # Define weights directly as standard parameters (ensures optimizer tracking is never lost)
        self.weights = nn.Parameter(torch.randn(n_data_qubits, 2))
        
        total_wires = n_data_qubits + n_ancillas
        dev = qml.device("default.qubit", wires=total_wires)
        
        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # inputs shape: (4,) - representing 2x2 image patch
            # weights shape: (4, 2) - representing RX and RZ angles for quantum convolution
            
            # 1. State Preparation / Angle Embedding (using RX and RZ gates as described in the paper)
            for i in range(self.n_data_qubits):
                qml.RX(inputs[..., i], wires=i)
                qml.RZ(inputs[..., i], wires=i)
                
            # 2. Quantum Convolution (parameterized RX and RZ gates followed by CNOT entanglement)
            for i in range(self.n_data_qubits):
                qml.RX(weights[i, 0], wires=i)
                qml.RZ(weights[i, 1], wires=i)
                
            # CNOT chain for spatial neighborhood entanglement
            for i in range(self.n_data_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_data_qubits - 1, 0])
            
            # 3. SWAP-Test Based Quantum Pooling (ancillas 4 & 5)
            qml.Hadamard(wires=4)
            qml.CSWAP(wires=[4, 0, 1])
            qml.Hadamard(wires=4)
            
            qml.Hadamard(wires=5)
            qml.CSWAP(wires=[5, 2, 3])
            qml.Hadamard(wires=5)
            
            # Return the overlap similarity expectation values (pooled output: 2 features)
            return [qml.expval(qml.PauliZ(4)), qml.expval(qml.PauliZ(5))]
            
        self.qnode = qnode
        
    def forward(self, x):
        # Input shape: [batch_size, C, H, W]
        batch_size, C, H, W = x.shape
        device = x.device
        
        # Extract patches of size 2x2 with stride 2
        H_out, W_out = H // 2, W // 2
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2) # [batch_size, C, H_out, W_out, 2, 2]
        
        # Reshape patches to [batch_size * C * H_out * W_out, 4]
        patches = patches.contiguous().view(-1, 4)
        
        # Scale to [-pi, pi] for angle embedding
        patches = patches * 3.14159
        
        # Copy input and weights to CPU for quantum execution
        patches_cpu = patches.cpu()
        weights_cpu = self.weights.cpu()
        
        # Run QNode manually on CPU (retains gradients and allows PyTorch autograd backprop to GPU parameters)
        res = self.qnode(patches_cpu, weights_cpu)
        if isinstance(res, list) or isinstance(res, tuple):
            out_cpu = torch.stack(res, dim=-1)
        else:
            out_cpu = res
            if out_cpu.shape[0] == 2 and len(out_cpu.shape) == 2 and out_cpu.shape[1] != 2:
                out_cpu = out_cpu.T
                
        out = out_cpu.to(device).float()
        
        # Reshape back to [batch_size, C, H_out, W_out, 2] and permute to [batch_size, C, 2, H_out, W_out]
        # Then flatten the channel dimension to C * 2
        out = out.view(batch_size, C, H_out, W_out, 2).permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(batch_size, C * 2, H_out, W_out)
        return out


# --- 6. Normal Quantum CNN (Base Paper Quantum Model Replicated) ---
class NormalQuantumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # True patch-wise quantum convolution layer
        self.q_conv_pooling = QuantumConvolutionLayer(n_data_qubits=4, n_ancillas=2)
        
        # High-capacity classifier to stabilize training, prevent underfitting, and converge to target accuracy
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
    def forward(self, x):
        # 1. Denormalize input x (range: [0, 1]) from the ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_orig = x * std + mean
        x_orig = torch.clamp(x_orig, 0.0, 1.0)
        
        # 2. Convert to grayscale: [batch_size, 1, H, W]
        x_gray = x_orig.mean(dim=1, keepdim=True)
        
        # 3. Downsample to 64x64 bilinearly as per the base paper
        x_small = torch.nn.functional.interpolate(x_gray, size=(64, 64), mode='bilinear', align_corners=False)
        
        # 4. Quantum convolution & pooling (stride 2) -> output shape: [batch_size, 2, 32, 32]
        q_feat = self.q_conv_pooling(x_small)
        
        # 5. Classifier
        logits = self.classifier(q_feat)
        return logits


# --- 7. Ensemble QCNN (Spatial-Ensemble Quantum Framework via Late Fusion Projection & Skip Connection) ---
class EnsembleQCNN(nn.Module):
    def __init__(self, mode='fine_tune'):
        super().__init__()
        resnet = get_resnet50()
        vgg = get_vgg19()
        
        if mode == 'freeze':
            for param in resnet.parameters():
                param.requires_grad = False
            for param in vgg.parameters():
                param.requires_grad = False
        elif mode == 'fine_tune':
            for param in resnet.parameters():
                param.requires_grad = False
            for param in vgg.parameters():
                param.requires_grad = False
            # Unfreeze resnet layer4 and vgg features[28:]
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            for param in vgg.features[28:].parameters():
                param.requires_grad = True
        else:  # 'unfreeze'
            pass
            
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.vgg_features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Learnable projection network from 2560 concatenated classical features
        # down to 16 features (which will be reshaped to 1x4x4 for the 4-qubit QCNN)
        self.proj = nn.Sequential(
            nn.Linear(2560, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16)
        )
        
        # True patch-wise quantum convolution layer
        self.q_conv_pooling = QuantumConvolutionLayer(n_data_qubits=4, n_ancillas=2)
        
        # Classical skip connection mapping to align dimensions (16 classical -> 8 features)
        self.skip = nn.Linear(16, 8)
        
        # Classifier (takes 8 quantum features + 8 classical skip features = 16 features -> 3 classes)
        self.classifier = nn.Sequential(
            nn.Linear(16, 3)
        )
        
    def forward(self, x):
        r_feat = self.resnet_features(x)
        v_feat = self.vgg_features(x)
        
        r_pooled = self.avgpool(r_feat).view(x.size(0), -1)
        v_pooled = self.avgpool(v_feat).view(x.size(0), -1)
        
        # Late feature fusion
        fused = torch.cat([r_pooled, v_pooled], dim=1)
        
        # Classical projection
        projected = self.proj(fused)
        
        # Reshape to [batch_size, 1, 4, 4] for QCNN embedding
        projected_tensor = projected.view(x.size(0), 1, 4, 4)
        
        # Scale to [0, 1] for angle embedding
        projected_scaled = torch.sigmoid(projected_tensor)
        
        # Quantum convolution and pooling (outputs 8 features)
        q_feat = self.q_conv_pooling(projected_scaled) # [batch_size, 2, 2, 2]
        q_feat_flat = torch.flatten(q_feat, start_dim=1) # [batch_size, 8]
        
        # Classical skip projection (outputs 8 features)
        skip_feat = self.skip(projected) # [batch_size, 8]
        
        # Combine quantum entangled features with classical features
        combined_feat = torch.cat([q_feat_flat, skip_feat], dim=1) # [batch_size, 16]
        
        logits = self.classifier(combined_feat)
        return logits
