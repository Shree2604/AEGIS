# week4_client.py
# CLIENT: Sends models using BOTH baseline and optimized strategies
# Run on Raspberry Pi or any machine

import os
import sys
import time
import json
import socket
import pickle
import gzip
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ----------------------------
# Model Definition
# ----------------------------

class DisasterCNN(nn.Module):
    def __init__(self, num_classes, channel_sizes=None):
        super(DisasterCNN, self).__init__()
        
        if channel_sizes is None:
            channel_sizes = [16, 32, 64, 128]
        
        self.features = nn.Sequential(
            nn.Conv2d(3, channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channel_sizes[3] * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.channel_sizes = channel_sizes
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# Dataset Loader
# ----------------------------

def load_disaster_dataset(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls_name])
    
    return image_paths, labels, class_names

class DisasterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ----------------------------
# Communication Strategies
# ----------------------------

def serialize_baseline(state_dict):
    """Strategy 1: Baseline (Full FP32)"""
    start = time.time()
    serialized = pickle.dumps(state_dict)
    elapsed = time.time() - start
    return serialized, elapsed, 'baseline'

def serialize_fp16_gzip(state_dict):
    """Strategy 2: FP16 + Gzip (Optimized)"""
    start = time.time()
    
    # Convert to FP16
    fp16_state = {}
    for key, value in state_dict.items():
        if value.dtype == torch.float32:
            fp16_state[key] = value.half()
        else:
            fp16_state[key] = value
    
    # Pickle and compress
    pickled = pickle.dumps(fp16_state)
    compressed = gzip.compress(pickled, compresslevel=6)
    
    elapsed = time.time() - start
    return compressed, elapsed, 'fp16_gzip'

def serialize_fc_only(state_dict):
    """Strategy 3: FC Layers Only"""
    start = time.time()
    fc_state = {k: v for k, v in state_dict.items() if 'classifier' in k}
    serialized = pickle.dumps(fc_state)
    elapsed = time.time() - start
    return serialized, elapsed, 'fc_only'

def serialize_delta(state_dict, prev_state):
    """Strategy 4: Delta Updates"""
    start = time.time()
    
    if prev_state is None:
        serialized = pickle.dumps(state_dict)
    else:
        delta_state = {}
        for key in state_dict.keys():
            if key in prev_state:
                delta_state[key] = state_dict[key] - prev_state[key]
            else:
                delta_state[key] = state_dict[key]
        serialized = pickle.dumps(delta_state)
    
    elapsed = time.time() - start
    return serialized, elapsed, 'delta'

# ----------------------------
# Client Training Function
# ----------------------------

def train_client(client_id, data_loader, num_epochs, global_state=None, prev_state=None, learning_rate=0.0005):
    """Train local model"""
    print(f"\n[Client {client_id}] Starting training with learning rate: {learning_rate}...")
    
    model = DisasterCNN(num_classes=4)
    
    # Load global model if available
    if global_state is not None:
        model.load_state_dict(global_state, strict=False)
        print(f"[Client {client_id}] Loaded global model")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_loss = epoch_loss / epoch_batches
        total_loss += avg_loss
        num_batches += 1
        
        print(f"[Client {client_id}]   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    final_loss = total_loss / num_batches
    return model, final_loss

# ----------------------------
# Send to Server
# ----------------------------

def send_to_server(server_ip, server_port, client_id, round_num, model, 
                   training_loss, strategy_name, prev_state=None):
    """Send model to server using specified strategy"""
    
    state_dict = model.state_dict()
    
    # Serialize based on strategy
    if strategy_name == 'baseline':
        model_bytes, serialize_time, strategy = serialize_baseline(state_dict)
    elif strategy_name == 'fp16_gzip':
        model_bytes, serialize_time, strategy = serialize_fp16_gzip(state_dict)
    elif strategy_name == 'fc_only':
        model_bytes, serialize_time, strategy = serialize_fc_only(state_dict)
    elif strategy_name == 'delta':
        model_bytes, serialize_time, strategy = serialize_delta(state_dict, prev_state)
    else:
        model_bytes, serialize_time, strategy = serialize_baseline(state_dict)
    
    size_kb = len(model_bytes) / 1024
    
    print(f"[Client {client_id}] 📦 {strategy}: {size_kb:.2f} KB, {serialize_time*1000:.2f} ms")
    
    # Send to server
    try:
        upload_start = time.time()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip, server_port))
            
            # Send metadata
            metadata = {
                'client_id': client_id,
                'round': round_num,
                'strategy': strategy,
                'model_size': len(model_bytes),
                'serialize_time': serialize_time,
                'training_loss': training_loss
            }
            
            metadata_json = json.dumps(metadata).encode()
            s.sendall(len(metadata_json).to_bytes(4, 'big'))
            s.sendall(metadata_json)
            
            # Send model
            s.sendall(model_bytes)
            
            # Receive acknowledgment
            ack = s.recv(1024).decode()
        
        upload_time = time.time() - upload_start
        
        print(f"[Client {client_id}] ✅ Upload complete: {upload_time:.2f}s")
        
        return {
            'strategy': strategy,
            'size_kb': size_kb,
            'serialize_time': serialize_time,
            'upload_time': upload_time,
            'total_time': serialize_time + upload_time
        }
        
    except Exception as e:
        print(f"[Client {client_id}] ❌ Upload failed: {e}")
        return None

# ----------------------------
# Main Client Loop
# ----------------------------

def run_client(data_dir, server_ip, server_port, num_rounds=10, num_epochs=25, batch_size=64, learning_rate=0.0005):
    """Main client loop with optimization comparison"""
    
    print("="*70)
    print("🚀 Week 4 Client: Communication Optimization")
    print("="*70)
    print(f"Server: {server_ip}:{server_port}")
    print(f"Dataset: {data_dir}")
    print("="*70)
    
    # Load dataset
    print("\n📁 Loading AIDER dataset...")
    image_paths, labels, class_names = load_disaster_dataset(data_dir)
    print(f"✅ Loaded {len(image_paths)} images from {len(class_names)} classes")
    
    # Create dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DisasterDataset(image_paths, labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Strategies to test (each client uses different strategy per round)
    strategies = ['baseline', 'fp16_gzip', 'fc_only', 'delta']
    
    comparison_results = {
        'baseline': [],
        'fp16_gzip': [],
        'fc_only': [],
        'delta': []
    }
    
    prev_state = None
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"📡 Round {round_num}/{num_rounds}")
        print(f"{'='*70}")
        
        # Train model
        model, training_loss = train_client(
            client_id=1,
            data_loader=data_loader,
            num_epochs=num_epochs,
            prev_state=prev_state,
            learning_rate=learning_rate
        )
        
        # Test ALL strategies for comparison
        print(f"\n📊 Testing all communication strategies...")
        
        for strategy in strategies:
            result = send_to_server(
                server_ip, server_port,
                client_id=1,
                round_num=round_num,
                model=model,
                training_loss=training_loss,
                strategy_name=strategy,
                prev_state=prev_state
            )
            
            if result:
                comparison_results[strategy].append(result)
            
            time.sleep(0.5)  # Small delay between uploads
        
        # Save current state for delta updates
        prev_state = model.state_dict()
        
        print(f"\n✅ Round {round_num} complete")
        time.sleep(2)
    
    # Generate comparison report
    print(f"\n{'='*70}")
    print(f"📊 Communication Optimization Results")
    print(f"{'='*70}")
    
    print(f"\n{'Strategy':<15} {'Avg Size (KB)':<15} {'Avg Time (ms)':<15} {'Reduction':<10}")
    print("-"*70)
    
    baseline_size = np.mean([r['size_kb'] for r in comparison_results['baseline']])
    
    for strategy in strategies:
        results = comparison_results[strategy]
        if results:
            avg_size = np.mean([r['size_kb'] for r in results])
            avg_time = np.mean([r['serialize_time'] for r in results]) * 1000
            reduction = (1 - avg_size / baseline_size) * 100
            
            print(f"{strategy:<15} {avg_size:<15.2f} {avg_time:<15.2f} {reduction:<10.1f}%")
    
    # Save results
    os.makedirs('Week4_Client_Results', exist_ok=True)
    
    import csv
    with open('Week4_Client_Results/communication_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Strategy', 'Size_KB', 'Serialize_Time_ms', 'Upload_Time_s'])
        
        for strategy, results in comparison_results.items():
            for i, result in enumerate(results, 1):
                writer.writerow([
                    i,
                    result['strategy'],
                    f"{result['size_kb']:.2f}",
                    f"{result['serialize_time']*1000:.2f}",
                    f"{result['upload_time']:.2f}"
                ])
    
    print(f"\n✅ Results saved to Week4_Client_Results/communication_comparison.csv")
    print(f"{'='*70}\n")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python week4_client.py <server_ip> <data_dir>")
        print("Example: python week4_client.py 192.168.1.100 AIDER/AIDER")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    data_dir = sys.argv[2]
    server_port = 9999
    
    # Default values match the specified metrics
    run_client(
        data_dir=data_dir,
        server_ip=server_ip,
        server_port=server_port,
        num_rounds=10,
        num_epochs=25,
        batch_size=64,
        learning_rate=0.0005
    )
