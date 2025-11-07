# pi_client.py
# Run on Raspberry Pi - Simulates 4 UAV clients with different constraints

import os
import sys
import time
import json
import socket
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import io
import pickle
from multiprocessing import Process, Queue

# ----------------------------
# Simplified CNN Model (matches your FedMA model structure)
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
# Dataset (simple version for testing)
# ----------------------------

class SimpleDisasterDataset(Dataset):
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
# UAV Client Constraints
# ----------------------------

UAV_CONSTRAINTS = {
    'UAV-1': {
        'name': 'Normal',
        'epochs': 5,
        'bandwidth_throttle': 1.0,  # 100%
        'skip_rounds': False,
        'battery_level': 100
    },
    'UAV-2': {
        'name': 'Low Battery',
        'epochs': 2,  # Reduced epochs to save energy
        'bandwidth_throttle': 1.0,
        'skip_rounds': False,
        'battery_level': 30
    },
    'UAV-3': {
        'name': 'Slow WiFi',
        'epochs': 5,
        'bandwidth_throttle': 0.5,  # 50% bandwidth (simulated delay)
        'skip_rounds': False,
        'battery_level': 80
    },
    'UAV-4': {
        'name': 'High CPU Load',
        'epochs': 3,
        'bandwidth_throttle': 1.0,
        'skip_rounds': True,  # Skip alternate rounds
        'battery_level': 60
    }
}

# ----------------------------
# Energy & Resource Monitor
# ----------------------------

class ResourceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_cpu_percent = None
        self.start_temp = None
    
    def start(self):
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.start_temp = self.get_temperature()
    
    def get_temperature(self):
        """Get CPU temperature on Raspberry Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return temp
        except:
            return 0.0
    
    def stop(self):
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        end_temp = self.get_temperature()
        
        duration = end_time - self.start_time
        avg_cpu = (self.start_cpu_percent + end_cpu_percent) / 2
        temp_delta = end_temp - self.start_temp
        
        return {
            'duration_sec': duration,
            'avg_cpu_percent': avg_cpu,
            'start_temp_c': self.start_temp,
            'end_temp_c': end_temp,
            'temp_delta_c': temp_delta
        }

# ----------------------------
# Virtual Client Worker
# ----------------------------

def client_worker(client_id, constraint, data_subset, server_host, server_port, round_num, result_queue):
    """
    Worker process for each virtual UAV client
    """
    print(f"\n[{client_id}] Starting training - Constraint: {constraint['name']}")
    
    # Check if we should skip this round
    if constraint['skip_rounds'] and round_num % 2 == 0:
        print(f"[{client_id}] ⏭️  Skipping round {round_num} (High CPU Load constraint)")
        result_queue.put({
            'client_id': client_id,
            'status': 'skipped',
            'round': round_num
        })
        return
    
    # Initialize resource monitor
    monitor = ResourceMonitor()
    monitor.start()
    
    # Create model
    model = DisasterCNN(num_classes=4)  # Assuming 4 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloader for this client
    # (In real scenario, this would be loaded from local data)
    # For testing, we'll create dummy data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Training loop
    model.train()
    total_loss = 0
    num_batches = 0
    
    num_epochs = constraint['epochs']
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        # Simulate training on local data
        # In real scenario, you'd iterate over client's dataloader
        for batch_idx in range(3):  # Simulate 3 batches
            # Create dummy batch (replace with real data)
            data = torch.randn(8, 3, 64, 64)  # 8 images
            target = torch.randint(0, 4, (8,))  # 4 classes
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_epoch_loss = epoch_loss / epoch_batches
        total_loss += avg_epoch_loss
        num_batches += epoch_batches
        
        print(f"[{client_id}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    avg_loss = total_loss / num_batches
    
    # Stop monitoring
    metrics = monitor.stop()
    
    # Serialize model state dict
    model_bytes = pickle.dumps(model.state_dict())
    model_size_kb = len(model_bytes) / 1024
    
    # Simulate bandwidth throttling (add delay)
    upload_delay = (1.0 / constraint['bandwidth_throttle']) - 1.0
    if upload_delay > 0:
        print(f"[{client_id}] 🌐 Simulating slow upload ({constraint['bandwidth_throttle']*100:.0f}% bandwidth)...")
        time.sleep(upload_delay * 2)  # Artificial delay
    
    # Send model update to server
    try:
        print(f"[{client_id}] 📤 Uploading model ({model_size_kb:.2f} KB) to server...")
        upload_start = time.time()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_host, server_port))
            
            # Send metadata
            metadata = {
                'client_id': client_id,
                'round': round_num,
                'model_size': len(model_bytes),
                'training_loss': avg_loss,
                'metrics': metrics,
                'battery_level': constraint['battery_level']
            }
            metadata_json = json.dumps(metadata).encode()
            s.sendall(len(metadata_json).to_bytes(4, 'big'))
            s.sendall(metadata_json)
            
            # Send model
            s.sendall(model_bytes)
            
            # Receive acknowledgment
            ack = s.recv(1024).decode()
            print(f"[{client_id}] ✅ Server response: {ack}")
        
        upload_time = time.time() - upload_start
        
        result_queue.put({
            'client_id': client_id,
            'status': 'success',
            'round': round_num,
            'training_loss': avg_loss,
            'upload_time_sec': upload_time,
            'model_size_kb': model_size_kb,
            'metrics': metrics
        })
        
    except Exception as e:
        print(f"[{client_id}] ❌ Error uploading to server: {e}")
        result_queue.put({
            'client_id': client_id,
            'status': 'failed',
            'error': str(e)
        })

# ----------------------------
# Main Pi Client Controller
# ----------------------------

def run_pi_clients(server_host, server_port, num_rounds=5):
    """
    Main function to spawn 4 virtual UAV clients
    """
    print("="*60)
    print("🚁 AEGIS Phase-3: Raspberry Pi UAV Client Simulator")
    print("="*60)
    print(f"Server: {server_host}:{server_port}")
    print(f"Virtual Clients: 4 (UAV-1, UAV-2, UAV-3, UAV-4)")
    print("="*60)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"📡 Round {round_num}/{num_rounds}")
        print(f"{'='*60}")
        
        # Create result queue for collecting results
        result_queue = Queue()
        
        # Spawn all 4 clients in parallel
        processes = []
        for client_id, constraint in UAV_CONSTRAINTS.items():
            p = Process(
                target=client_worker,
                args=(client_id, constraint, None, server_host, server_port, round_num, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Wait for all clients to finish
        for p in processes:
            p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Print round summary
        print(f"\n{'='*60}")
        print(f"📊 Round {round_num} Summary")
        print(f"{'='*60}")
        for res in results:
            if res['status'] == 'success':
                print(f"{res['client_id']}: ✅ Loss={res['training_loss']:.4f}, "
                      f"Upload={res['upload_time_sec']:.2f}s, "
                      f"Size={res['model_size_kb']:.2f}KB")
            elif res['status'] == 'skipped':
                print(f"{res['client_id']}: ⏭️  Skipped (constraint)")
            else:
                print(f"{res['client_id']}: ❌ Failed - {res.get('error', 'Unknown')}")
        
        print(f"\nWaiting for server aggregation...")
        time.sleep(3)  # Give server time to aggregate
        
        # TODO: Download new global model from server
        # (Will implement in Week 3)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pi_client.py <server_ip>")
        print("Example: python pi_client.py 192.168.1.100")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    server_port = 9999
    
    run_pi_clients(server_ip, server_port, num_rounds=3)
