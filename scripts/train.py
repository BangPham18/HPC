import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import SimpleCNN
from data_loader import get_mnist_data_loaders

def load_config(config_path='configs/training_config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def train(model, train_loader, optimizer, criterion, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
            if writer:
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('accuracy', 100. * correct / total, epoch * len(train_loader) + batch_idx)
            running_loss = 0.0

def test(model, test_loader, criterion, device, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    if writer:
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('test accuracy', accuracy, epoch)
    
    return accuracy

def main():
    # Tải cấu hình
    config = load_config()
    
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tạo thư mục cho outputs
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    
    # Thiết lập TensorBoard writer
    writer = SummaryWriter(log_dir='outputs/logs')
    
    # Tạo model và di chuyển tới thiết bị
    model = SimpleCNN().to(device)
    
    # Thiết lập loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Tải dữ liệu
    train_loader, test_loader = get_mnist_data_loaders(
        batch_size=config['batch_size']
    )
    
    # Huấn luyện model
    best_acc = 0.0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        start_time = time.time()
        
        train(model, train_loader, optimizer, criterion, device, epoch, writer)
        accuracy = test(model, test_loader, criterion, device, epoch, writer)
        
        # Lưu model tốt nhất
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f"outputs/models/best_model.pth")
        
        # Lưu checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'accuracy': accuracy
            }, f"outputs/models/checkpoint_epoch_{epoch+1}.pth")
        
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()