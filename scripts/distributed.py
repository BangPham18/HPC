import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import SimpleCNN
from data_loader import get_mnist_data_loaders
from train import load_config, train, test

def setup(rank, world_size, master_addr, master_port):
    """
    Thiết lập môi trường phân tán
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Khởi tạo quá trình phân tán
    dist.init_process_group(
        backend='gloo',  # hoặc 'nccl' nếu sử dụng GPU
        init_method=f'env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """
    Dọn dẹp môi trường phân tán
    """
    dist.destroy_process_group()

def run_worker(rank, world_size, master_addr, master_port):
    """
    Hàm chạy cho mỗi worker
    """
    # Thiết lập môi trường phân tán
    setup(rank, world_size, master_addr, master_port)
    
    # Tải cấu hình
    config = load_config()
    
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] Using device: {device}")
    
    # Tạo model, wrap với DDP, và di chuyển tới thiết bị
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=None if device.type == 'cpu' else [device.index])
    
    # Thiết lập loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=config['learning_rate'])
    
    # Tải dữ liệu với DistributedSampler
    train_loader, test_loader = get_mnist_data_loaders(
        batch_size=config['batch_size'],
        distributed=True,
        world_size=world_size,
        rank=rank
    )
    
    # Tạo thư mục output 
    if rank == 0:
        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/logs", exist_ok=True)
        writer = SummaryWriter(log_dir='outputs/logs')
    else:
        writer = None
    
    # Huấn luyện model
    best_acc = 0.0
    for epoch in range(config['epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Đảm bảo DistributedSampler trộn dữ liệu khác nhau mỗi epoch
        train_loader.sampler.set_epoch(epoch)
        
        train(ddp_model, train_loader, optimizer, criterion, device, epoch, writer if rank == 0 else None)
        
        # Chỉ đánh giá và lưu model trên node master (rank=0)
        if rank == 0:
            accuracy = test(ddp_model, test_loader, criterion, device, epoch, writer)
            
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
    
    if rank == 0:
        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
        if writer:
            writer.close()
    
    # Dọn dẹp
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed Deep Learning Training')
    parser.add_argument('--rank', type=int, default=None, help='Node rank for distributed training')
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)), 
                        help='Number of processes for distributed training')
    parser.add_argument('--master-addr', type=str, default=os.environ.get('MASTER_ADDR', 'localhost'),
                        help='Master node address')
    parser.add_argument('--master-port', type=str, default=os.environ.get('MASTER_PORT', '29500'),
                        help='Master node port')
    args = parser.parse_args()
    
    # Xác định rank nếu không được cung cấp
    if args.rank is None:
        # Trong Docker Swarm, chúng ta cần tự xác định rank
        hostname = os.uname()[1]
        if hostname == args.master_addr or hostname == 'master':
            args.rank = 0
        else:
            # Gán rank động dựa trên hostname
            # Lưu ý: Đây chỉ là ví dụ đơn giản, trong thực tế cần cơ chế phức tạp hơn
            import hashlib
            worker_id = int(hashlib.md5(hostname.encode()).hexdigest(), 16) % 1000
            args.rank = worker_id % (args.world_size - 1) + 1
    
    print(f"Starting process with rank {args.rank} out of {args.world_size}")
    run_worker(args.rank, args.world_size, args.master_addr, args.master_port)

if __name__ == "__main__":
    main()