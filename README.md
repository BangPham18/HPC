# Dự án huấn luyện Deep Learning với Docker Swarm

Dự án này triển khai một hệ thống phân tán sử dụng Docker Swarm để huấn luyện mô hình deep learning trên nhiều node. Mô hình sử dụng là một mạng CNN đơn giản được huấn luyện trên tập dữ liệu MNIST.

## Yêu cầu

- Docker Engine phiên bản 19.03 trở lên
- Docker Swarm được thiết lập (ít nhất một node manager và các node worker)
- Mạng giữa các node cần được cấu hình đúng để cho phép giao tiếp

## Thiết lập Docker Swarm

Nếu bạn chưa thiết lập Docker Swarm, hãy làm theo các bước sau:

1. Khởi tạo Swarm trên node manager:
```bash
   docker swarm init --advertise-addr <IP_ADDRESS>
```
2. Register worker nodes using the command shown after initializing Swarm:
```bash
    docker swarm join --token <TOKEN> <IP_ADDRESS>:2377
```
3. Check nodes in Swarm:
```bash
docker node ls
```

##How to use

Clone the project to the manager node:
bashgit clone <repository-url>
cd docker-swarm-dl

Adjust the parameters in the file .envas needed:
WORLD_SIZE=3          # Tổng số process (1 master + 2 worker)
WORKER_REPLICAS=2     # Số container worker cần tạo

Build and deploy stack:
bashdocker build -t dl-training:latest .
docker stack deploy -c docker-compose.yml dl-training

Track training progress:
bashdocker service logs -f dl-training_master

Once completed, the trained model will be saved in the folderoutputs/models/
Stop and clear stack when done:
bashdocker stack rm dl-training


Directory structure

Dockerfile: Define container for training
docker-compose.yml: Docker Swarm Deployment Configuration
requirements.txt: Required Python libraries
scripts/: Contains training scripts and model definitions
configs/: Configuration files for training
.env: Environment variables for Docker Compose

Extend
You can extend this project by:

Change the model inscripts/model.py
Use different datasets by adjustingscripts/data_loader.py
Tuning the hyperparameters inconfigs/training_config.json
Add monitoring features by integrating with Prometheus and Grafana
Optimize data sharing between workers