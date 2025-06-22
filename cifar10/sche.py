from cnn import Net
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def train(use_lr_scheduler='none', scheduler_type='steplr'):
    is_support = torch.cuda.is_available()
    if is_support:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    CIFAR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 0.015
    #----------------- 标准化处理 ------------------
    # transforms.ToTensor()将图像数据从[0, 255]的整数范围缩放到[0.0, 1.0]的浮点数范围。
    # 调整张量维度顺序为(C, H, W)（通道、高度、宽度）
    # transforms.Normalize(mean, std)对张量进行标准化（归一化），使数据分布接近零均值、单位方差
    # 计算公式：(输入值 - 均值)/标准差
    # 最终每个像素值被映射到[-1, 1]区间
    # transform是标准化处理参数
    train_transform = transforms.Compose(
    [transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),          # 随机裁剪（保留主体）
    transforms.RandomHorizontalFlip(p=0.5),        # 水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色扰动
    transforms.RandomRotation(10),                 # 小幅旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #--------------- 加载训练集、验证集、测试集 -----------------
    # 使用 torchvision.datasets.CIFAR10 加载 CIFAR-10 数据集，并将其分为训练集和测试集
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    
    # MNIST
    # full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    # testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
   
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    valset.dataset.transform = test_transform
    #---------------- 创建数据加载器  ---------------------
    # BATCH_SIZE: 每批加载BATCH_SIZE张图像,根据GPU显存调整
    # num_workers=8：使用8个子进程并行加载数据（加速数据读取）CPU核心数的1/2~3/4
    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valloader = DataLoader(dataset=valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    net = Net()
    net.to(device)

    #-------------- 定义损失函数,优化器 --------------
    # nn.CrossEntropyLoss()输入网络输出的logits,输出标量损失值
    # lr -> 学习率（控制参数更新步长，关键超参数）
    # momentum -> 动量（加速收敛，减少震荡）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    # 学习率调度器设置
    scheduler = None
    if use_lr_scheduler != 'none':
        if scheduler_type == 'steplr':
            scheduler = StepLR(optimizer, step_size=11, gamma=0.67)  # 每10个epoch学习率×0.67
            print(f"Using StepLR scheduler: step_size=10, gamma=0.67")
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
            print(f"Using ReduceLROnPlateau scheduler: factor=0.5, patience=5")
    
    # 记录学习率变化
    learning_rates = []
    
    #------------- 训练网络 ------------
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    # EPOCHS -> 训练的轮次
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            if i % 10 == 9:      # 每10个batch触发一次
                avg_loss = running_loss / 10
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(trainloader)}], Loss: {avg_loss:.4f}')
                train_losses.append(avg_loss)
                running_loss = 0.0

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Accuracy: {val_acc:.2f}%')

        # 每个epoch结束时更新学习率
        if scheduler is not None:
            if scheduler_type == 'steplr':
                scheduler.step()
            elif scheduler_type == 'plateau':
                scheduler.step(val_acc)  # 根据验证准确率调整
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            print(f"Current learning rate: {current_lr:.7f}")
        else:
            learning_rates.append(optimizer.param_groups[0]['lr'])

        #------------------- 保存模型 -----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), './model/sche_best_model_cifar10.pth')
            # MNIST
            # torch.save(net.state_dict(), './model/mnist/sche_best_model_mnist.pth')
            print(f'Saved new best model with accuracy: {val_acc:.2f}%')
                    
    print('Finished Training')

    # ------------在整个测试集上测试-------------------------------------------
    # **************** 改进9: 最终测试评估 ****************
    # 加载最佳模型
    net.load_state_dict(torch.load('./model/sche_best_model_cifar10.pth', weights_only=True))
    # MNIST
    # net.load_state_dict(torch.load('./model/mnist/sche_best_model_mnist.pth', weights_only=True))
    net.eval()
    
    # 在测试集上评估
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测和标签用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    
    # **************** 改进10: 可视化分析 ****************
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./model/sche_training_metrics.png')
    # plt.savefig('./model/mnist/sche_training_metrics.png')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_names, yticklabels=CIFAR10_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./model/sche_confusion_matrix.png')
    # plt.savefig('./model/mnist/sche_confusion_matrix.png')

    # 添加学习率变化曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule ({scheduler_type})')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./model/lr_schedule_{use_lr_scheduler}_{scheduler_type}.png')
    # plt.savefig(f'./model/mnist/lr_schedule_{use_lr_scheduler}_{scheduler_type}.png')

def main():
    experiments = [
        # {'use_lr_scheduler': 'none', 'scheduler_type': ''},       # 对照组：无调度器
        {'use_lr_scheduler': 'step', 'scheduler_type': 'steplr'},  # 实验组1：StepLR
        # {'use_lr_scheduler': 'plateau', 'scheduler_type': 'plateau'}  # 实验组2：ReduceLROnPlateau
    ]

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Starting experiment: Scheduler={exp['use_lr_scheduler']} Type={exp['scheduler_type']}")
        print(f"{'='*50}")
        train(use_lr_scheduler=exp['use_lr_scheduler'], scheduler_type=exp['scheduler_type'])
        print(f"\n{'='*50}")
        print(f"Completed experiment: Scheduler={exp['use_lr_scheduler']} Type={exp['scheduler_type']}")
        print(f"{'='*50}\n")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  # 防止打包成 exe 时出错
    main()
