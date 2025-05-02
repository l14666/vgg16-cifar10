import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.vgg16 import vgg16
from utils.data_loader import get_cifar10_loaders, get_cifar10_classes

def train(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备(CPU/GPU)
    
    返回:
        train_loss: 平均训练损失
        train_acc: 训练准确率
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 处理BCEWithLogitsLoss的特殊情况
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            # 将目标转换为one-hot编码
            target_one_hot = torch.zeros(targets.size(0), 10, device=device)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            loss = criterion(outputs, target_one_hot)
        else:
            loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        
        # 对于不同的输出激活函数，预测方式可能不同
        if isinstance(criterion, torch.nn.NLLLoss):
            # 对于NLLLoss，输出已经是log-probabilities
            _, predicted = outputs.max(1)
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            # 对于BCEWithLogitsLoss，需要应用sigmoid并取最大值
            predicted = torch.sigmoid(outputs).max(1)[1]
        else:
            # 对于其他损失函数，直接取最大值
            _, predicted = outputs.max(1)
            
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Batch: {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.3f}%')
    
    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    """
    测试模型
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备(CPU/GPU)
    
    返回:
        test_loss: 平均测试损失
        test_acc: 测试准确率
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 处理BCEWithLogitsLoss的特殊情况
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                # 将目标转换为one-hot编码
                target_one_hot = torch.zeros(targets.size(0), 10, device=device)
                target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                loss = criterion(outputs, target_one_hot)
            else:
                loss = criterion(outputs, targets)
            
            # 统计
            test_loss += loss.item()
            
            # 对于不同的输出激活函数，预测方式可能不同
            if isinstance(criterion, torch.nn.NLLLoss):
                # 对于NLLLoss，输出已经是log-probabilities
                _, predicted = outputs.max(1)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                # 对于BCEWithLogitsLoss，需要应用sigmoid并取最大值
                predicted = torch.sigmoid(outputs).max(1)[1]
            else:
                # 对于其他损失函数，直接取最大值
                _, predicted = outputs.max(1)
                
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.3f} | Test Acc: {100.*correct/total:.3f}%')
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='VGG16 on CIFAR-10')
    
    # 数据集参数
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading (default: 4)')
    parser.add_argument('--augment', action='store_true', default=True, help='use data augmentation')
    
    # 模型参数
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu'], 
                        help='activation function (default: relu)')
    parser.add_argument('--kernel-size', type=int, default=3, help='kernel size for convolution (default: 3)')
    parser.add_argument('--feature-maps', type=int, default=64, help='initial feature maps (default: 64)')
    parser.add_argument('--fc-layers', type=int, default=3, help='number of fully connected layers (default: 3)')
    parser.add_argument('--neurons', type=int, default=4096, help='number of neurons in fully connected layers (default: 4096)')
    parser.add_argument('--init-weights', action='store_true', default=True, help='initialize weights')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop'], 
                        help='optimizer (default: sgd)')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'nll', 'focal'], 
                        help='loss function (default: cross_entropy)')
    
    # 其他参数
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 检查CUDA可用性
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载数据
    print("==> 准备数据...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment
    )
    
    # 创建模型
    print("==> 创建模型...")
    model = vgg16(
        num_classes=10,
        init_weights=args.init_weights,
        activation=args.activation,
        kernel_size=args.kernel_size,
        feature_maps=args.feature_maps,
        fc_layers=args.fc_layers,
        neurons=args.neurons
    ).to(device)
    
    # 定义损失函数
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'nll':
        # 负对数似然损失，需要在模型最后一层添加LogSoftmax
        criterion = nn.NLLLoss()
    elif args.loss == 'focal':
        # Focal Loss，用于处理类别不平衡问题
        gamma = 2.0
        alpha = 0.25
        def focal_loss(pred, target):
            ce_loss = torch.nn.functional.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = alpha * (1 - pt) ** gamma * ce_loss
            return loss.mean()
        criterion = focal_loss
    
    # 定义优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=10, 
        verbose=True
    )
    
    # 训练模型
    print("==> 开始训练...")
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # 训练和测试
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # 调整学习率
        scheduler.step(test_loss)
        
        # 保存最佳模型
        if test_acc > best_acc:
            print('保存模型...')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'args': args
            }
            torch.save(state, f'{args.save_dir}/best_model.pth')
            best_acc = test_acc
    
    print(f'最佳测试准确率: {best_acc:.3f}%')

if __name__ == '__main__':
    main()
