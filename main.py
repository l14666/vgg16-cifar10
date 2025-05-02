import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.vgg16 import vgg16
from utils.data_loader import get_cifar10_loaders, get_cifar10_classes
from train import train, test
def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='VGG16 on CIFAR-10')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'],
                        help='运行模式 (default: train)')
    
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
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop'], 
                        help='optimizer (default: sgd)')
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'nll', 'focal'], 
                        help='loss function (default: cross_entropy)')
    
    # 其他参数
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='checkpoint file name')
    
    return parser.parse_args()

def visualize_model(model, test_loader, device, num_images=25):
    """
    可视化模型预测结果
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备(CPU/GPU)
        num_images: 可视化图像数量
    """
    classes = get_cifar10_classes()
    model.eval()
    
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 预测
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    # 绘制图像
    plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(images))):
        plt.subplot(5, 5, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        # 反归一化
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f'Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}',
                 color=('green' if preds[i] == labels[i] else 'red'))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 检查CUDA可用性
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 创建TensorBoard日志目录
    log_dir = os.path.join('runs', f"{args.optimizer}_{args.activation}_loss{args.loss}_lr{args.lr}_bs{args.batch_size}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 加载数据
    print("==> 准备数据...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment
    )
    
    # 创建模型
    print("==> 创建模型...")
    
    # 根据损失函数选择输出层激活函数
    output_activation = None
    if args.loss == 'nll':
        output_activation = 'log_softmax'
    
    model = vgg16(
        num_classes=10,
        init_weights=args.init_weights,
        activation=args.activation,
        kernel_size=args.kernel_size,
        feature_maps=args.feature_maps,
        fc_layers=args.fc_layers,
        neurons=args.neurons,
        output_activation=output_activation
    ).to(device)
    
    # 定义损失函数
    if args.loss == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'nll':
        # 负对数似然损失，需要在模型最后一层添加LogSoftmax
        criterion = torch.nn.NLLLoss()
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
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模型
        # 定义学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
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
        print(f'TensorBoard日志保存在: {log_dir}')
        writer.close()
    
    elif args.mode == 'test':
        # 加载检查点
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        if os.path.isfile(checkpoint_path):
            print(f"==> 加载检查点 '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            print(f"==> 加载成功 (epoch {checkpoint['epoch']})")
        else:
            print(f"==> 未找到检查点 '{checkpoint_path}'")
            return
        
        # 测试模型
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'测试准确率: {test_acc:.3f}%')
    
    elif args.mode == 'visualize':
        # 加载检查点
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        if os.path.isfile(checkpoint_path):
            print(f"==> 加载检查点 '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            print(f"==> 加载成功 (epoch {checkpoint['epoch']})")
        else:
            print(f"==> 未找到检查点 '{checkpoint_path}'")
            return
        
        # 可视化模型预测
        visualize_model(model, test_loader, device)

if __name__ == '__main__':
    main()
