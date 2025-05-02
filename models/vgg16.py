import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, activation='relu', 
                 kernel_size=3, feature_maps=64, fc_layers=3, neurons=4096,
                 output_activation=None):
        """
        VGG16模型实现
        
        参数:
            num_classes (int): 分类类别数
            init_weights (bool): 是否初始化权重
            activation (str): 激活函数类型 ('relu', 'leaky_relu', 'elu')
            kernel_size (int): 卷积核大小
            feature_maps (int): 初始特征图数量
            fc_layers (int): 全连接层数量
            neurons (int): 全连接层神经元数量
            output_activation (str): 输出层激活函数 (None, 'softmax', 'log_softmax', 'sigmoid')
        """
        super(VGG16, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 特征提取部分
        self.features = self._make_layers(kernel_size, feature_maps)
        
        # 分类器部分
        classifier_layers = []
        
        # 添加全连接层
        if fc_layers >= 1:
            classifier_layers.extend([
                nn.Linear(512 * 1 * 1, neurons),
                self.activation,
                nn.Dropout(0.5)
            ])
        
        # 添加额外的全连接层
        for _ in range(fc_layers - 2):
            if _ >= 0:  # 确保fc_layers >= 2
                classifier_layers.extend([
                    nn.Linear(neurons, neurons),
                    self.activation,
                    nn.Dropout(0.5)
                ])
        
        # 添加最后一个全连接层
        if fc_layers >= 2:
            classifier_layers.append(nn.Linear(neurons, num_classes))
        else:
            classifier_layers.append(nn.Linear(512 * 1 * 1, num_classes))
            
        # 添加输出层激活函数
        if output_activation == 'softmax':
            classifier_layers.append(nn.Softmax(dim=1))
        elif output_activation == 'log_softmax':
            classifier_layers.append(nn.LogSoftmax(dim=1))
        elif output_activation == 'sigmoid':
            classifier_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 初始化权重
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, kernel_size, feature_maps):
        cfg = [feature_maps, feature_maps, 'M', 
               feature_maps*2, feature_maps*2, 'M', 
               feature_maps*4, feature_maps*4, feature_maps*4, 'M', 
               feature_maps*8, feature_maps*8, feature_maps*8, 'M', 
               feature_maps*8, feature_maps*8, feature_maps*8, 'M']
        
        layers = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=kernel_size//2)
                layers += [conv2d, nn.BatchNorm2d(v), self.activation]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg16(pretrained=False, **kwargs):
    """
    构建VGG16模型
    
    参数:
        pretrained (bool): 是否使用预训练权重
        **kwargs: 其他参数
    """
    model = VGG16(**kwargs)
    if pretrained:
        # 这里可以加载预训练权重
        pass
    return model
