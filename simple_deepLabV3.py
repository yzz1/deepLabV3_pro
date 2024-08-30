import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, mean_iou

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = plt.imread(self.image_paths[index])
        label = plt.imread(self.label_paths[index])
        return image, label

# 加载预训练的 DeepLabV3+模型
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

# 定义改进后的模型
class ImprovedDeepLabV3Plus(torch.nn.Module):
    def __init__(self):
        super(ImprovedDeepLabV3Plus, self).__init__()
        # 加载原始的 DeepLabV3+模型
        self.base_model = model
        # 引入注意力机制模块
        self.attention_module = AttentionModule()
        # 定义多尺度特征融合模块
        self.multi_scale_fusion_module = MultiScaleFusionModule()

    def forward(self, x):
        # 获取原始模型的输出
        base_output = self.base_model(x)['out']
        # 应用注意力机制
        attended_output = self.attention_module(base_output)
        # 进行多尺度特征融合
        fused_output = self.multi_scale_fusion_module(attended_output)
        return fused_output

# 注意力机制模块
class AttentionModule(torch.nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        # 可以使用通道注意力或空间注意力等机制
        # 这里以通道注意力为例
        self.channel_attention = ChannelAttention()

    def forward(self, x):
        # 计算通道注意力权重
        attention_weights = self.channel_attention(x)
        # 将注意力权重应用到输入特征图上
        return x * attention_weights

# 通道注意力模块
class ChannelAttention(torch.nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

# 多尺度特征融合模块
class MultiScaleFusionModule(torch.nn.Module):
    def __init__(self):
        super(MultiScaleFusionModule, self).__init__()
        # 可以使用不同的融合策略，如加权融合等
        # 这里以简单的拼接融合为例
        self.conv = torch.nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1)

    def forward(self, x):
        # 假设输入特征图有多个尺度
        scale1, scale2, scale3 = x
        fused = torch.cat([scale1, scale2, scale3], dim=1)
        return self.conv(fused)

# 数据增强函数
def data_augmentation(image, label):
    # 可以添加随机裁剪、旋转、翻转等数据增强操作
    # 这里以随机水平翻转为例
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)
    return image, label

# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted_labels = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels.flatten())
            ground_truths.extend(labels.cpu().numpy().flatten())
    accuracy = accuracy_score(ground_truths, predictions)
    iou = mean_iou(ground_truths, predictions, labels=np.unique(ground_truths))
    return running_loss / len(test_loader), accuracy, iou

# 主函数
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    train_image_paths = ["path_to_train_images/1.jpg", "path_to_train_images/2.jpg", "..."]
    train_label_paths = ["path_to_train_labels/1.png", "path_to_train_labels/2.png", "..."]
    test_image_paths = ["path_to_test_images/1.jpg", "path_to_test_images/2.jpg", "..."]
    test_label_paths = ["path_to_test_labels/1.png", "path_to_test_labels/2.png", "..."]

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(train_image_paths, train_label_paths)
    test_dataset = CustomDataset(test_image_paths, test_label_paths)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=None)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=None)

    # 定义模型、优化器和损失函数
    model = ImprovedDeepLabV3Plus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练和测试循环
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy, iou = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, IoU: {iou:.4f}")