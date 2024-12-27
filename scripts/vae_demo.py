import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
DATA_ROOT = r'C:/Code/Notebook/nlp/text-datasets/'
# 设置超参数
batch_size = 128
epochs = 10
learning_rate = 1e-3
latent_dim = 20  # 潜在空间的维度

# 数据预处理（使用MNIST数据集）
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义编码器（Encoder）
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 潜在变量的均值
        self.fc22 = nn.Linear(400, latent_dim)  # 潜在变量的标准差

    def forward(self, x):
        h1 = torch.relu(self.fc1(x.view(-1, 28*28)))
        z_mean = self.fc21(h1)
        z_log_var = self.fc22(h1)
        return z_mean, z_log_var

# 定义解码器（Decoder）
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h3))
        return reconstruction

# 定义VAE（包括编码器、解码器及重参数化）
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

# 定义损失函数
def loss_function(reconstruction, x, z_mean, z_log_var):
    BCE = nn.functional.binary_cross_entropy(reconstruction, x.view(-1, 28*28), reduction='sum')
    # KL散度
    # p(z) ~ N(0, I), q(z|x) ~ N(mu, sigma^2)
    # D_KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 其中，mu和sigma是从编码器获得的
    # z_mean是mu，z_log_var是log(sigma^2)
    # 因此，KL散度是：
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    return BCE + KL

# 初始化模型和优化器
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = model(data)
        loss = loss_function(reconstruction, data, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}")

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

# 生成图像
def generate_images(epoch, num_images=10):
    model.eval()
    with torch.no_grad():
        # 随机生成潜在变量
        z = torch.randn(num_images, latent_dim).to(device)
        sample = model.decoder(z).cpu()
        sample = sample.view(num_images, 28, 28)

        # 显示生成的图像
        fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
        for i in range(num_images):
            axes[i].imshow(sample[i], cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练VAE
for epoch in range(1, epochs + 1):
    train(epoch)
    generate_images(epoch)
