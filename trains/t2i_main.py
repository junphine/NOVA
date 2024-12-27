import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import DiffusionPipeline, UNet2DModel
from datasets import load_dataset
from PIL import Image
import os
import torchvision

# 假设你已经有一个图像-文本对数据集
# 这里我们使用一个虚拟的数据集加载器作为示例
class TextImageDataset(Dataset):
    def __init__(self, texts, images, transform=None):
        self.texts = texts
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return text, image

# 加载数据集
# texts 和 images 应该分别包含文本和图像路径的列表
texts = [...]  # 你的文本列表
images = [...]  # 你的图像路径列表

# 数据转换（例如，调整图像大小）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])

dataset = TextImageDataset(texts, images, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 初始化模型
# 这里我们使用一个预训练的 Diffusion 模型和一个预训练的文本编码器（CLIP）
model_id = "CompVis/stable-diffusion-v1-4"
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
feature_extractor = CLIPVitFeatureExtractor.from_pretrained(model_id)
unet = UNet2DModel.from_pretrained(model_id)
scheduler = Scheduler(num_train_timesteps=1000, num_inference_timesteps=50)

# 封装成 DiffusionPipeline
pipeline = DiffusionPipeline(
    unet,
    scheduler,
    feature_extractor,
    text_encoder=tokenizer,  # 假设我们直接用CLIP的文本编码器作为条件输入
).to("cuda")

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(pipeline.parameters(), lr=1e-4)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    pipeline.train()
    running_loss = 0.0
    for i, (texts, images) in enumerate(dataloader):
        texts = [tokenizer(text, truncation=True, padding="max_length", max_length=77, return_tensors="pt").input_ids.squeeze() for text in texts]
        texts = torch.stack(texts).to("cuda")
        images = images.to("cuda")

        # 前向传播
        with torch.no_grad():
            noise = torch.randn_like(images).to("cuda")
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device="cuda").long()

        output_images = pipeline(texts, noise=noise, timesteps=timesteps).images

        # 计算损失
        loss = criterion(output_images, images)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# 保存模型
pipeline.save_pretrained("./my-diffusion-model")
tokenizer.save_pretrained("./my-diffusion-model")
feature_extractor.save_pretrained("./my-diffusion-model")