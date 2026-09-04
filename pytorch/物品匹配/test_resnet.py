import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision.transforms import v2


def build_resnet50():
    # 使用 ImageNet 预训练权重
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # 去掉最后的分类层，只保留特征提取部分
    model.fc = nn.Identity()

    model.eval()

    return model, weights


def extract_embedding(model, weights, image_path, device):
    image = Image.open(image_path).convert("RGB")

    transform = weights.transforms()
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(image)

    # L2 Normalize
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding

def cosine_similarity(embedding_a, embedding_b):
    # 计算余弦相似度
    return torch.sum(
        embedding_a * embedding_b,
        dim=1
    )

def main():
    device = torch.device(
        "mps" if torch.mps.is_available() else "cpu"
    )

    from pathlib import Path

    current_dir = Path.cwd()
    print(current_dir)
    print("Device:", device)

    model, weights = build_resnet50()
    model = model.to(device)

    # TODO: 改成你的一张饮品图片
    image_path = "dataset/coke/img.png"
    image_path2 = "dataset/red_tea/img.png"

    embedding1 = extract_embedding(
        model,
        weights,
        image_path,
        device
    )
    embedding2 = extract_embedding(
        model,
        weights,
        image_path2,
        device
    )

    print("Embedding shape:", embedding1.shape)
    print("Embedding norm:", torch.norm(embedding1, p=2, dim=1))

    similarity = cosine_similarity(embedding1, embedding2)
    print("Similarity:", similarity)

if __name__ == "__main__":
    main()