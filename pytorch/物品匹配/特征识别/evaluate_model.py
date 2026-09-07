import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def build_model(device):
    model = models.resnet50(weights=None)

    checkpoint = torch.load(
        "checkpoints/resnet50_best.pth",
        map_location=device
    )
    num_classes = checkpoint["num_classes"]
    model.fc = nn.Linear(
        2048,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(
            256
        ),
        transforms.CenterCrop(
            224
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])
    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }
    return model, transform,idx_to_class

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def predict(image_paths, model, transform, idx_to_class, device):
    """
    对一批没有标签的图片进行预测。

    参数：
        image_paths: 图片路径列表
        model: 已加载好的模型
        transform: 推理阶段的 transform
        idx_to_class: 类别索引 -> 类别名称
        device: cpu / cuda

    返回：
        results: 每张图片的预测结果
    """

    model.eval()

    results = []

    # =========================
    # 1. 读取并预处理所有图片
    # =========================

    tensors = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        tensor = transform(image)

        tensors.append(tensor)

    # =========================
    # 2. 拼成 Batch
    # =========================

    batch = torch.stack(tensors)

    batch = batch.to(device)

    # =========================
    # 3. 批量推理
    # =========================

    with torch.no_grad():

        outputs = model(batch)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

    # =========================
    # 4. 整理结果
    # =========================

    for image_path, prediction, confidence in zip(
            image_paths,
            predictions,
            confidences
    ):
        prediction = prediction.item()
        confidence = confidence.item()

        results.append({
            "path": image_path,
            "class": idx_to_class[prediction],
            "confidence": confidence
        })

    return results



def main():
    device = get_device()
    # 加载模型
    model, transform,idx_to_class = build_model(device)

    image_paths = [
        "test_images/东方树叶/img.png",
        "test_images/东方树叶/img_1.png",
        "test_images/东方树叶/img_2.png",
        "test_images/东方树叶/img_3.png",
        "test_images/无糖可乐/img.png",
        "test_images/无糖可乐/img_1.png",
        "test_images/无糖可乐/img_2.png",
        "test_images/无糖可乐/img_3.png",
    ]
    results = predict(image_paths, model, transform, idx_to_class, device)
    for result in results:
        print(result)

    # 进行推理
    # 输出结果

if __name__ == '__main__':
    main()