import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# 项目根目录（保证从任何位置运行都能找到文件）
BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / 'imagenet_labels.json'
MODEL_PATH = BASE_DIR / 'models' / 'resnet18-f37072fd.pth'


def load_model():
    """
    加载 ResNet18 预训练模型。
    优先读取项目 models/ 目录里的本地权重，完全不依赖网络；
    本地没有时才回退到 torchvision 自动下载（仅第一次会联网）。
    """
    if MODEL_PATH.exists():
        model = models.resnet18(weights=None)
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state)
    else:
        model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    return model


def get_transform():
    """图片预处理：缩放 224x224 -> 张量 -> 归一化"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_labels(path=LABELS_PATH):
    """加载 ImageNet 分类标签"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(model, transform, labels, img, top_k=3):
    """
    识别一张图片，返回置信度最高的 top_k 个结果。
    img 为 PIL.Image 对象。
    返回格式：[{'label': '...', 'confidence': 87.65}, ...]
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        results.append({'label': labels[idx], 'confidence': round(prob, 2)})
    return results


if __name__ == '__main__':
    # 命令行模式：识别本地图片 test.jpg（保留原来的用法）
    print("正在加载AI模型...")
    model = load_model()
    transform = get_transform()
    labels = load_labels()
    print(f"已加载 {len(labels)} 个分类标签，准备识别：\n")

    img_path = BASE_DIR / 'test.jpg'
    print(f"正在分析图片：{img_path}")
    img = Image.open(img_path).convert('RGB')

    results = predict(model, transform, labels, img)

    print("-" * 30)
    for i, r in enumerate(results):
        print(f"Top{i+1}: {r['label']}，置信度：{r['confidence']:.2f}%")
    print("-" * 30)
    print("识别完成！")
