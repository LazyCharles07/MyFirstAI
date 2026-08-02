# 图片识别网站

基于 ResNet18（ImageNet 1000 类）的图片识别网站。
模型通过 ONNX Runtime Web 在**浏览器本地**运行，图片不会上传到任何服务器。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `index.html` | 页面 |
| `app.js` | 识别逻辑（加载模型、预处理、推理、展示结果） |
| `imagenet_labels.json` | 1000 个分类标签 |
| `model/resnet18.onnx` | ResNet18 模型（约 45MB） |
| `onnx/` | onnxruntime-web 浏览器端运行时 |

## 本地预览

在 `web` 目录下运行：

```bash
python -m http.server 8000
```

然后浏览器访问 http://127.0.0.1:8000

## 部署到 GitHub Pages

1. 在 GitHub 新建一个仓库；
2. 把本目录下**所有文件**上传到仓库根目录（`index.html` 必须在根目录）；
3. 仓库 Settings → Pages → Source 选择 “Deploy from a branch” → 分支 `main` → 目录 `/ (root)` → Save；
4. 等 1~2 分钟，访问 `https://<你的用户名>.github.io/<仓库名>/`。

首次访问需要下载约 45MB 的模型，页面上有进度提示；之后浏览器会自动缓存。
