# 图片识别网站

基于 ResNet18（ImageNet 1000 类）的图片识别网站。
模型通过 ONNX Runtime Web 在**浏览器本地**运行，图片不会上传到任何服务器。

## 目录结构

```
仓库根目录（GitHub Pages 网站）
├── index.html              页面
├── app.js                  识别逻辑
├── imagenet_labels.json    1000 个分类标签
├── model/resnet18.onnx     ResNet18 模型（int8 量化，约 13MB）
├── onnx/                   onnxruntime-web 浏览器端运行时
└── python/                 本地 Flask 版（不参与网页部署）
    ├── app.py
    ├── recognizer.py
    ├── models/
    ├── templates/
    └── requirements.txt
```
