@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动本地图片识别服务器（http://127.0.0.1:5000）...
echo 模型加载约需 5 秒，浏览器会自动打开。
start "" http://127.0.0.1:5000
python python\app.py
pause
