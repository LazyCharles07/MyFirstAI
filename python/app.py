import time
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from PIL import Image

import recognizer

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 上传上限 16MB

# 启动时加载一次模型（之后所有请求共用，不用每次重新加载）
print("正在加载AI模型...")
model = recognizer.load_model()
transform = recognizer.get_transform()
labels = recognizer.load_labels()
print(f"已加载 {len(labels)} 个分类标签，服务已就绪！")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def open_image(file_stream):
    """读取上传图片并提前缩小，避免大图解码拖慢识别"""
    img = Image.open(file_stream)
    # JPEG 直接以低分辨率解码，大图提速数倍
    if img.format == 'JPEG':
        img.draft('RGB', (1024, 1024))
    # 透明背景（PNG 等）贴到白底，避免黑底
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert('RGB')
    # 识别只需要 224x224，先缩到 512 以内再交给模型
    img.thumbnail((512, 512))
    return img


def save_preview(img):
    """保存一张缩小后的展示图（最长边 800px），结果页加载更快、占空间更小"""
    preview = img.copy()
    preview.thumbnail((800, 800))
    name = f"{uuid.uuid4().hex}.jpg"
    UPLOAD_DIR.mkdir(exist_ok=True)
    preview.save(UPLOAD_DIR / name, 'JPEG', quality=85, optimize=True)
    return name


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files.get('image')

    # 1. 检查是否选了图片
    if file is None or file.filename == '':
        return render_template('index.html', error='请先选择一张图片')

    # 2. 检查文件格式
    if not allowed_file(file.filename):
        return render_template('index.html', error='不支持的图片格式，请上传 PNG/JPG/JPEG/BMP/WEBP/GIF')

    # 3. 读取并识别
    try:
        img = open_image(file.stream)
    except Exception:
        return render_template('index.html', error='文件无法解析为图片，请换一张试试')

    t0 = time.perf_counter()
    results = recognizer.predict(model, transform, labels, img, top_k=3)
    elapsed = time.perf_counter() - t0

    # 4. 保存缩小版图片用于结果页展示
    saved_name = save_preview(img)

    return render_template(
        'result.html',
        results=results,
        image_url=f'/uploads/{saved_name}',
        original_name=file.filename,
        elapsed=elapsed,
    )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', error='图片太大了，请上传 16MB 以内的图片'), 413


if __name__ == '__main__':
    # threaded=True：识别请求不会阻塞页面的其他访问
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
