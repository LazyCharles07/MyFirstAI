// ---------- 配置 ----------
var MODEL_URL = 'model/resnet18.onnx';
var LABELS_URL = 'imagenet_labels.json';
var WASM_PATH = 'onnx/';
var INPUT_SIZE = 224;

// 浏览器端单线程运行（GitHub Pages 不支持线程所需的响应头）
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL(WASM_PATH, document.baseURI).href;

var session = null;
var labels = null;
var modelLoadPromise = null;
var modelReady = false;
var previewURL = null;

// ---------- DOM ----------
var fileInput = document.getElementById('file');
var preview = document.getElementById('preview');
var placeholder = document.getElementById('placeholder');
var btn = document.getElementById('btn');
var statusBox = document.getElementById('status');
var errorBox = document.getElementById('error');
var resultSection = document.getElementById('resultSection');
var resultImg = document.getElementById('resultImg');
var resultTime = document.getElementById('resultTime');
var resultList = document.getElementById('resultList');
var againBtn = document.getElementById('againBtn');

// ---------- 工具函数 ----------
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

async function loadLabels() {
  var resp = await fetch(LABELS_URL);
  labels = await resp.json();
}

async function loadModel(onProgress) {
  if (session) return session;
  if (!modelLoadPromise) {
    modelLoadPromise = (async function () {
      var resp = await fetch(MODEL_URL);
      if (!resp.ok) throw new Error('模型下载失败：HTTP ' + resp.status);
      var total = Number(resp.headers.get('Content-Length')) || 0;
      var reader = resp.body.getReader();
      var chunks = [];
      var received = 0;
      while (true) {
        var part = await reader.read();
        if (part.done) break;
        chunks.push(part.value);
        received += part.value.length;
        if (onProgress) onProgress(received, total);
      }
      statusBox.textContent = '模型下载完成，正在初始化…';
      var buffer = new Uint8Array(received);
      var offset = 0;
      for (var i = 0; i < chunks.length; i++) {
        buffer.set(chunks[i], offset);
        offset += chunks[i].length;
      }
      session = await ort.InferenceSession.create(buffer.buffer, { executionProviders: ['wasm'] });
      modelReady = true;
      statusBox.textContent = '模型已就绪，可以开始识别';
      return session;
    })().catch(function (err) {
      modelLoadPromise = null;
      modelReady = false;
      throw err;
    });
  }
  return modelLoadPromise;
}

function updateLoadProgress(received, total) {
  if (!total) return;
  var pct = Math.min(100, Math.round(received / total * 100));
  var mb = (received / 1048576).toFixed(1);
  var totalMB = (total / 1048576).toFixed(1);
  statusBox.innerHTML =
    '正在加载模型 ' + mb + ' / ' + totalMB + ' MB（首次加载约 13MB，之后有缓存）' +
    '<div class="bar-bg"><div class="bar" style="width:' + pct + '%"></div></div>';
}

// ---------- 图片预处理 ----------
async function fileToTensor(file) {
  var bitmap = await createImageBitmap(file); // 自动处理拍照方向
  var canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  var ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  bitmap.close();

  var imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  var data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  var mean = [0.485, 0.456, 0.406];
  var std = [0.229, 0.224, 0.225];
  var area = INPUT_SIZE * INPUT_SIZE;
  var p = 0;
  for (var i = 0; i < imageData.length; i += 4) {
    data[p] = (imageData[i] / 255 - mean[0]) / std[0];
    data[p + area] = (imageData[i + 1] / 255 - mean[1]) / std[1];
    data[p + 2 * area] = (imageData[i + 2] / 255 - mean[2]) / std[2];
    p++;
  }
  return new ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ---------- 识别 ----------
async function recognize(file) {
  var sess = await loadModel(updateLoadProgress);
  var tensor = await fileToTensor(file);
  var t0 = performance.now();
  var outputs = await sess.run({ input: tensor });
  var elapsed = (performance.now() - t0) / 1000;
  var logits = outputs.output.data;

  // softmax
  var max = -Infinity;
  for (var i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  var exps = new Float64Array(logits.length);
  var sum = 0;
  for (var j = 0; j < logits.length; j++) {
    exps[j] = Math.exp(logits[j] - max);
    sum += exps[j];
  }
  var order = [];
  for (var k = 0; k < logits.length; k++) order.push(k);
  order.sort(function (a, b) { return exps[b] - exps[a]; });

  return {
    elapsed: elapsed,
    top: order.slice(0, 3).map(function (idx) {
      return { label: labels[idx], confidence: exps[idx] / sum * 100 };
    })
  };
}

function showResults(result, imageURL) {
  resultImg.src = imageURL;
  resultTime.textContent = '识别耗时 ' + result.elapsed.toFixed(2) + ' 秒';
  resultList.innerHTML = '';
  result.top.forEach(function (item, index) {
    var row = document.createElement('div');
    row.className = 'result' + (index === 0 ? ' top1' : '');
    row.innerHTML =
      '<div class="rank r' + (index + 1) + '">' + (index + 1) + '</div>' +
      '<div class="info">' +
      '  <div class="label"></div>' +
      '  <div class="bar-bg"><div class="bar" style="width:' + item.confidence.toFixed(1) + '%"></div></div>' +
      '</div>' +
      '<div class="pct">' + item.confidence.toFixed(2) + '%</div>';
    row.querySelector('.label').textContent = item.label;
    resultList.appendChild(row);
  });
  resultSection.style.display = 'block';
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- 事件 ----------
fileInput.addEventListener('change', function () {
  var file = fileInput.files[0];
  if (!file) return;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  preview.src = previewURL;
  preview.hidden = false;
  placeholder.hidden = true;
  hideError();
  btn.disabled = false; // 选好图就能点，模型没加载完会自动等
});

btn.addEventListener('click', async function () {
  var file = fileInput.files[0];
  if (!file) {
    showError('请先选择一张图片');
    return;
  }
  hideError();
  btn.disabled = true;
  try {
    btn.textContent = modelReady ? '识别中，请稍候…' : '正在加载模型，请稍候…';
    var result = await recognize(file);
    showResults(result, previewURL);
    btn.textContent = '识别完成';
  } catch (err) {
    showError('识别失败：' + err.message);
    btn.textContent = '识别';
    btn.disabled = false;
  }
});

againBtn.addEventListener('click', function () {
  resultSection.style.display = 'none';
  btn.textContent = '识别';
  btn.disabled = true;
  fileInput.value = '';
  preview.hidden = true;
  placeholder.hidden = false;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
});

// ---------- 启动：后台加载标签和模型（不阻塞选图） ----------
loadLabels();
loadModel(updateLoadProgress).catch(function () {
  statusBox.textContent = '模型加载失败，选好图片后点"识别"可重试';
});
