电力巡检无人机或摄像头对输电线路异物识别，提升电网安全性 深度学习YOLOV8模型训练输电线异物检测数据集 电塔异物数据集的识别检测鸟巢气球垃圾风筝等进行检测识别
@[toc]
以下文字及代码仅供参考学习使用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43250e4046c94c6190839cd9dde6e6e7.jpeg)


### **电塔电线覆盖物检测YOLO数据集**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 电塔电线覆盖物检测YOLO数据集 |
| **图像总数** | 1,386 张 |
| **标注格式** | YOLO `.txt` 格式（可用于YOLO系列模型训练） |
| **数据划分比例** | 训练集 : 验证集 = 8 : 2 |
| **训练集数量** | 1,108 张 |
| **验证集数量** | 278 张 |
| **类别数量** | 5 类 |
| **类别名称** | `kite`（风筝）、`trash`（垃圾）、`balloon`（气球）、`nest`（鸟巢）、`cloth`（布条） |
| **是否增强** | 未提及（默认为真实图像） |

---

### **各类别详细统计表**

| 类别（英文） | 类别（中文） | 图片数量 | 标注框数量 | 备注 |
|--------------|--------------|----------|------------|------|
| kite | 风筝 | 117 | 163 | 单图平均约1.4个目标 |
| trash | 垃圾 | 147 | 673 | 单图可能含多个小目标 |
| balloon | 气球 | 130 | 159 | 单图平均约1.2个目标 |
| nest | 鸟巢 | 1,059 | 1,064 | 出现频率最高，是主要类别 |
| cloth | 布条 | 14 | 14 | 样本极少，可能存在类别不平衡 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a91303dcae24470cb302858834edbaf2.jpeg)

### **数据集划分表**

| 数据集类型 | 图像数量 | 占比 |
|------------|----------|------|
| **训练集** | 1,108 张 | 80% |
| **验证集** | 278 张 | 20% |
| **总计** | 1,386 张 | 100% |

---


--- 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e8dbe252df74160a6955574583ae77c.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c294af13f0d247cabb7af90642a7cef2.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ee4d94986786469a82c6ec5846e341ab.png)

从 **环境搭建 → 数据准备 → 模型训练 → 推理评估** 的 **完整 YOLOv8 实战流程**。

---

## ✅ 一、环境搭建（CUDA + Anaconda + Python + 依赖）

### 1. 确认 CUDA 驱动（GPU 加速）

打开终端：
```bash
nvidia-smi
```
查看是否支持 CUDA（推荐 11.8 或 12.1）。

> ✅ 若无 GPU，也可使用 CPU，但训练较慢。

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装后打开 **Anaconda Prompt**（Windows）或终端（Mac/Linux）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建环境
conda create -n power-tower python=3.9

# 激活环境
conda activate power-tower
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics YOLOv8
pip install ultralytics

# 安装 OpenCV 和其他工具
pip install opencv-python matplotlib pillow tqdm
```

✅ 验证安装：
```python
import torch
print(torch.cuda.is_available())  # 应输出 True
```

---

## ✅ 二、数据集结构（YOLO 格式）

### 文件目录结构

```
datasets/
└── power_cover/
    ├── images/
    │   ├── train/      # 1108 张
    │   └── val/        # 278 张
    ├── labels/
    │   ├── train/      # 对应 .txt 标签
    │   └── val/
    └── data.yaml
```

> ✅ 你已有划分好的数据，只需按此结构组织即可。

---

### 1. 类别说明

| class_id | 类别     | 图片数 | 标注框数 | 说明               |
|----------|----------|--------|----------|--------------------|
| 0        | kite     | 117    | 163      | 风筝               |
| 1        | trash    | 147    | 673      | 垃圾袋、塑料       |
| 2        | balloon  | 130    | 159      | 气球               |
| 3        | nest     | 1059   | 1064     | 鸟巢               |
| 4        | cloth    | 14     | 14       | 布条（极少数类）   |

> ⚠️ **严重类别不平衡**：`cloth` 仅 14 张，需特殊处理。

---

### 2. 创建 `data.yaml`

在 `datasets/power_cover/` 下创建 `data.yaml`：

```yaml
# data.yaml
train: ../datasets/power_cover/images/train
val: ../datasets/power_cover/images/val

nc: 5
names: ['kite', 'trash', 'balloon', 'nest', 'cloth']
```

---

## ✅ 三、训练代码（YOLOv8 官方预训练模型）

### 📌 `train.py` —— 针对小样本与不平衡优化

```python
# train.py
from ultralytics import YOLO
import torch

# 检查 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 推荐使用 yolov8m（小目标更强）或 yolov8l
model = YOLO('yolov8m.pt')  # 可选: yolov8l.pt（显存足够）

# 开始训练
results = model.train(
    data='datasets/power_cover/data.yaml',  # 数据配置文件
    epochs=300,                             # 小样本需多轮训练
    imgsz=640,                              # 可选 640 或 1024（提升小目标检测）
    batch=16,                               # 批次大小（根据显存调整）
    name='power_cover_yolov8m',             # 实验名称
    device=device,                          # 使用 GPU
    workers=4,                              # 数据加载线程
    optimizer='AdamW',                      # 收敛更稳
    lr0=0.001,
    weight_decay=0.0005,
    patience=40,                            # 早停轮数
    project='runs/power',                   # 保存路径
    amp=True,                               # 自动混合精度
    close_mosaic=20,                        # 最后20轮关闭 Mosaic
    # --- 数据增强（关键！）---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.6,                              # MixUp 增强稀有类
    copy_paste=0.4,                         # Copy-Paste 增强 cloth/kite
    degrees=15.0,                           # 旋转增强（异物方向多变）
)
```

📌 **关键策略说明**：
- `epochs=300`：小样本需多轮训练
- `mixup=0.6`, `copy_paste=0.4`：增强稀有类（`cloth`, `kite`）泛化能力
- `yolov8m/l`：适合小目标（如布条、气球）
- `imgsz=640` 或 `1024`：若显存允许，建议 `1024` 提升小目标召回

---

## ✅ 四、推理代码（检测新图像）

### 📌 `predict.py`

```python
# predict.py
from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO('runs/power/power_cover_yolov8m/weights/best.pt')

# 推理单张图像
results = model.predict(
    source='power_line.jpg',                # 输入图像路径
    save=True,                              # 保存带框图像
    conf=0.25,                              # 置信度阈值（稀有类可略低）
    iou=0.45,
    show=False,
    project='runs/predict',
    name='power_test',
    imgsz=640
)

# 打印结果
for result in results:
    boxes = result.boxes
    print(f"共检测到 {len(boxes)} 个异物：")
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        cls_name = model.names[cls_id]
        print(f"  {cls_name}: {conf:.3f}")
```

---

## ✅ 五、评估模型性能（在验证集上）

### 📌 `evaluate.py`

```python
# evaluate.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/power/power_cover_yolov8m/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='datasets/power_cover/data.yaml',
    split='val',
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/eval',
    name='power_val'
)

# 输出关键指标
print(f"mAP@0.5:  {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall:    {metrics.box.r:.4f}")

# 按类别输出（重点关注 cloth、kite）
print("\n=== 按类别性能 ===")
for i, name in enumerate(model.names):
    print(f"{name}: P={metrics.box.p[i]:.4f}, R={metrics.box.r[i]:.4f}, mAP={metrics.box.ap[i]:.4f}")
```

---

## ✅ 六、类别不平衡应对策略

| 问题 | 建议 |
|------|------|
| `cloth` 仅 14 张 | 启用 `copy_paste` 增强 |
| `nest` 占比过高 | 检查是否主导 loss，可考虑 `class weights`（需自定义） |
| 小目标（kite, cloth） | 使用 `yolov8m/l` + `imgsz=1024` |
| 数据总量中等 | 多轮训练 + 强增强（mixup, mosaic） |

> 🔔 提示：若 `cloth` 检出率低，可手动复制 `cloth` 图像并轻微变换后加入训练集（数据增补）。

---

## ✅ 七、训练结果可视化

训练完成后，查看：
- `runs/power/power_cover_yolov8m/results.png`：训练曲线
- `confusion_matrix.png`：是否误检 `trash` 为 `cloth`
- `PR_curve.png`：`cloth` 类别 PR 曲线是否偏低

---


---

### 🔔 温馨提示

- 你的 `val` 集合理（278张），可反映真实性能
- 关注 `Recall` 指标，避免漏检 `cloth` 或 `kite`
- 若未来部署到无人机，可导出为 ONNX 或 TensorRT



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a4d76c8eecb41c0bcbe5881119fe164.png)

