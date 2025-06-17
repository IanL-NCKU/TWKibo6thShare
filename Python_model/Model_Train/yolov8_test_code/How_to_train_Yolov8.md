# YOLOv8 訓練完整教學指南

本教學將詳細介紹如何使用 YOLOv8 進行物件偵測模型訓練，包含環境設置、檔案配置以及所有參數的詳細說明。

## 目錄
1. [環境設置](#環境設置)
2. [檔案結構與準備](#檔案結構與準備)
3. [數據集配置](#數據集配置)
4. [訓練配置參數詳解](#訓練配置參數詳解)
5. [執行訓練](#執行訓練)
6. [常見問題與優化建議](#常見問題與優化建議)

---

## 環境設置

### 1. 安裝 Anaconda
如果您沒有 Python 環境，請至 [Anaconda 官網](https://www.anaconda.com/download) 下載並安裝。

### 2. 創建 Python 環境
建議使用 Python 3.9 版本：

```bash
# 開啟 Anaconda Prompt
conda create -n yolo_env python=3.9
conda activate yolo_env
```

### 3. 安裝依賴套件

```bash
# 安裝 PyTorch（支援 CUDA）
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 安裝 Ultralytics YOLO
pip install ultralytics

# 如果有 requirements.txt，執行以下命令
# ⚠️ 重要：一定要先 cd 到 requirements.txt 所在的目錄
cd \TWKibo6thShare\Python_model  # 切換到 requirements.txt 所在目錄
pip install -r requirements.txt
```

---

## 檔案結構與準備

### 基本檔案結構
```
project_folder/
├── yolov8n.pt                    # 預訓練模型
├── kiborpc_yolo_train.py         # 訓練腳本
├── cfg.yaml                      # 訓練配置檔案
├── kiborpcdata.yaml             # 數據集配置檔案
└── dataset/
    ├── train/                   # 訓練數據
    ├── val/                     # 驗證數據
    └── test/                    # 測試數據
```

### 重要提醒
⚠️ **模型與尺寸限制**
- **一定要使用 YOLOv8n 進行訓練**（建議使用 320×320 尺寸）
- **所有提供的權重檔案都是 320×320 尺寸進行訓練**
- 使用時請注意輸入尺寸必須保持一致

---

## 數據集配置

### kiborpcdata.yaml 檔案說明

```yaml
# 數據集路徑配置
train: E:\Ian\kiborpc\Kibodataset_aug\train    # 訓練集路徑
val: E:\Ian\kiborpc\Kibodataset_aug\val        # 驗證集路徑  
test: E:\Ian\kiborpc\Kibodataset_aug\test      # 測試集路徑

# 類別定義
names:
    0: coin          # 硬幣
    1: compass       # 指南針
    2: coral         # 珊瑚
    3: crystal       # 水晶
    4: diamond       # 鑽石
    5: emerald       # 綠寶石
    6: fossil        # 化石
    7: key           # 鑰匙
    8: letter        # 信件
    9: shell         # 貝殼
    10: treasure_box # 寶箱
```

**修改方式：**
- 將路徑改為您的數據集實際路徑
- 根據您的項目修改類別名稱和數量
- 確保路徑使用正確的分隔符號（Windows 使用 `\` 或 `/`）

---

## 訓練配置參數詳解

### 基本訓練設置

| 參數 | 預設值 | 說明 | 建議調整範圍 |
|------|--------|------|-------------|
| `epochs` | 200 | 訓練輪數 | 100-500（視數據集大小） |
| `batch` | 32 | 批次大小 | 8-64（視 GPU 記憶體） |
| `imgsz` | 320 | 輸入圖像尺寸 | **建議固定使用 320**（與權重檔案一致） |
| `device` | 0 | 使用的 GPU 編號 | 0, 1, 2...（多GPU時） |
| `cache` | 'disk' | 快取模式 | 'disk', 'ram', false |
| `workers` | 8 | 數據載入工作執行緒數 | 4-16（視 CPU 核心數） |

### 學習率與優化器設置

| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `optimizer` | 'SGD' | 優化器類型 | 'SGD', 'Adam', 'AdamW' |
| `cos_lr` | true | 是否使用餘弦學習率調度 | 通常保持 true |
| `lr0` | 0.01 | 初始學習率 | SGD: 0.01, Adam: 0.001 |
| `lrf` | 0.01 | 最終學習率比例 | 0.01-0.1 |
| `momentum` | 0.937 | SGD 動量參數 | 0.9-0.95 |
| `weight_decay` | 0.0005 | 權重衰減 | 0.0001-0.001 |
| `warmup_epochs` | 3.0 | 暖身訓練輪數 | 1-5 |

### 損失函數權重

| 參數 | 預設值 | 說明 | 調整原則 |
|------|--------|------|----------|
| `box` | 10 | 邊界框損失權重 | 小物件增加，大物件減少 |
| `cls` | 2 | 分類損失權重 | 類別不平衡時調整 |
| `dfl` | 3 | 分布焦點損失權重 | 通常不需調整 |

### 數據增強參數

#### 色彩空間增強
| 參數 | 預設值 | 說明 | 調整範圍 |
|------|--------|------|----------|
| `hsv_h` | 0.015 | 色調變化幅度 | 0-0.1 |
| `hsv_s` | 0.7 | 飽和度變化幅度 | 0-1.0 |
| `hsv_v` | 0.4 | 明度變化幅度 | 0-1.0 |

#### 幾何變換增強
| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `degrees` | 90.0 | 旋轉角度範圍 | 0-180度 |
| `translate` | 0.1 | 平移範圍比例 | 0-0.3 |
| `scale` | 0.5 | 縮放範圍 | 0.1-0.9 |
| `shear` | 0.0 | 剪切變形角度 | 0-10度 |
| `perspective` | 0.0 | 透視變換幅度 | 0-0.001 |

#### 翻轉增強
| 參數 | 預設值 | 說明 | 使用時機 |
|------|--------|------|----------|
| `flipud` | 0.5 | 上下翻轉機率 | 對稱物件適用 |
| `fliplr` | 0.5 | 左右翻轉機率 | 對稱物件適用 |

#### 混合增強技術
| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `mosaic` | 0.75 | 馬賽克增強機率 | 0.5-1.0 |
| `mixup` | 0.3 | 混合增強機率 | 0-0.5 |
| `copy_paste` | 0.3 | 複製貼上增強機率 | 0-0.5 |

### 驗證設置

| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `iou` | 0.7 | IoU 閾值 | 密集物件降低至 0.5 |
| `max_det` | 50 | 最大偵測數量 | 視場景物件密度調整 |
| `conf` | null | 信心度閾值 | 0.1-0.9 |

### 進階訓練選項

| 參數 | 預設值 | 說明 | 使用時機 |
|------|--------|------|----------|
| `patience` | 100 | 早停耐心值 | 防止過擬合 |
| `save_period` | -1 | 檢查點儲存頻率 | 長時間訓練時設為 10-50 |
| `amp` | true | 混合精度訓練 | 加速訓練，節省記憶體 |
| `freeze` | null | 凍結層數 | 微調時使用 |
| `multi_scale` | false | 多尺度訓練 | 提升多尺度偵測能力 |

---

## 執行訓練

### 1. 準備訓練腳本

⚠️ **重要：必須使用 YOLOv8n 模型進行訓練**

```python
from ultralytics import YOLO

if __name__ == "__main__":
    print("開始 YOLOv8 訓練")
    
    # 檔案路徑設定
    yolo_path = r'path/to/yolov8n.pt'           # ⚠️ 必須使用 yolov8n.pt
    datayaml_path = r'path/to/kiborpcdata.yaml' # 數據集配置路徑
    cfgyaml_path = r'path/to/cfg.yaml'          # 訓練配置路徑

    # 載入模型
    model = YOLO(yolo_path)  # 載入 YOLOv8n 預訓練模型

    # 開始訓練
    results = model.train(
        data=datayaml_path,
        cfg=cfgyaml_path,
    )
```

### 2. 執行訓練

```bash
# 啟動環境
conda activate yolo_env

# 切換到專案目錄
cd path/to/your/project

# 執行訓練
python kiborpc_yolo_train.py
```

### 3. 監控訓練過程

訓練過程中會產生以下輸出：
- `runs/kibo320/krpc_aug_yolov8n_32b_cos_lr_no_rot_with03_mixup_copy_paste`
- 訓練日誌、圖表、檢查點檔案

---

## 常見問題與優化建議

### 記憶體不足
- 減少 `batch` 大小
- 降低 `imgsz` 尺寸
- 設定 `cache: false`

### 訓練過慢
- 增加 `batch` 大小（在記憶體允許範圍內）
- 啟用 `amp: true`
- 增加 `workers` 數量

### 過擬合問題
- 增加數據增強強度
- 減少 `epochs`
- 設定 `patience` 進行早停

### 欠擬合問題
- 減少數據增強強度
- 增加 `epochs`
- 調整學習率 `lr0`

### 小物件偵測優化
- 使用較大的 `imgsz`
- 增加 `box` 損失權重
- 降低 `iou` 閾值

### 密集物件場景優化
- 降低 `iou` 閾值到 0.5
- 增加 `max_det` 數量
- 增加 `mosaic` 機率

### 類別不平衡處理
- 調整 `cls` 權重
- 使用加權採樣
- 增加少數類別的數據增強

---

## 總結

成功的 YOLO 訓練需要：

1. **正確的環境設置**：確保所有依賴套件正確安裝
2. **適當的數據準備**：高品質的標註數據和合理的數據分割
3. **參數調優**：根據具體任務調整超參數
4. **持續監控**：觀察訓練指標，及時調整策略

建議新手從預設參數開始，逐步根據訓練結果調整參數。記住，沒有一套參數適用於所有場景，需要根據具體數據集和任務需求進行客製化調整。