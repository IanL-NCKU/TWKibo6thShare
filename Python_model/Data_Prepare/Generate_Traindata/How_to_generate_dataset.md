# How to generate dataset - KIBO Training Data Generator 中文教學

## 概述

KIBO Training Data Generator 是一個用於生成物件偵測訓練資料的 Python 工具，特別設計用於生成 YOLO 格式的標籤檔案。此工具可以自動生成包含寶物（treasure items）和地標（landmark items）的合成圖像，並應用各種資料增強技術。

## 主要功能

- **自動生成合成訓練圖像**：結合寶物物件和地標物件創建多樣化的訓練場景
- **YOLO 格式標籤生成**：自動生成對應的邊界框標註
- **豐富的資料增強**：包含幾何變換、光學效果、雜訊添加等
- **兩種圖像類型**：
  - **目標圖像**（Target Images）：包含 2 個地標物件 + 1 個寶物物件
  - **遺失圖像**（Lost Images）：包含多個地標物件，可能包含寶物物件

## 系統需求

### 必要套件
```bash
# 基礎套件
pip install opencv-python        # 圖像處理
pip install pillow              # PIL 圖像庫
pip install numpy               # 數值運算
pip install matplotlib          # 圖表繪製

# 深度學習套件
pip install torch               # PyTorch 核心
pip install torchvision         # PyTorch 視覺工具

# 系統內建（通常已安裝）
# tkinter - 圖形介面選擇資料夾
```

## 檔案結構

```
project/
├── Generate_kibotrainingdata.py    # 主要執行檔案
├── Datahelp.py                     # 資料處理和增強工具
├── checkyolo.py                    # YOLO 標籤檢查工具
└── source_images/                  # 來源圖像資料夾
    ├── crystal.png                 # 寶物物件
    ├── diamond.png                 # 寶物物件
    ├── emerald.png                 # 寶物物件
    ├── landmark1.png               # 地標物件
    └── ...
```

## 快速開始教學

### 步驟 1: 準備來源圖像

將所有來源圖像（PNG 格式，包含透明通道）放置在一個資料夾中：
```
Python_model\Data_Prepare\item_template_images_nobackground
```
- **寶物物件**：檔名中包含 "crystal"、"diamond"、"emerald" 的圖像
- **地標物件**：其他所有 PNG 圖像

**重要提醒**：
- 所有圖像必須是 PNG 格式
- 必須包含 Alpha 透明通道（RGBA 格式）
- 圖像背景應為透明

### 步驟 2: 執行程式

```bash
python Generate_kibotrainingdata.py
```

程式會依序開啟兩個檔案選擇對話框：
1. **選擇輸出資料夾**：生成的資料集將儲存在此處
2. **選擇來源圖像資料夾**：包含所有 PNG 來源圖像

### 步驟 3: 檢視輸出結果

程式執行完成後，輸出資料夾將包含：

```
output_folder/
├── images/                    # 生成的訓練圖像
│   ├── 0000.png              # 合成圖像檔案
│   ├── 0001.png
│   └── ...
├── labels/                    # YOLO 格式標籤檔案
│   ├── 0000.txt              # 對應的標籤檔案
│   ├── 0001.txt              # 格式：class_id x_center y_center width height
│   └── ...
├── classes_count/             # 類別計數檔案
│   ├── 0000.txt              # 每張圖像中各類別的數量統計
│   ├── 0001.txt
│   └── ...
└── class_names.txt           # 類別名稱對照表（格式：id:class_name）
```

## 詳細參數配置說明

### datainfo 參數字典

```python
datainfo = {
    # 圖像尺寸設定
    'Blank_image_size': (480, 640),        # 空白畫布尺寸 (高度, 寬度)
    'templateresize': (320, 320),          # 模板圖像調整尺寸 (寬度, 高度)
    
    # 類別和路徑設定
    'All_itemclass': All_class,            # 所有物件類別名稱列表
    'Treasure_item': Treasure_item_path,   # 寶物物件檔案路徑列表
    'Landmark_item': Landmark_item_path,   # 地標物件檔案路徑列表
    
    # 生成機率控制
    'Lost_treasure_rate': 0.2,             # 遺失圖像中包含寶物的機率 (0.0-1.0)
    'Targetitem_rate': 0.2,                # 生成目標圖像的機率 (0.0-1.0)
    'Overlap_rate': 0.5,                   # 物件重疊的機率 (0.0-1.0)
    
    # 批次設定
    'Batch_number': 10                     # 每批次生成的圖像數量
}
```

**參數詳細說明**：

- **Blank_image_size**: 定義合成圖像的畫布大小，所有物件將放置在此尺寸的畫布上
- **templateresize**: 所有來源圖像會先調整到此尺寸後再進行後續處理
- **All_itemclass**: 從檔名自動提取的所有類別名稱，用於建立類別索引
- **Treasure_item / Landmark_item**: 自動根據檔名關鍵字分類的物件路徑
- **Lost_treasure_rate**: 控制在"遺失圖像"類型中是否包含寶物物件
- **Targetitem_rate**: 控制生成"目標圖像"與"遺失圖像"的比例
- **Overlap_rate**: 控制物件間是否允許重疊擺放
- **Batch_number**: 單次執行生成的圖像數量

### transforminfo 參數字典

```python
transforminfo = {
    'Augmentation': Augmentation_dict,     # 資料增強參數字典
    'Max_landmark_items': 7,               # 單張圖像中地標物件的最大數量
    'Outer_effect': 0.4,                   # 外部效果的整體應用機率 (0.0-1.0)
    'Output_transform': None               # 輸出變換（目前未使用，保留擴展）
}
```

**參數詳細說明**：

- **Max_landmark_items**: 在"遺失圖像"類型中，地標物件數量會隨機選擇 1 到此數值之間
- **Outer_effect**: 控制是否對最終合成圖像應用外部效果（如模糊、雜訊等）
- **Output_transform**: 預留參數，可用於未來擴展輸出格式轉換功能

## 資料增強配置詳解

### 內部變換（Inner Transformations）
應用於所有物件（圖像和遮罩同時變換）：

```python
'inner': {
    # 變換開關
    'flip': True,                          # 啟用水平/垂直翻轉
    'rotate': True,                        # 啟用旋轉變換
    'scale': True,                         # 啟用縮放變換
    
    # 變換參數
    'flip_rate': 0.5,                      # 翻轉機率 (0.0-1.0)
    'rotate_range': (-30, 30),             # 旋轉角度範圍（度）
    'scale_range': (0.4, 0.8)              # 縮放倍數範圍
}
```

**內部變換說明**：
- 這些變換會同時應用到圖像和對應的遮罩，保持標註的準確性
- **flip_rate**: 每次翻轉（水平和垂直）都有此機率被應用
- **rotate_range**: 隨機選擇此範圍內的角度進行旋轉
- **scale_range**: 隨機縮放到此倍數範圍，小於 1.0 表示縮小

### 外部效果（Outer Effects）
僅應用於最終合成圖像（不影響標註）：

```python
'outer': {
    # === 變換機率設定 (0.0-1.0) ===
    'brightness': 0.2,                     # 亮度調整機率
    'contrast': 0.2,                       # 對比度調整機率
    'sharpness': 0.2,                      # 銳化調整機率
    'gaussian_noise': 0.1,                 # 高斯雜訊添加機率
    'gaussian_blur': 0.5,                  # 高斯模糊機率
    'motion_blur': 0.5,                    # 運動模糊機率
    'gradient_brightness': 0.3,            # 漸層亮度機率
    'perspective': 0.2,                    # 透視變換機率
    
    # === 亮度、對比度、銳化參數 ===
    'brightness_range': (0.5, 1.5),       # 亮度調整倍數範圍
    'contrast_range': (0.8, 1.2),         # 對比度調整倍數範圍  
    'sharpness_range': (0.8, 1.2),        # 銳化調整倍數範圍
    
    # === 模糊效果參數 ===
    'blur_kernel_size': (7, 7),           # 高斯模糊核心大小 (必須為奇數)
    'blur_sigma': (0.5, 3),               # 高斯模糊標準差範圍
    'motion_blur_kernel_size': (7, 27),   # 運動模糊核心大小範圍 (會自動選擇奇數)
    'motion_blur_angle_range': (-90, 90), # 運動模糊角度範圍（度）
    'motion_blur_distance_range': (5, 40), # 運動模糊距離範圍
    
    # === 漸層亮度參數 ===
    'gradient_brightness_range': (0.1, 10), # 漸層亮度強度範圍
    
    # === 透視變換參數 ===
    'perspective_distortion': 0.3,         # 透視扭曲強度 (0.0-1.0)
    
    # === 高斯雜訊參數 ===
    'gaussian_noise_list': [               # [平均值範圍, 最小標準差範圍, 最大標準差範圍]
        (-0.25, 0.25),                     # 雜訊平均值範圍
        (0.01, 0.1),                       # 最小標準差範圍
        (0.1, 0.5)                         # 最大標準差範圍
    ]
}
```

**外部效果詳細說明**：

#### 光學調整效果
- **brightness_range**: 1.0 為原始亮度，>1.0 變亮，<1.0 變暗
- **contrast_range**: 1.0 為原始對比度，>1.0 增強對比，<1.0 降低對比
- **sharpness_range**: 1.0 為原始銳化，>1.0 更銳利，<1.0 更模糊

#### 模糊效果
- **gaussian_blur**: 添加高斯模糊以模擬相機失焦效果
- **motion_blur**: 添加運動模糊以模擬相機或物體移動效果
- **blur_kernel_size**: 核心越大，模糊效果越強
- **blur_sigma**: 標準差越大，模糊範圍越廣

#### 特殊效果
- **gradient_brightness**: 在圖像上添加漸層亮度變化，模擬不均勻光照
- **perspective**: 添加透視扭曲，模擬不同拍攝角度
- **gaussian_noise**: 添加隨機雜訊，模擬感光元件雜訊

## YOLO 標籤格式說明

生成的標籤檔案採用標準 YOLO 格式：
```
class_id x_center y_center width height
```

**格式說明**：
- **class_id**: 物件類別的整數 ID（從 0 開始）
- **x_center**: 邊界框中心點的 X 座標（正規化到 0-1）
- **y_center**: 邊界框中心點的 Y 座標（正規化到 0-1）
- **width**: 邊界框的寬度（正規化到 0-1）
- **height**: 邊界框的高度（正規化到 0-1）

**範例標籤檔案內容**：
```
0 0.5 0.3 0.2 0.15
1 0.7 0.8 0.1 0.12
2 0.2 0.6 0.18 0.2
```

## 檢查和驗證工具

### 使用 checkyolo.py 驗證結果

```bash
python checkyolo.py
```

**工具功能**：
1. 讀取指定資料夾中的 YOLO 標籤檔案
2. 在對應圖像上繪製彩色邊界框
3. 顯示類別 ID 標籤
4. 將標註圖像儲存到 `check` 資料夾

**輸出範例**：
- 不同類別使用不同顏色的邊界框
- 每個邊界框上方顯示 "Class X" 標籤
- 檔案名稱格式：`annotated_原檔名.jpg`

## 進階使用技巧

### 1. 自訂增強參數

根據您的資料特性調整參數：

```python
# 增加幾何變換強度（適合需要更多變化的情況）
'rotate_range': (-45, 45),      # 更大的旋轉範圍
'scale_range': (0.3, 0.9),     # 更大的縮放範圍

# 調整光學效果（適合模擬不同光照條件）
'brightness_range': (0.3, 1.7), # 更大的亮度變化
'blur_sigma': (1.0, 5.0),       # 更強的模糊效果

# 修改雜訊參數（適合模擬低品質圖像）
'gaussian_noise_list': [
    (-0.1, 0.1),               # 較小的平均值變化
    (0.005, 0.05),             # 較小的最小標準差
    (0.05, 0.2)                # 較小的最大標準差
]
```

### 2. 批次大量生成

```python
# 在 main 函數中修改生成迴圈
current_count = 0
total_batches = 50              # 增加批次數量

for batch_idx in range(total_batches):
    print(f"正在生成第 {batch_idx+1}/{total_batches} 批次...")
    
    Kibo_datasets.initialize_batchdata()
    
    for count, data in enumerate(Kibo_datasets):
        img, label, classes = data
        
        # 儲存檔案的程式碼...
        current_count += 1
    
    print(f"已完成 {current_count} 張圖像")
```

### 3. 調整圖像類型比例

```python
# 修改 datainfo 參數來控制圖像類型
datainfo = {
    # ... 其他參數
    'Targetitem_rate': 0.3,        # 30% 目標圖像
    'Lost_treasure_rate': 0.1,     # 10% 遺失圖像包含寶物
    # 結果：30% 目標圖像，7% 含寶物遺失圖像，63% 純地標圖像
}
```

### 4. 效能最佳化設定

```python
# 針對不同硬體調整參數
datainfo = {
    # 低記憶體設定
    'Batch_number': 5,             # 減少批次大小
    'Blank_image_size': (320, 320), # 縮小畫布尺寸
    
    # 高效能設定
    'Batch_number': 20,            # 增加批次大小
    'Blank_image_size': (640, 640), # 增大畫布尺寸
}

transforminfo = {
    # 降低增強複雜度以提升速度
    'Outer_effect': 0.2,           # 降低外部效果機率
    'Max_landmark_items': 5,       # 減少最大物件數量
}
```

## 輸出品質檢查清單

完成生成後，建議進行以下檢查：

1. **檔案完整性檢查**
   - [ ] `images/` 資料夾包含所有圖像檔案
   - [ ] `labels/` 資料夾包含對應的標籤檔案
   - [ ] 圖像和標籤檔案數量一致

2. **標籤準確性檢查**
   - [ ] 使用 `checkyolo.py` 檢視標註結果
   - [ ] 確認邊界框位置正確
   - [ ] 驗證類別 ID 對應關係

3. **圖像品質檢查**
   - [ ] 圖像亮度和對比度適中
   - [ ] 物件尺寸合理
   - [ ] 增強效果自然

4. **資料分布檢查**
   - [ ] 檢查 `classes_count/` 中的類別分布
   - [ ] 確認目標圖像和遺失圖像比例
   - [ ] 驗證各類別出現頻率

## 授權與支援

此工具基於 Python 開源生態系統開發，請確保遵循相關套件的授權條款。

**技術支援提醒**：
- 確保所有依賴套件版本兼容
- 建議在虛擬環境中運行
- 定期備份重要的生成資料
- 保存成功的參數配置以便重現