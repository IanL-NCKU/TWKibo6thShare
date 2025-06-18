# YOLO 轉 ONNX 完整教學指南

本指南將詳細介紹如何將訓練好的 YOLOv8 模型轉換為 ONNX 格式，以及不同 ONNX Runtime 版本的對應設定。

## 目錄
1. [什麼是 ONNX？](#什麼是-ONNX)
2. [為什麼要轉換為 ONNX？](#為什麼要轉換為-ONNX)
3. [版本對應表](#版本對應表)
4. [環境準備](#環境準備)
5. [轉換方法詳解](#轉換方法詳解)
6. [參數詳細說明](#參數詳細說明)
7. [常見問題與解決方案](#常見問題與解決方案)
8. [驗證與測試](#驗證與測試)

---

## 什麼是 ONNX？

**ONNX (Open Neural Network eXchange)** 是一個開放的神經網路交換格式，它允許不同深度學習框架之間進行模型轉換和部署。

### 🎯 ONNX 的優勢
- **跨平台部署**：可在不同作業系統運行
- **跨框架支援**：支援 PyTorch、TensorFlow、scikit-learn 等
- **高效推理**：優化推理速度，降低延遲
- **硬體最佳化**：支援 CPU、GPU、邊緣設備等

---

## 為什麼要轉換為 ONNX？

### 📱 部署需求
1. **生產環境部署**：伺服器端高效推理
2. **邊緣設備運行**：手機、嵌入式系統
3. **跨語言支援**：C++、Java、C# 等語言調用
4. **雲端服務整合**：Azure ML、AWS SageMaker 等

### ⚡ 效能優勢
- **推理速度提升**：相比原始 PyTorch 模型快 2-5 倍
- **記憶體使用降低**：模型大小通常減少 20-50%
- **批次處理優化**：支援動態批次大小

---

## 版本對應表

### 🔧 ONNX Runtime 與 Opset 版本對應

| ONNX Runtime 版本 | 支援的最高 Opset | 建議 Opset | 發布時間 | 備註 |
|-------------------|-----------------|------------|----------|------|
| **1.15.1** | **18** | **15** | 2023-06 | 🌟 推薦版本 |
| **1.12.1** | **17** | **11** | 2022-08 | 穩定版本 |


### 📋 推薦配置

#### 🌟 最新穩定配置 (推薦)
```python
# 適用於 ONNX Runtime 1.15.1
opset = 15
```

#### 🔒 相容穩定配置
```python
# 適用於 ONNX Runtime 1.12.1
opset = 11
```

#### ⚠️ 安全通用配置
```python
# 適用於大多數版本
opset = 11  # 通用相容性最佳
```

---

## 環境準備

### 1. 安裝必要套件

```bash
# 確保有最新的 ultralytics
pip install ultralytics>=8.0.0

# 安裝 ONNX 相關套件
pip install onnx>=1.12.0
pip install onnxruntime>=1.12.1  # 或 onnxruntime-gpu

# 可選：安裝模型最佳化工具
pip install onnx-simplifier
```

### 2. 檢查環境

```python
import ultralytics
import onnx
import onnxruntime
import torch

print(f"Ultralytics: {ultralytics.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print(f"PyTorch: {torch.__version__}")
```

---

## 轉換方法詳解

### 🎯 基本轉換方法

```python
from ultralytics import YOLO
import shutil
import os

# ============================================
# 基本設定
# ============================================

# 輸入檔案路徑（訓練好的 .pt 檔案）
loadyolo_path = r'path/to/your/best.pt'

# 輸出檔案路徑（想要儲存的 .onnx 檔案位置）
target_path = r'path/to/output/model.onnx'

print("=== YOLO 轉 ONNX 轉換開始 ===")

# ============================================
# 方法一：基本轉換（推薦）
# ============================================

# 載入訓練好的模型
model = YOLO(loadyolo_path)

# 執行轉換
onnx_path = model.export(
    format='onnx',          # 輸出格式
    imgsz=320,             # 輸入圖片尺寸（必須與訓練時一致）
    opset=17,              # ONNX opset 版本（根據 Runtime 版本選擇）
    simplify=True,         # 簡化模型結構
    dynamic=False,         # 是否支援動態形狀
    half=False             # 是否使用半精度
)

print(f"✅ 模型轉換完成: {onnx_path}")

# 移動到目標位置
if target_path != onnx_path:
    shutil.move(onnx_path, target_path)
    print(f"✅ 檔案移動至: {target_path}")

print("=== 轉換完成 ===")
```

### 🔧 進階轉換選項

#### 版本 1.15.1 最佳化配置
```python
# 適用於 ONNX Runtime 1.15.1
model.export(
    format='onnx',
    imgsz=320,
    opset=17,              # 最新支援版本
    simplify=True,         # 啟用模型簡化
    dynamic=False,         # 固定輸入尺寸以提升效能
    half=False,            # 使用 FP32 確保精度
    int8=False,            # 不使用 INT8 量化
    device='cpu'           # 指定轉換設備
)
```

#### 版本 1.12.1 相容配置
```python
# 適用於 ONNX Runtime 1.12.1
model.export(
    format='onnx',
    imgsz=320,
    opset=15,              # 相容版本
    simplify=True,
    dynamic=False,
    half=False,
    device='cpu'
)
```

#### 通用相容配置
```python
# 最大相容性配置
model.export(
    format='onnx',
    imgsz=320,
    opset=11,              # 最通用版本
    simplify=False,        # 不簡化以確保相容性
    dynamic=False,
    half=False,
    device='cpu'
)
```

### 📦 批次轉換腳本

```python
import os
from ultralytics import YOLO

def batch_convert_to_onnx(model_dir, output_dir, opset=17):
    """
    批次轉換多個 .pt 檔案為 .onnx
    
    Args:
        model_dir: 包含 .pt 檔案的目錄
        output_dir: 輸出 .onnx 檔案的目錄
        opset: ONNX opset 版本
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 尋找所有 .pt 檔案
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    print(f"找到 {len(pt_files)} 個模型檔案")
    
    for pt_file in pt_files:
        print(f"\n處理: {pt_file}")
        
        # 完整路徑
        pt_path = os.path.join(model_dir, pt_file)
        onnx_name = pt_file.replace('.pt', '.onnx')
        onnx_path = os.path.join(output_dir, onnx_name)
        
        try:
            # 載入並轉換模型
            model = YOLO(pt_path)
            exported_path = model.export(
                format='onnx',
                imgsz=320,
                opset=opset,
                simplify=True,
                dynamic=False,
                half=False
            )
            
            # 移動到目標位置
            if exported_path != onnx_path:
                shutil.move(exported_path, onnx_path)
            
            print(f"✅ 成功: {onnx_name}")
            
        except Exception as e:
            print(f"❌ 失敗: {pt_file} - {str(e)}")
    
    print(f"\n批次轉換完成！")

# 使用範例
# batch_convert_to_onnx(
#     model_dir=r'path/to/models',
#     output_dir=r'path/to/onnx_models',
#     opset=17
# )
```

---

## 參數詳細說明

### 🔧 model.export() 參數完整解析

| 參數 | 類型 | 預設值 | 說明 | 建議設定 |
|------|------|--------|------|----------|
| `format` | str | 'torchscript' | 輸出格式 | **'onnx'** |
| `imgsz` | int/tuple | 640 | 輸入圖片尺寸 | **320**（與訓練一致） |
| `opset` | int | None | ONNX opset 版本 | **17**(1.15.1) / **15**(1.12.1) |
| `simplify` | bool | False | 是否簡化模型 | **True**（推薦） |
| `dynamic` | bool | False | 動態輸入尺寸 | **False**（效能最佳） |
| `half` | bool | False | 半精度浮點數 | **False**（精度優先） |
| `int8` | bool | False | INT8 量化 | **False**（除非需要） |
| `device` | str | None | 轉換設備 | **'cpu'**（穩定） |

### 📊 參數選擇建議

#### 🎯 效能優先設定
```python
model.export(
    format='onnx',
    imgsz=320,
    opset=17,
    simplify=True,      # 最佳化模型結構
    dynamic=False,      # 固定尺寸以提升速度
    half=True,          # 使用半精度（需確認硬體支援）
    device='cpu'
)
```

#### 🎯 精度優先設定
```python
model.export(
    format='onnx',
    imgsz=320,
    opset=17,
    simplify=False,     # 保持原始結構
    dynamic=False,
    half=False,         # 使用全精度
    int8=False,         # 不量化
    device='cpu'
)
```

#### 🎯 相容性優先設定
```python
model.export(
    format='onnx',
    imgsz=320,
    opset=11,           # 最廣泛支援的版本
    simplify=False,     # 避免相容性問題
    dynamic=False,
    half=False,
    device='cpu'
)
```

---

## 常見問題與解決方案

### ❌ 問題 1：Opset 版本不相容

**錯誤訊息：**
```
RuntimeError: Unsupported opset version: 18
```

**解決方案：**
```python
# 降低 opset 版本
model.export(opset=15)  # 或更低版本
```

### ❌ 問題 2：模型尺寸不匹配

**錯誤訊息：**
```
Input shape mismatch
```

**解決方案：**
```python
# 確保使用與訓練時相同的尺寸
model.export(imgsz=320)  # 與訓練時一致
```

### ❌ 問題 3：記憶體不足

**錯誤訊息：**
```
CUDA out of memory
```

**解決方案：**
```python
# 使用 CPU 進行轉換
model.export(device='cpu')

# 或者清理 GPU 記憶體
import torch
torch.cuda.empty_cache()
```

### ❌ 問題 4：模型簡化失敗

**錯誤訊息：**
```
Model simplification failed
```

**解決方案：**
```python
# 關閉模型簡化
model.export(simplify=False)
```

### ❌ 問題 5：動態形狀問題

**錯誤訊息：**
```
Dynamic shapes not supported
```

**解決方案：**
```python
# 使用固定形狀
model.export(dynamic=False)
```

---

## 驗證與測試

### 🧪 轉換後驗證腳本

```python
import onnx
import onnxruntime as ort
import numpy as np

def verify_onnx_model(onnx_path, input_shape=(1, 3, 320, 320)):
    """
    驗證 ONNX 模型是否正常
    
    Args:
        onnx_path: ONNX 模型路徑
        input_shape: 輸入張量形狀
    """
    print(f"驗證模型: {onnx_path}")
    
    try:
        # 1. 檢查模型結構
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ 模型結構檢查通過")
        
        # 2. 建立推理會話
        ort_session = ort.InferenceSession(onnx_path)
        print("✅ ONNX Runtime 載入成功")
        
        # 3. 檢查輸入輸出
        input_name = ort_session.get_inputs()[0].name
        input_shape_model = ort_session.get_inputs()[0].shape
        output_shape = ort_session.get_outputs()[0].shape
        
        print(f"📊 輸入名稱: {input_name}")
        print(f"📊 輸入形狀: {input_shape_model}")
        print(f"📊 輸出形狀: {output_shape}")
        
        # 4. 測試推理
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = ort_session.run(None, {input_name: dummy_input})
        
        print(f"✅ 推理測試成功")
        print(f"📊 輸出張量形狀: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 驗證失敗: {str(e)}")
        return False

# 使用範例
verify_onnx_model(r'path/to/your/model.onnx')
```

### 📊 效能比較測試

```python
import time
import torch
import onnxruntime as ort
from ultralytics import YOLO

def performance_comparison(pt_path, onnx_path, test_iterations=100):
    """
    比較 PyTorch 和 ONNX 模型的推理效能
    """
    print("=== 效能比較測試 ===")
    
    # 準備測試資料
    dummy_input = torch.randn(1, 3, 320, 320).float()
    
    # 1. PyTorch 模型測試
    print("\n🔥 PyTorch 模型測試")
    pt_model = YOLO(pt_path)
    pt_model.model.eval()
    
    # 暖身
    with torch.no_grad():
        for _ in range(10):
            _ = pt_model.model(dummy_input)
    
    # 測試
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_iterations):
            _ = pt_model.model(dummy_input)
    pt_time = time.time() - start_time
    
    print(f"PyTorch 平均推理時間: {pt_time/test_iterations*1000:.2f} ms")
    
    # 2. ONNX 模型測試
    print("\n⚡ ONNX 模型測試")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    numpy_input = dummy_input.numpy()
    
    # 暖身
    for _ in range(10):
        _ = ort_session.run(None, {input_name: numpy_input})
    
    # 測試
    start_time = time.time()
    for _ in range(test_iterations):
        _ = ort_session.run(None, {input_name: numpy_input})
    onnx_time = time.time() - start_time
    
    print(f"ONNX 平均推理時間: {onnx_time/test_iterations*1000:.2f} ms")
    
    # 3. 結果比較
    speedup = pt_time / onnx_time
    print(f"\n📊 效能提升: {speedup:.2f}x")
    print(f"📊 時間節省: {(1-onnx_time/pt_time)*100:.1f}%")

# 使用範例
# performance_comparison('model.pt', 'model.onnx')
```

---

## 總結與建議

### 🎯 最佳實踐

1. **版本選擇**
   - 生產環境：使用 ONNX Runtime 1.15.1 + opset 17 or 15
   - 穩定環境：使用 ONNX Runtime 1.12.1 + opset 15 or 13
   - 廣泛相容：使用 opset 11

2. **參數設定**
   - 固定輸入尺寸：`dynamic=False`
   - 啟用簡化：`simplify=True`
   - 使用 CPU 轉換：`device='cpu'`

3. **檔案管理**
   - 使用描述性檔名：`yolo_v8n_400.onnx`
   - 保留原始 .pt 檔案作為備份
   - 建立版本控制記錄

### 📋 轉換檢查清單

- [ ] 確認 ONNX Runtime 版本
- [ ] 選擇對應的 opset 版本
- [ ] 設定正確的輸入尺寸
- [ ] 執行轉換並檢查錯誤
- [ ] 驗證模型載入與推理
- [ ] 進行效能比較測試
- [ ] 建立檔案備份與記錄
