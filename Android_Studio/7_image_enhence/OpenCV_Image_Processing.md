# OpenCV ArUco 標記處理完整指南

## 概述

本指南整理了在 Android 平台上使用 OpenCV 4.5.3.0 進行 ArUco 標記檢測和圖像處理的完整流程，包含透視變換、CLAHE 增強、二值化等技術。

## 目錄

1. [基礎 ArUco 標記檢測](#基礎-aruco-標記檢測)
2. [圖像處理流程](#圖像處理流程)
3. [warpPerspective vs resize 比較](#warpperspective-vs-resize-比較)
4. [CLAHE 對比度增強](#clahe-對比度增強)
5. [二值化方法比較](#二值化方法比較)
6. [完整代碼實現](#完整代碼實現)
7. [性能優化建議](#性能優化建議)

## 基礎 ArUco 標記檢測

### 導入必要的庫

```java
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;
```

### 標記檢測基礎代碼

```java
// 創建 ArUco 字典
Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

// 檢測標記
List<Mat> corners = new ArrayList<>();
Mat ids = new Mat();
Aruco.detectMarkers(image, dictionary, corners, ids);

if (corners.size() > 0) {
    // 獲取相機參數
    double[][] intrinsics = api.getNavCamIntrinsics();
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
    
    // 填充相機矩陣
    cameraMatrix.put(0, 0, intrinsics[0]);
    distCoeffs.put(0, 0, intrinsics[1]);
    
    // 估算標記姿態
    Mat rvecs = new Mat();
    Mat tvecs = new Mat();
    float markerLength = 0.05f; // 5cm 標記
    
    Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
}
```

## 圖像處理流程

### 1. 透視變換與裁剪

透視變換用於校正 ArUco 標記的視角，將傾斜的梯形變為正矩形：

```java
private Mat cropAndSaveRegion(Mat image, org.opencv.core.Point[] cropPoints2D) {
    // 定義目標點 (640x480 矩形)
    org.opencv.core.Point[] dstPoints = {
        new org.opencv.core.Point(0, 0),       // 左上
        new org.opencv.core.Point(639, 0),     // 右上
        new org.opencv.core.Point(639, 479),   // 右下
        new org.opencv.core.Point(0, 479)      // 左下
    };
    
    // 創建變換矩陣
    MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
    MatOfPoint2f dstPointsMat = new MatOfPoint2f(dstPoints);
    Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
    
    // 應用透視變換
    Mat croppedImage = new Mat();
    Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
    
    return croppedImage;
}
```

### 2. 調整圖像大小

將圖像調整為 YOLO 格式的 320x320：

```java
// 調整為 320x320
Mat yoloImage = new Mat();
Imgproc.resize(croppedImage, yoloImage, new Size(320, 320));

// 打印最小/最大值
Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(yoloImage);
Log.i(TAG, String.format("圖像統計 - 最小值: %.2f, 最大值: %.2f", 
      minMaxResult.minVal, minMaxResult.maxVal));
```

## warpPerspective vs resize 比較

| 方面 | `warpPerspective` | `resize` |
|------|-------------------|----------|
| **用途** | 校正透視失真 + 調整大小 | 僅調整大小 |
| **輸入需求** | 需要4個角點對 + 變換矩陣 | 僅需目標大小 |
| **質量** | 可將梯形變為矩形 | 保持原有透視 |
| **使用場景** | 斜拍的圖像 | 圖像已經是正的 |

### 實際效果比較

```java
// 方法1: warpPerspective (推薦用於 ArUco)
// 同時校正角度 + 調整大小到 320x320
Imgproc.warpPerspective(image, output, perspectiveMatrix, new Size(320, 320));

// 方法2: resize (僅用於調整大小)  
// 簡單縮放，保持原有角度
Imgproc.resize(image, output, new Size(320, 320));

// 方法3: 組合使用
// 先校正到大尺寸，再高質量縮放
Imgproc.warpPerspective(image, temp, perspectiveMatrix, new Size(640, 640));
Imgproc.resize(temp, output, new Size(320, 320), 0, 0, Imgproc.INTER_AREA);
```

**ArUco 標記建議**: 使用 `warpPerspective` 直接到 320x320，因為標記通常是斜拍的。

## CLAHE 對比度增強

CLAHE (對比度限制自適應直方圖均衡化) 在 OpenCV 4.5.3.0 中可用，對 ArUco 標記檢測效果顯著：

### 基本使用

```java
// 創建 CLAHE 對象
CLAHE clahe = Imgproc.createCLAHE();

// 設置參數
clahe.setClipLimit(2.0);  // 對比度控制 (1.0-4.0)
clahe.setTilesGridSize(new Size(8, 8));  // 網格大小

// 應用 CLAHE
Mat enhancedImage = new Mat();
clahe.apply(inputImage, enhancedImage);
```

### CLAHE 參數調整

| 參數 | 推薦值 | 說明 |
|------|-------|------|
| **Clip Limit** | 2.0 | 控制對比度增強強度 |
| **Tiles Grid** | 8x8 | 適合 320x320 圖像 |

```java
// 不同參數效果測試
double[] clipLimits = {1.0, 2.0, 3.0, 4.0};
int[][] tileSizes = {{4,4}, {8,8}, {16,16}, {32,32}};

for (double clip : clipLimits) {
    CLAHE clahe = Imgproc.createCLAHE();
    clahe.setClipLimit(clip);
    clahe.setTilesGridSize(new Size(8, 8));
    
    Mat result = new Mat();
    clahe.apply(inputImage, result);
    
    api.saveMatImage(result, String.format("clahe_clip%.1f.png", clip));
    result.release();
}
```

### ArUco 標記的 CLAHE 優勢

✅ **更好的邊緣定義** - CLAHE 增強標記邊界  
✅ **處理不均勻光照** - 適應局部照明變化  
✅ **改善二值化效果** - 更好的前景/背景分離  
✅ **更魯棒的檢測** - 在困難光照條件下工作  

## 二值化方法比較

### 固定閾值 vs Otsu 自動閾值

| 方面 | **固定二值化 (127)** | **Otsu 算法** |
|------|---------------------|---------------|
| **工作原理** | 對整個圖像使用相同閾值 | 自動找到最佳閾值 |
| **速度** | ⚡ 非常快 | 🐌 稍慢 (分析直方圖) |
| **適應性** | ❌ 對所有圖像相同 | ✅ 適應每個圖像 |
| **質量** | ⚠️ 時好時壞 | ✅ 通常更好 |
| **光照敏感性** | ❌ 在不均勻光照下失效 | ✅ 處理光照變化 |

### 代碼實現比較

```java
// 方法1: 固定閾值 (原有方法)
Mat binaryFixed = new Mat();
Imgproc.threshold(inputImage, binaryFixed, 127, 255, Imgproc.THRESH_BINARY);
// → 總是使用 127，不管圖像內容

// 方法2: Otsu 自動閾值 (推薦)
Mat binaryOtsu = new Mat();
double optimalThreshold = Imgproc.threshold(inputImage, binaryOtsu, 0, 255, 
                                          Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
// → 自動為這個特定圖像找到最佳閾值
// → optimalThreshold 包含它選擇的值 (例如 142.5)
```

### Otsu 算法工作原理

1. **分析**: 查看圖像中的所有像素值
2. **直方圖**: 創建每個亮度值的像素數量圖表
3. **智能數學**: 嘗試每個可能的閾值 (0-255)
4. **計算**: 對每個閾值，測量分離效果
5. **選擇**: 選擇提供最佳分離的閾值

### 實際場景比較

```java
// 場景1: 光照良好的 ArUco 標記
// 圖像均值: ~120
// 固定(127): 效果不錯 ✅
// Otsu: 找到 ~125，非常相似 ✅
// 結論: 兩者都可以，固定更快

// 場景2: 明亮/過曝圖像  
// 圖像均值: ~200
// 固定(127): 太低，噪點變白 ❌
// Otsu: 找到 ~180，好很多 ✅
// 結論: Otsu 明顯獲勝

// 場景3: 暗/陰影圖像
// 圖像均值: ~60
// 固定(127): 太高，標記消失 ❌
// Otsu: 找到 ~85，保留標記 ✅  
// 結論: Otsu 明顯獲勝

// 場景4: CLAHE 增強後
// 增強對比度: 更大動態範圍
// 固定(127): 可能不是最優 ⚠️
// Otsu: 適應增強對比度 ✅
// 結論: Otsu + CLAHE = 最佳組合
```

## 完整代碼實現

### 返回二值化結果的 cropAndSaveRegion 方法

```java
private Mat cropAndSaveRegion(Mat image, org.opencv.core.Point[] cropPoints2D) {
    try {
        // 1. 透視變換到 640x480
        org.opencv.core.Point[] dstPoints = {
            new org.opencv.core.Point(0, 0), new org.opencv.core.Point(639, 0),
            new org.opencv.core.Point(639, 479), new org.opencv.core.Point(0, 479)
        };
        
        MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
        MatOfPoint2f dstPointsMat = new MatOfPoint2f(dstPoints);
        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
        
        Mat croppedImage = new Mat();
        Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
        
        // 2. 調整為 320x320 (YOLO 格式)
        Mat yoloImage = new Mat();
        Imgproc.resize(croppedImage, yoloImage, new Size(320, 320));
        
        // 打印統計信息
        Core.MinMaxLocResult yoloStats = Core.minMaxLoc(yoloImage);
        Log.i(TAG, String.format("YOLO 圖像 (320x320) - 最小值: %.2f, 最大值: %.2f", 
              yoloStats.minVal, yoloStats.maxVal));
        
        // 3. 應用 CLAHE 增強對比度
        Mat claheImage = new Mat();
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(2.0);
        clahe.setTilesGridSize(new Size(8, 8));
        clahe.apply(yoloImage, claheImage);
        
        Core.MinMaxLocResult claheStats = Core.minMaxLoc(claheImage);
        Log.i(TAG, String.format("CLAHE 增強 (320x320) - 最小值: %.2f, 最大值: %.2f", 
              claheStats.minVal, claheStats.maxVal));
        
        // 4. Otsu 自動二值化 (推薦)
        Mat binarizedOtsu = new Mat();
        double otsuThreshold = Imgproc.threshold(claheImage, binarizedOtsu, 0, 255, 
                                               Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        
        Core.MinMaxLocResult otsuStats = Core.minMaxLoc(binarizedOtsu);
        Log.i(TAG, String.format("Otsu 二值化 (%.1f) - 最小值: %.2f, 最大值: %.2f", 
              otsuThreshold, otsuStats.minVal, otsuStats.maxVal));
        
        // 5. 保存處理步驟
        api.saveMatImage(croppedImage, "01_cropped_640x480.png");
        api.saveMatImage(yoloImage, "02_yolo_320x320.png");
        api.saveMatImage(claheImage, "03_clahe_enhanced.png");
        api.saveMatImage(binarizedOtsu, "04_binary_otsu.png");
        
        // 6. 比較：固定閾值二值化
        Mat binaryFixed = new Mat();
        Imgproc.threshold(claheImage, binaryFixed, 127, 255, Imgproc.THRESH_BINARY);
        api.saveMatImage(binaryFixed, "05_binary_fixed_127.png");
        
        Log.i(TAG, "=== 處理摘要 ===");
        Log.i(TAG, "1. 透視變換: 640x480");
        Log.i(TAG, "2. YOLO 調整: 320x320");
        Log.i(TAG, "3. CLAHE 增強應用");
        Log.i(TAG, "4. Otsu 自動二值化");
        Log.i(TAG, "5. 返回: 二值化 Otsu 結果");
        
        // 清理資源 (不釋放返回的 binarizedOtsu)
        srcPointsMat.release();
        dstPointsMat.release();
        perspectiveMatrix.release();
        croppedImage.release();
        yoloImage.release();
        claheImage.release();
        binaryFixed.release();
        
        return binarizedOtsu; // 返回最佳品質的二值化結果
        
    } catch (Exception e) {
        Log.e(TAG, "圖像處理錯誤: " + e.getMessage());
        return new Mat(); // 錯誤時返回空 Mat
    }
}
```

### 調用示例

```java
// 在 checkCornerAndCropRegion 方法中使用
if (cropPoints2D.length == 4) {
    // 獲取處理後的二值化圖像
    Mat processedBinaryImage = cropAndSaveRegion(image, cropPoints2D);
    
    if (!processedBinaryImage.empty()) {
        Log.i(TAG, String.format("獲得二值化圖像: %dx%d", 
              processedBinaryImage.cols(), processedBinaryImage.rows()));
        
        // 進一步處理
        // 1. 輪廓檢測
        findContoursInBinaryImage(processedBinaryImage);
        
        // 2. 對象檢測
        detectObjectsInBinaryImage(processedBinaryImage);
        
        // 3. 特徵提取
        extractFeatures(processedBinaryImage);
        
        // 使用完畢後釋放
        processedBinaryImage.release();
    }
}
```

## 性能優化建議

### 1. 方法選擇指南

| 場景 | 推薦方法 | 原因 |
|------|---------|------|
| **穩定光照** | 固定閾值(127) | 最快，可靠 |
| **變化光照** | CLAHE + Otsu | 最穩健 |
| **實時處理** | 固定閾值 | 速度優先 |
| **高質量需求** | CLAHE + Otsu | 質量優先 |

### 2. 參數調整建議

```java
// 根據應用場景調整 CLAHE 參數
if (isRealTimeProcessing) {
    clahe.setClipLimit(1.5);  // 較低，速度優先
    clahe.setTilesGridSize(new Size(4, 4));  // 較小網格
} else {
    clahe.setClipLimit(2.5);  // 較高，質量優先  
    clahe.setTilesGridSize(new Size(16, 16));  // 較大網格
}
```

### 3. 記憶體管理

```java
// 正確的記憶體管理模式
Mat processedImage = cropAndSaveRegion(sourceImage, corners);
try {
    // 使用 processedImage 進行處理
    doSomethingWith(processedImage);
} finally {
    // 確保釋放
    if (!processedImage.empty()) {
        processedImage.release();
    }
}
```

### 4. 品質評估

```java
private void evaluateImageQuality(Mat binaryImage) {
    // 統計像素分布
    Scalar sumScalar = Core.sumElems(binaryImage);
    int whitePixels = (int)(sumScalar.val[0] / 255);
    int totalPixels = (int)binaryImage.total();
    double whitePercentage = (whitePixels * 100.0) / totalPixels;
    
    Log.i(TAG, String.format("圖像品質 - 白色: %.1f%%, 黑色: %.1f%%", 
          whitePercentage, 100.0 - whitePercentage));
    
    // 品質判斷
    if (whitePercentage < 10 || whitePercentage > 90) {
        Log.w(TAG, "警告: 圖像可能過度二值化");
    } else {
        Log.i(TAG, "圖像品質良好");
    }
}
```

## 結論

本指南提供了使用 OpenCV 4.5.3.0 在 Android 平台上進行 ArUco 標記處理的完整解決方案。關鍵要點：

1. **使用 `warpPerspective`** 而非 `resize` 來校正 ArUco 標記的透視失真
2. **CLAHE 增強** 顯著改善在不同光照條件下的檢測效果
3. **Otsu 自動閾值** 比固定閾值更穩健，特別是與 CLAHE 結合使用時
4. **合適的參數設置** 對於不同應用場景至關重要
5. **正確的記憶體管理** 防止內存洩漏

推薦的最佳實踐組合：**透視變換 → CLAHE 增強 → Otsu 自動二值化**，這提供了最穩健和高質量的 ArUco 標記處理效果。