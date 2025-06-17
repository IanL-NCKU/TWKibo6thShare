# Kibo-RPC Yolopatrol4 vs Yolofirstry 更新技術文檔

## 概述
本文檔詳細說明了從 Yolofirstry 到 Yolopatrol4 的主要技術更新和改進。Yolopatrol4 是一個完整的四區域巡邏系統，相比 Yolofirstry 的單區域處理有了顯著的功能擴展和技術改進。

## 主要技術更新點

### 1. 多區域巡邏系統 🚀
**Yolofirstry (舊版)**：
```java
// 只處理一個固定點
Point point = new Point(10.9d, -9.92284d, 5.195d);
Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
```

**Yolopatrol4 (新版)**：
```java
// 完整的四區域坐標和姿態陣列
private final Point[] AREA_POINTS = {
    new Point(10.9d, -10.0000d, 5.195d),    // Area 1
    new Point(10.925d, -8.875d, 4.602d),    // Area 2
    new Point(10.925d, -7.925d, 4.60093d),  // Area 3
    new Point(10.766d, -6.852d, 4.945d)     // Area 4
};

private final Quaternion[] AREA_QUATERNIONS = {
    new Quaternion(0f, 0f, -0.707f, 0.707f), // Area 1
    new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 2
    new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 3
    new Quaternion(0f, 0f, 1f, 0f)           // Area 4
};
```

### 2. 智能數據管理系統 📊
**新增功能**：
- **區域寶藏跟蹤**：`Map<Integer, Set<String>> areaTreasure` 用於跟蹤每個區域的寶藏類型
- **全局寶藏記錄**：`Set<String> foundTreasures` 統一管理所有發現的寶藏
- **區域Landmark映射**：`Map<String, Map<String, Integer>> areaLandmarks` 詳細記錄每個區域的landmark資訊

```java
// 初始化區域寶藏跟蹤系統
Map<Integer, Set<String>> areaTreasure = new HashMap<>();
for (int i = 1; i <= 4; i++) {
    areaTreasure.put(i, new HashSet<String>());
}
```

### 3. 完整Target處理系統 🎯
**Yolofirstry**：缺少target處理實現

**Yolopatrol4**：完整的target圖像處理管道
```java
/**
 * 處理target圖像以識別宇航員持有的寶藏類型
 */
private String processTargetImage(Mat targetImage, Size resizeSize)

/**
 * 基本的target圖像增強處理
 */
private Mat enhanceTargetImage(Mat image, Size resizeSize)

/**
 * 在指定區域中查找寶藏類型
 */
private int findTreasureInArea(String treasureType, Map<Integer, Set<String>> areaTreasure)
```

### 4. 增強的圖像處理管道 🖼️
**改進的ArUco marker處理**：
```java
// 新增：智能marker選擇，選擇最接近圖像中心的marker
Object[] filtered = keepClosestMarker(corners, ids, image);
List<Mat> filteredCorners = (List<Mat>) filtered[0];
Mat filteredIds = (Mat) filtered[1];
```

**增強的文件命名系統**：
```java
// 動態生成帶區域ID的文件名
String rawImageFilename = "area_" + areaId + "_raw.png";
String cropFilename = String.format("area_%d_cropped_region_%.0fx%.0f.png", 
                                   areaId, cropWarpSize.width, cropWarpSize.height);
```

### 5. 改進的錯誤處理和穩定性 🛡️
**新增功能**：
- 每個區域處理間的穩定性延遲
- 更詳細的異常處理和日志記錄
- 智能的fallback機制

```java
// 區域間穩定性延遲
try {
    Thread.sleep(500);
} catch (InterruptedException e) {
    Log.w(TAG, "Sleep interrupted");
}
```

### 6. 增強的YOLO檢測集成 🤖
**統一的檢測接口**：
```java
// 支持不同類型的圖像檢測
Object[] detected_items = detectitemfromcvimg(
    claHeBinImage, 
    0.5f,      // conf_threshold (可調節)
    "lost",    // img_type ("lost" 或 "target") 
    0.45f,     // standard_nms_threshold
    0.8f,      // overlap_nms_threshold
    320        // img_size
);
```

### 7. 智能的KeepClosestMarker算法 🎯
**新版本特色**：
```java
/**
 * 修復版本：只保留最接近圖像中心的marker
 * 正確處理ArUco的corner數據格式
 */
private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image)
```

### 8. 全面的任務流程管理 📋
**Yolopatrol4的完整流程**：
1. **初始化階段**：設置數據結構和參數
2. **多區域巡邏**：依序處理4個區域
3. **數據聚合**：整合所有區域的檢測結果
4. **宇航員交互**：處理target圖像並識別目標
5. **目標導航**：移動到目標寶藏位置
6. **任務完成**：拍攝目標快照

## 性能和可靠性改進

### 記憶體管理優化
- 更好的Mat對象生命週期管理
- 及時的資源釋放和清理
- 減少記憶體洩漏風險

### 配置靈活性
```java
// 可配置的圖像處理參數
Size cropWarpSize = new Size(640, 480);   // 裁剪/變形尺寸
Size resizeSize = new Size(320, 320);     // 最終處理尺寸
```

### 詳細的執行日志
```java
Log.i(TAG, "=== AREA PROCESSING SUMMARY ===");
for (int i = 1; i <= 4; i++) {
    Log.i(TAG, "Area " + i + " treasures: " + areaTreasure.get(i));
    Log.i(TAG, "Area " + i + " landmarks: " + areaLandmarks.get("area" + i));
}
```

## 總結

Yolopatrol4 代表了從單區域概念驗證到完整多區域巡邏系統，主要改進包括：

✅ **可擴展性**：從單區域擴展到四區域完整巡邏  
✅ **智能化**：增加了智能marker選擇和target處理  
✅ **穩定性**：改進的錯誤處理和資源管理  
✅ **可維護性**：更好的代碼結構和日志系統  
✅ **功能完整性**：完整的任務流程實現  

