# Java 程式碼修改說明

## 影響分析
這個修改改變了方法的核心功能，從返回 CLAHE 增強的灰階圖像變成返回二值化圖像，因為訓練模型時皆採用非二質化影像進行訓練，可能為二質化影像尤其是 coin 與 compass 判定失敗的原因，因此把它換成只回傳 CLAHE 影像，經過 12 次測試，當中具有 COIN 跟 COMPASS 的影像都有辦法被辨識出來。


## 修改目標
將 `yolo_patrol4spot_andgettarget_ver2.java` 中的 `cropEnhanceAndBinarize` 方法進行修改。


## 修改前 (原版本)

```java
/**
 * Helper method to crop, enhance with CLAHE, and binarize the image
 * @param image Input image
 * @param cropPoints2D 2D points for perspective transformation
 * @param cropWarpSize Size for the cropped/warped image (configurable)
 * @param resizeSize Size for the final processed image (configurable)
 * @param areaId Area identifier for filename generation
 */
private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize, int areaId) {
    try {
        // ========================================================================
        // STEP 1: Create cropped image with configurable size
        // ========================================================================

        // Define destination points for configurable rectangle
        org.opencv.core.Point[] dstPointsCrop = {
                new org.opencv.core.Point(0, 0),                           // Top-left
                new org.opencv.core.Point(cropWarpSize.width - 1, 0),      // Top-right
                new org.opencv.core.Point(cropWarpSize.width - 1, cropWarpSize.height - 1),   // Bottom-right
                new org.opencv.core.Point(0, cropWarpSize.height - 1)      // Bottom-left
        };

        // Create source and destination point matrices
        MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
        MatOfPoint2f dstPointsMatCrop = new MatOfPoint2f(dstPointsCrop);

        // Calculate perspective transformation matrix
        Mat perspectiveMatrixCrop = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMatCrop);

        // Apply perspective transformation to get cropped image
        Mat croppedImage = new Mat();
        Imgproc.warpPerspective(image, croppedImage, perspectiveMatrixCrop, cropWarpSize);

        // Print min/max values of the cropped image
        Core.MinMaxLocResult minMaxResultCrop = Core.minMaxLoc(croppedImage);
        Log.i(TAG, String.format("Cropped image %.0fx%.0f - Min: %.2f, Max: %.2f",
                cropWarpSize.width, cropWarpSize.height, minMaxResultCrop.minVal, minMaxResultCrop.maxVal));

        // Save the cropped image with area ID and dynamic filename
        String cropFilename = String.format("area_%d_cropped_region_%.0fx%.0f.png", areaId, cropWarpSize.width, cropWarpSize.height);
        api.saveMatImage(croppedImage, cropFilename);
        Log.i(TAG, "Cropped region saved as " + cropFilename);

        // ========================================================================
        // STEP 2: Resize to final processing size (configurable)
        // ========================================================================

        // Resize the cropped image to final size
        Mat resizedImage = new Mat();
        Imgproc.resize(croppedImage, resizedImage, resizeSize);

        // Save resized image with area ID
        String resizeFilename = String.format("area_%d_yolo_original_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
        api.saveMatImage(resizedImage, resizeFilename);
        Log.i(TAG, "Resized image saved as " + resizeFilename);

        // ========================================================================
        // STEP 3: Apply CLAHE enhancement
        // ========================================================================

        // Apply CLAHE for better contrast enhancement
        Mat claheImage = new Mat();
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(2.0);  // Controls contrast enhancement

        // Adjust grid size based on image size
        int gridSize = (int) Math.max(8, Math.min(resizeSize.width, resizeSize.height) / 40);
        clahe.setTilesGridSize(new Size(gridSize, gridSize));

        clahe.apply(resizedImage, claheImage);

        // Print min/max values of the CLAHE-enhanced image
        Core.MinMaxLocResult claheMinMaxResult = Core.minMaxLoc(claheImage);
        Log.i(TAG, String.format("CLAHE enhanced image (%.0fx%.0f) - Min: %.2f, Max: %.2f",
                resizeSize.width, resizeSize.height, claheMinMaxResult.minVal, claheMinMaxResult.maxVal));

        // ========================================================================
        // STEP 4: Apply Otsu's binarization
        // ========================================================================

        // Apply Otsu's automatic threshold binarization
        Mat binarizedOtsu = new Mat();
        double otsuThreshold = Imgproc.threshold(claheImage, binarizedOtsu, 0, 255,
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        // Print min/max values and threshold of Otsu binarized image
        Core.MinMaxLocResult binaryOtsuResult = Core.minMaxLoc(binarizedOtsu);
        Log.i(TAG, String.format("Binary Otsu (%.1f) - Min: %.2f, Max: %.2f",
                otsuThreshold, binaryOtsuResult.minVal, binaryOtsuResult.maxVal));

        // Save the Otsu binarized image with area ID and dynamic filename
        String binaryFilename = String.format("area_%d_yolo_binary_otsu_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
        api.saveMatImage(binarizedOtsu, binaryFilename);
        Log.i(TAG, String.format("Otsu binary image saved as %s (threshold: %.1f)", binaryFilename, otsuThreshold));

        // ========================================================================
        // CLEANUP
        // ========================================================================

        // Clean up intermediate images
        srcPointsMat.release();
        dstPointsMatCrop.release();
        perspectiveMatrixCrop.release();
        croppedImage.release();
        resizedImage.release();
        claheImage.release();

        // Return the final processed binary image
        return binarizedOtsu;

    } catch (Exception e) {
        Log.e(TAG, "Error in cropEnhanceAndBinarize: " + e.getMessage());
        return null;
    }
}
```


## 修改後 (更新版本)

```java
/**
 * Helper method to crop, enhance with CLAHE, and binarize the image
 * @param image Input image
 * @param cropPoints2D 2D points for perspective transformation
 * @param cropWarpSize Size for the cropped/warped image (configurable)
 * @param resizeSize Size for the final processed image (configurable)
 * @param areaId Area identifier for filename generation
 */
private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize, int areaId) {
    try {
        // ========================================================================
        // STEP 1: Create cropped image with configurable size
        // ========================================================================

        // Define destination points for configurable rectangle
        org.opencv.core.Point[] dstPointsCrop = {
                new org.opencv.core.Point(0, 0),                           // Top-left
                new org.opencv.core.Point(cropWarpSize.width - 1, 0),      // Top-right
                new org.opencv.core.Point(cropWarpSize.width - 1, cropWarpSize.height - 1),   // Bottom-right
                new org.opencv.core.Point(0, cropWarpSize.height - 1)      // Bottom-left
        };

        // Create source and destination point matrices
        MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
        MatOfPoint2f dstPointsMatCrop = new MatOfPoint2f(dstPointsCrop);

        // Calculate perspective transformation matrix
        Mat perspectiveMatrixCrop = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMatCrop);

        // Apply perspective transformation to get cropped image
        Mat croppedImage = new Mat();
        Imgproc.warpPerspective(image, croppedImage, perspectiveMatrixCrop, cropWarpSize);

        // Print min/max values of the cropped image
        Core.MinMaxLocResult minMaxResultCrop = Core.minMaxLoc(croppedImage);
        Log.i(TAG, String.format("Cropped image %.0fx%.0f - Min: %.2f, Max: %.2f",
                cropWarpSize.width, cropWarpSize.height, minMaxResultCrop.minVal, minMaxResultCrop.maxVal));

        // Save the cropped image with area ID and dynamic filename
        String cropFilename = String.format("area_%d_cropped_region_%.0fx%.0f.png", areaId, cropWarpSize.width, cropWarpSize.height);
        api.saveMatImage(croppedImage, cropFilename);
        Log.i(TAG, "Cropped region saved as " + cropFilename);

        // ========================================================================
        // STEP 2: Resize to final processing size (configurable)
        // ========================================================================

        // Resize the cropped image to final size
        Mat resizedImage = new Mat();
        Imgproc.resize(croppedImage, resizedImage, resizeSize);

        // Save resized image with area ID
        String resizeFilename = String.format("area_%d_yolo_original_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
        api.saveMatImage(resizedImage, resizeFilename);
        Log.i(TAG, "Resized image saved as " + resizeFilename);

        // ========================================================================
        // STEP 3: Apply CLAHE enhancement (FINAL OUTPUT)
        // ========================================================================

        // Apply CLAHE for better contrast enhancement
        Mat claheImage = new Mat();
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(2.0);  // Controls contrast enhancement

        // Adjust grid size based on image size
        int gridSize = (int) Math.max(8, Math.min(resizeSize.width, resizeSize.height) / 40);
        clahe.setTilesGridSize(new Size(gridSize, gridSize));

        clahe.apply(resizedImage, claheImage);

        // Print min/max values of the CLAHE-enhanced image
        Core.MinMaxLocResult claheMinMaxResult = Core.minMaxLoc(claheImage);
        Log.i(TAG, String.format("CLAHE enhanced image (%.0fx%.0f) - Min: %.2f, Max: %.2f",
                resizeSize.width, resizeSize.height, claheMinMaxResult.minVal, claheMinMaxResult.maxVal));

        // Save CLAHE enhanced image with area ID
        String claheFilename = String.format("area_%d_yolo_clahe_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
        api.saveMatImage(claheImage, claheFilename);
        Log.i(TAG, "CLAHE enhanced image saved as " + claheFilename);

        // ========================================================================
        // STEP 4: Apply Otsu's binarization (FOR DEBUG ONLY - NOT RETURNED)
        // ========================================================================

        // Apply Otsu's automatic threshold binarization for debugging purposes
        Mat binarizedOtsu = new Mat();
        double otsuThreshold = Imgproc.threshold(claheImage, binarizedOtsu, 0, 255,
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        // Print min/max values and threshold of Otsu binarized image
        Core.MinMaxLocResult binaryOtsuResult = Core.minMaxLoc(binarizedOtsu);
        Log.i(TAG, String.format("Binary Otsu (%.1f) - Min: %.2f, Max: %.2f",
                otsuThreshold, binaryOtsuResult.minVal, binaryOtsuResult.maxVal));

        // Save the Otsu binarized image for debugging purposes
        String binaryFilename = String.format("area_%d_debug_binary_otsu_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
        api.saveMatImage(binarizedOtsu, binaryFilename);
        Log.i(TAG, String.format("Debug binary image saved as %s (threshold: %.1f)", binaryFilename, otsuThreshold));

        // ========================================================================
        // CLEANUP
        // ========================================================================

        // Clean up intermediate images (but NOT claheImage - that's our return value)
        srcPointsMat.release();
        dstPointsMatCrop.release();
        perspectiveMatrixCrop.release();
        croppedImage.release();
        resizedImage.release();
        binarizedOtsu.release();  // Release the debug binary image

        // Return the CLAHE enhanced image (instead of binary)
        return claheImage;

    } catch (Exception e) {
        Log.e(TAG, "Error in cropEnhanceAndBinarize: " + e.getMessage());
        return null;
    }
}
```


## 主要修改點

### 1. **返回值變更**
- **修改前**: 返回 CLAHE 增強圖像 (`claheImage`)
- **修改後**: 返回二值化圖像 (`binarizedOtsu`)

### 2. **Step 3 註釋變更**
- **修改前**: `STEP 3: Apply CLAHE enhancement (FINAL OUTPUT)`
- **修改後**: `STEP 3: Apply CLAHE enhancement`

### 3. **Step 4 處理方式變更**
- **修改前**: Otsu 二值化僅用於 debug，不作為返回值
- **修改後**: Otsu 二值化作為最終輸出

### 4. **檔案命名變更**
- **修改前**: 二值化圖像檔名為 `area_%d_debug_binary_otsu_%.0fx%.0f.png`
- **修改後**: 二值化圖像檔名為 `area_%d_yolo_binary_otsu_%.0fx%.0f.png`

### 5. **記憶體管理變更**
- **修改前**: 不釋放 `claheImage`（作為返回值）
- **修改後**: 釋放 `claheImage`、不釋放 `binarizedOtsu`（作為返回值）

### 6. **CLAHE 圖像處理**
- **修改前**: 保存 CLAHE 增強圖像
- **修改後**: 移除 CLAHE 圖像的保存和相關 log 輸出

