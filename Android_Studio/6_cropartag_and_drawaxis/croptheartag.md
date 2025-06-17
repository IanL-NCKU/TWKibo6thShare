# 📦 **程式碼**
```java
package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

//import org.opencv.core.Mat;

// new imports
import android.util.Log;

import java.util.List;
import java.util.ArrayList;


// new OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
// OpenCV Core imports
//import org.opencv.core.CvType;
//import org.opencv.core.Scalar;
//import org.opencv.core.Rect;
//import org.opencv.core.Point3;
//import org.opencv.core.MatOfPoint3f;
//import org.opencv.core.MatOfPoint2f;
//import org.opencv.core.MatOfDouble;

// OpenCV Image Processing
import org.opencv.imgproc.Imgproc;

// OpenCV Calibration
import org.opencv.calib3d.Calib3d;
import gov.nasa.arc.astrobee.Result;
/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

    // The TAG is used for logging.
    // You can use it to check the log in the Android Studio.
    private final String TAG = this.getClass().getSimpleName();

    @Override
    protected void runPlan1(){

        // Log the start of the mission.
        Log.i(TAG, "Start mission");
        // The mission starts.
        api.startMission();
        

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image = api.getMatNavCam();
        // Save the image to a file.
        api.saveMatImage(image, "test.png");


        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // 
        /**
         * Retrieves a predefined Aruco dictionary for 6x6 markers containing 250 distinct patterns.
         * This dictionary is used for detecting and tracking Aruco markers in images.
         *
         * The call to Aruco.getPredefinedDictionary(Aruco.DICT_6X6_250) selects a standard set of marker patterns,
         * making it easier to consistently identify markers during image processing.
         */
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        
        // Detect markers in the image using the specified dictionary.
        // The detectMarkers function analyzes the image and identifies the locations of Aruco markers.
        // The detected markers are stored in the corners list.
        // The corners list contains the coordinates of the detected markers in the image.
        
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        // The ids list contains the IDs of the detected markers.
        Aruco.detectMarkers(image, dictionary, corners, ids);

        if (corners.size() > 0) {
            // Get camera parameters
            double[][] intrinsics = api.getNavCamIntrinsics();
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            
            // Fill camera matrix
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);
            distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

            
            // Estimate pose for each marker
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            
            // Marker size in meters (adjust according to your actual marker size)
            float markerLength = 0.05f; // 5cm markers
            
            Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
            
            // Draw markers and coordinate frames
            Mat imageWithFrame = image.clone();
            Mat imagewithcroparea = image.clone();

            Aruco.drawDetectedMarkers(imageWithFrame, corners, ids);
            
            // type the number of corners in log
            Log.i(TAG, "Detected " + corners.size() + " markers.");
            
            // Draw coordinate frame for each marker
            for (int i = 0; i < corners.size(); i++) {
                
                // extract the corners name variable "currentCorners"
                Mat currentCorners = corners.get(i);
                

                Mat UndistortImg = new Mat();

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = new Mat(3, 1, CvType.CV_64F);
                    Mat tvec = new Mat(3, 1, CvType.CV_64F);
                    
                    rvecs.row(i).copyTo(rvec);
                    tvecs.row(i).copyTo(tvec);
                    
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    // Simply draw the coordinate frame
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f); // 0.1m axis length

                    // corners.get(i) is Mat we need to convert it to MatOfPoint2f
                    MatOfPoint2f cornerPoints = new MatOfPoint2f(currentCorners);

                    // check corner[0] and crop region
                    checkCornerAndCropRegion(imagewithcroparea, cameraMatrix, distCoeffs, rvec, tvec, cornerPoints);
                    api.saveMatImage(imagewithcroparea, "marker_" + i + "_crop_area.png");
                    // drawCropArea(imagewithcroparea, corners.get(i).toArray());


                    // Release individual vectors
                    cornerPoints.release();
                    rvec.release();
                    tvec.release();
                }


                // Extract rotation and translation vectors for this marker
                // rvecs.row(i).copyTo(rvec);
                // tvecs.row(i).copyTo(tvec);

                // Draw coordinate frame (X=red, Y=green, Z=blue)
                // drawCoordinateFrame(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, markerLength);
                
                // Crop region around the marker
                Mat croppedRegion = cropMarkerRegion(image, corners.get(i));
                
                // Save cropped image for debugging
                api.saveMatImage(croppedRegion, "marker_" + i + "_cropped.png"); 
                // see what color type of imageWithFrame is
                Log.i(TAG, "Image with frame type: " + imageWithFrame.type());

                // make imageWithFrame from GRAY to RGB
                
                // Save image with drawn frames
                api.saveMatImage(imageWithFrame, "marker_" + i + "_with_frame.png");

                
                Calib3d.undistort(image, UndistortImg, cameraMatrix, distCoeffs);

                // Save undistorted image for debugging
                api.saveMatImage(UndistortImg, "marker_" + i + "_undistorted.png");


                // Clean up individual vectors
                // rvec.release();
                // tvec.release();
                croppedRegion.release();
                UndistortImg.release();
                imageWithFrame.release();

            }

            // Clean up
            rvecs.release();
            tvecs.release();
            // UndistortImg.release();
            cameraMatrix.release();
            distCoeffs.release();
        }

        // Clean up
        ids.release();
        for (Mat corner : corners) {
            corner.release();
        }
        
        
        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(1, "item_name", 1);

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */
        /* **************************************************** */

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();

        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        /* ********************************************************** */

        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();

        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */

        // Take a snapshot of the target item.
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2(){
       // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }



    private void checkCornerAndCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, 
                                        Mat rvec, Mat tvec, MatOfPoint2f corners) {
        try {
            // 1. Check if corner[0] is close to expected 3D position (-0.025, 0.025, 0)
            org.opencv.core.Point[] cornerPoints = corners.toArray();
            if (cornerPoints.length > 0) {
                // Project expected 3D point to 2D
                org.opencv.core.Point3[] expectedPoint3D = {new org.opencv.core.Point3(-0.025, 0.025, 0)};
                MatOfPoint3f expectedPointMat = new MatOfPoint3f(expectedPoint3D);
                MatOfPoint2f projectedExpected = new MatOfPoint2f();
                
                // Convert distortion coefficients
                double[] distData = new double[5];
                distCoeffs.get(0, 0, distData);
                MatOfDouble distCoeffsDouble = new MatOfDouble();
                distCoeffsDouble.fromArray(distData);
                
                // Project expected point
                Calib3d.projectPoints(expectedPointMat, rvec, tvec, cameraMatrix, distCoeffsDouble, projectedExpected);
                org.opencv.core.Point[] projectedPoints = projectedExpected.toArray();
                
                if (projectedPoints.length > 0) {
                    org.opencv.core.Point expectedCorner = projectedPoints[0];
                    org.opencv.core.Point actualCorner = cornerPoints[0];
                    
                    // Calculate distance between expected and actual corner
                    // double distance = Math.sqrt(Math.pow(expectedCorner.x - actualCorner.x, 2) + 
                    //                         Math.pow(expectedCorner.y - actualCorner.y, 2));
                    
                    double distance = getdistance(expectedCorner, actualCorner);
                    // Check if close (within 10 pixels tolerance)
                    if (distance > 10.0) {
                        // Log the real 3D position of corner[0]
                        // To get 3D position, we need to reverse project (assuming z=0 for simplicity)
                        Log.i(TAG, String.format("Corner[0] not at expected position. Detected at 2D: (%.2f, %.2f), Expected 2D: (%.2f, %.2f), Distance: %.2f pixels", 
                                actualCorner.x, actualCorner.y, expectedCorner.x, expectedCorner.y, distance));
                    } else {
                        Log.i(TAG, "Corner[0] is close to expected position (-2.5, 2.5, 0)");
                    }
                }
                
                // Clean up
                expectedPointMat.release();
                projectedExpected.release();
                distCoeffsDouble.release();
            }
            
            // 2. Project the 4 crop area corners to 2D
            // Manually adjustment
            org.opencv.core.Point3[] cropCorners3D = {
                new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
                new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
                new org.opencv.core.Point3(-0.2325, -0.1125, 0), // Bottom-right
                new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
            };
            
            MatOfPoint3f cropCornersMat = new MatOfPoint3f(cropCorners3D);
            MatOfPoint2f cropCorners2D = new MatOfPoint2f();
            
            // Convert distortion coefficients again
            double[] distData = new double[5];
            distCoeffs.get(0, 0, distData);
            MatOfDouble distCoeffsDouble = new MatOfDouble();
            distCoeffsDouble.fromArray(distData);
            
            // Project crop corners to 2D
            Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
            org.opencv.core.Point[] cropPoints2D = cropCorners2D.toArray();
            
            if (cropPoints2D.length == 4) {
                // 3. Create perspective transformation and crop
                cropAndSaveRegion(image, cropPoints2D);
            }
            
            // Clean up
            cropCornersMat.release();
            cropCorners2D.release();
            distCoeffsDouble.release();
            
        } catch (Exception e) {
            Log.e(TAG, "Error in checkCornerAndCropRegion: " + e.getMessage());
        }
    }

    private void cropAndSaveRegion(Mat image, org.opencv.core.Point[] cropPoints2D) {
        try {
            // Define destination points for 640x480 rectangle
            org.opencv.core.Point[] dstPoints = {
                new org.opencv.core.Point(0, 0),       // Top-left
                new org.opencv.core.Point(639, 0),     // Top-right
                new org.opencv.core.Point(639, 479),   // Bottom-right
                new org.opencv.core.Point(0, 479)      // Bottom-left
            };
            
            // Create source and destination point matrices
            MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
            MatOfPoint2f dstPointsMat = new MatOfPoint2f(dstPoints);
            
            // Calculate perspective transformation matrix
            Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
            
            // Apply perspective transformation
            Mat croppedImage = new Mat();
            Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
            
            // Save the cropped image
            api.saveMatImage(croppedImage, "cropped_region_640x480.png");
            Log.i(TAG, "Cropped region saved as 640x480 image");
            
            // Optional: Draw the crop area on original image for visualization
            drawCropArea(image, cropPoints2D);
            
            // Clean up
            srcPointsMat.release();
            dstPointsMat.release();
            perspectiveMatrix.release();
            croppedImage.release();
            
        } catch (Exception e) {
            Log.e(TAG, "Error cropping region: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------

    /**
     * Crops a region around the detected marker with some padding
     */
    private Mat cropMarkerRegion(Mat image, Mat markerCorners) {
        // Get the four corners of the marker
        float[] cornerData = new float[(int)(markerCorners.total() * markerCorners.channels())];
        markerCorners.get(0, 0, cornerData);
        
        // Find bounding rectangle
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;
        
        for (int i = 0; i < cornerData.length; i += 2) {
            float x = cornerData[i];
            float y = cornerData[i + 1];
            
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
        
        // Add padding (20% of marker size)
        float padding = Math.max(maxX - minX, maxY - minY) * 0.2f;
        
        int x = Math.max(0, (int)(minX - padding));
        int y = Math.max(0, (int)(minY - padding));
        int width = Math.min(image.cols() - x, (int)(maxX - minX + 2 * padding));
        int height = Math.min(image.rows() - y, (int)(maxY - minY + 2 * padding));
        
        // Create rectangle and crop
        Rect cropRect = new Rect(x, y, width, height);
        Mat croppedImage = new Mat(image, cropRect);
        
        return croppedImage.clone();
    }

    private void drawCropArea(Mat image, org.opencv.core.Point[] cropPoints2D) {
        // turn the image to RGB
        Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2RGB);
        // Draw the crop area outline on the original image
        for (int i = 0; i < cropPoints2D.length; i++) {
            org.opencv.core.Point pt1 = cropPoints2D[i];
            org.opencv.core.Point pt2 = cropPoints2D[(i + 1) % cropPoints2D.length];
            Imgproc.line(image, pt1, pt2, new Scalar(255, 255, 0), 2); // Yellow lines
            // input to log
            Log.i(TAG, String.format("Crop Point %d: (%.2f, %.2f)", i, pt1.x, pt1.y));
        }
        
        // Add corner labels
        for (int i = 0; i < cropPoints2D.length; i++) {
            Imgproc.putText(image, String.valueOf(i), cropPoints2D[i], 
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 0), 2);
        }
    }

    private Result sureMoveToPoint(Point point, Quaternion quaternion, boolean printRobotPosition, int maxRetries) {
        Result result = api.moveTo(point, quaternion, printRobotPosition);
        
        int retryCount = 0;
        while (!result.hasSucceeded() && retryCount < maxRetries) {
            result = api.moveTo(point, quaternion, true); // Use true for retries
            retryCount++;
        }
        
        return result;
    }


    private double getdistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        // Calculate the distance between two points using the Euclidean distance formula
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }
}

```


## 📦 **程式包與匯入模組**

### **基礎匯入**
```java
package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import android.util.Log;
import java.util.List;
import java.util.ArrayList;
```

**功能說明：**
- 定義程式包名稱
- 匯入Astrobee機器人控制API
- 匯入Android日誌功能
- 匯入Java集合類別，用於儲存檢測到的標記

### **OpenCV電腦視覺函式庫**
```java
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import gov.nasa.arc.astrobee.Result;
```

**功能說明：**
- `Dictionary` - ArUco標記字典
- `Aruco` - ArUco標記檢測功能
- `core.*` - OpenCV核心功能（Mat, Point, Size等）
- `Imgproc` - 影像處理功能（顏色轉換、透視變換等）
- `Calib3d` - 相機標定和3D幾何運算
- `Result` - API呼叫結果

---

## 🚀 **runPlan1() - 主要任務執行方法**

### **任務初始化與機器人定位**
```java
@Override
protected void runPlan1(){
    Log.i(TAG, "Start mission");
    api.startMission();
    
    Point point = new Point(10.9d, -9.92284d, 5.195d);
    Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
    api.moveTo(point, quaternion, false);
    
    Mat image = api.getMatNavCam();
    api.saveMatImage(image, "test.png");
}
```

**功能說明：**
- 記錄任務開始日誌並啟動任務
- 移動機器人到初始觀察位置
- 獲取導航相機影像並儲存為測試檔案

### **ArUco標記檢測核心邏輯**
```java
Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
List<Mat> corners = new ArrayList<>();
Mat ids = new Mat();
Aruco.detectMarkers(image, dictionary, corners, ids);
```

**功能說明：**
- 建立5x5大小的ArUco標記字典，包含250個不同的標記圖案
- 在影像中檢測ArUco標記
- `corners` 儲存檢測到的標記角點座標
- `ids` 儲存標記的ID編號

### **相機參數設定與姿態估計**
```java
if (corners.size() > 0) {
    double[][] intrinsics = api.getNavCamIntrinsics();
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
    
    cameraMatrix.put(0, 0, intrinsics[0]);
    distCoeffs.put(0, 0, intrinsics[1]);
    
    Mat rvecs = new Mat();
    Mat tvecs = new Mat();
    float markerLength = 0.05f; // 5cm markers
    
    Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
}
```

**功能說明：**
- 檢查是否檢測到標記
- 獲取相機內參數（焦距、主點、畸變係數）
- 建立相機矩陣和畸變係數矩陣
- 估計每個標記的3D姿態（旋轉和平移向量）

### **標記處理與可視化**
```java
for (int i = 0; i < corners.size(); i++) {
    Mat currentCorners = corners.get(i);
    
    if (rvecs.rows() > 0 && tvecs.rows() > 0) {
        Mat rvec = new Mat(3, 1, CvType.CV_64F);
        Mat tvec = new Mat(3, 1, CvType.CV_64F);
        
        rvecs.row(i).copyTo(rvec);
        tvecs.row(i).copyTo(tvec);
        
        Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
        Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
        
        MatOfPoint2f cornerPoints = new MatOfPoint2f(currentCorners);
        checkCornerAndCropRegion(imagewithcroparea, cameraMatrix, distCoeffs, rvec, tvec, cornerPoints);
    }
}
```

**功能說明：**
- 遍歷每個檢測到的標記
- 提取當前標記的旋轉和平移向量
- 轉換影像為RGB格式並繪製3D座標軸
- 呼叫精密裁剪方法處理標記區域

---

## 🔧 **checkCornerAndCropRegion() - 檢查角點位置並裁剪區域**

```java
private void checkCornerAndCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, 
                                    Mat rvec, Mat tvec, MatOfPoint2f corners) {
    try {
        // 1. 驗證角點位置精度
        org.opencv.core.Point[] cornerPoints = corners.toArray();
        if (cornerPoints.length > 0) {
            org.opencv.core.Point3[] expectedPoint3D = {new org.opencv.core.Point3(-0.025, 0.025, 0)};
            MatOfPoint3f expectedPointMat = new MatOfPoint3f(expectedPoint3D);
            MatOfPoint2f projectedExpected = new MatOfPoint2f();
            
            Calib3d.projectPoints(expectedPointMat, rvec, tvec, cameraMatrix, distCoeffsDouble, projectedExpected);
            
            double distance = getdistance(expectedCorner, actualCorner);
            if (distance > 10.0) {
                Log.i(TAG, "Corner[0] not at expected position...");
            } else {
                Log.i(TAG, "Corner[0] is close to expected position (-2.5, 2.5, 0)");
            }
        }
        
        // 2. 定義裁剪區域的3D角點
        org.opencv.core.Point3[] cropCorners3D = {
            new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
            new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
            new org.opencv.core.Point3(-0.2325, -0.1125, 0),  // Bottom-right
            new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
        };
        
        // 3. 投影到2D並執行裁剪
        Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
        cropAndSaveRegion(image, cropPoints2D);
        
    } catch (Exception e) {
        Log.e(TAG, "Error in checkCornerAndCropRegion: " + e.getMessage());
    }
}
```

**功能說明：**
- **精度驗證**：檢查檢測到的角點是否接近預期的3D位置（-2.5cm, 2.5cm, 0）
- **3D到2D投影**：將預期的3D座標點投影到2D影像座標進行比較
- **裁剪區域定義**：定義四個3D角點作為裁剪區域的邊界
- **座標轉換**：將3D裁剪區域投影到2D影像座標
- **錯誤處理**：捕獲並記錄處理過程中的異常

---

## ✂️ **cropAndSaveRegion() - 裁剪並保存區域**

```java
private void cropAndSaveRegion(Mat image, org.opencv.core.Point[] cropPoints2D) {
    try {
        // 定義目標矩形（640x480像素）
        org.opencv.core.Point[] dstPoints = {
            new org.opencv.core.Point(0, 0),       // Top-left
            new org.opencv.core.Point(639, 0),     // Top-right
            new org.opencv.core.Point(639, 479),   // Bottom-right
            new org.opencv.core.Point(0, 479)      // Bottom-left
        };
        
        // 計算透視變換矩陣
        MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
        MatOfPoint2f dstPointsMat = new MatOfPoint2f(dstPoints);
        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
        
        // 執行透視變換
        Mat croppedImage = new Mat();
        Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
        
        // 保存裁剪後的影像
        api.saveMatImage(croppedImage, "cropped_region_640x480.png");
        Log.i(TAG, "Cropped region saved as 640x480 image");
        
        // 可視化裁剪區域
        drawCropArea(image, cropPoints2D);
        
    } catch (Exception e) {
        Log.e(TAG, "Error cropping region: " + e.getMessage());
    }
}
```

**功能說明：**
- **目標矩形定義**：設定640x480像素的標準矩形作為輸出格式
- **透視變換矩陣計算**：根據源點和目標點計算變換矩陣
- **影像矯正**：執行透視變換，將傾斜的影像區域矯正為正矩形
- **結果保存**：將處理後的影像保存為檔案
- **可視化**：在原始影像上繪製裁剪區域邊界

---

## 📐 **cropMarkerRegion() - 裁剪標記區域**

```java
private Mat cropMarkerRegion(Mat image, Mat markerCorners) {
    // 提取標記角點數據
    float[] cornerData = new float[(int)(markerCorners.total() * markerCorners.channels())];
    markerCorners.get(0, 0, cornerData);
    
    // 找出包圍標記的最小矩形
    float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
    float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;
    
    for (int i = 0; i < cornerData.length; i += 2) {
        float x = cornerData[i];
        float y = cornerData[i + 1];
        
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }
    
    // 添加20%的邊距
    float padding = Math.max(maxX - minX, maxY - minY) * 0.2f;
    
    int x = Math.max(0, (int)(minX - padding));
    int y = Math.max(0, (int)(minY - padding));
    int width = Math.min(image.cols() - x, (int)(maxX - minX + 2 * padding));
    int height = Math.min(image.rows() - y, (int)(maxY - minY + 2 * padding));
    
    // 建立裁剪矩形並執行裁剪
    Rect cropRect = new Rect(x, y, width, height);
    Mat croppedImage = new Mat(image, cropRect);
    
    return croppedImage.clone();
}
```

**功能說明：**
- **角點數據提取**：從Mat格式中提取標記的四個角點座標
- **邊界矩形計算**：找出包圍所有角點的最小矩形
- **邊距添加**：為裁剪區域添加20%的邊距以確保完整性
- **邊界檢查**：確保裁剪區域不超出影像邊界
- **矩形裁剪**：建立裁剪矩形並返回裁剪後的影像

---

## 🎨 **drawCropArea() - 繪製裁剪區域**

```java
private void drawCropArea(Mat image, org.opencv.core.Point[] cropPoints2D) {
    // 轉換影像為RGB格式
    Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2RGB);
    
    // 繪製裁剪區域輪廓
    for (int i = 0; i < cropPoints2D.length; i++) {
        org.opencv.core.Point pt1 = cropPoints2D[i];
        org.opencv.core.Point pt2 = cropPoints2D[(i + 1) % cropPoints2D.length];
        Imgproc.line(image, pt1, pt2, new Scalar(255, 255, 0), 2); // 黃色線條
        Log.i(TAG, String.format("Crop Point %d: (%.2f, %.2f)", i, pt1.x, pt1.y));
    }
    
    // 添加角點標籤
    for (int i = 0; i < cropPoints2D.length; i++) {
        Imgproc.putText(image, String.valueOf(i), cropPoints2D[i], 
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 0), 2);
    }
}
```

**功能說明：**
- **格式轉換**：將灰度影像轉換為RGB格式以支援彩色繪製
- **輪廓繪製**：用黃色線條連接四個裁剪區域角點形成封閉輪廓
- **座標記錄**：將每個角點的座標記錄到日誌中
- **標籤添加**：在每個角點位置添加數字標籤便於識別

---

## 🎯 **sureMoveToPoint() - 確保移動到指定點**

```java
private Result sureMoveToPoint(Point point, Quaternion quaternion, boolean printRobotPosition, int maxRetries) {
    Result result = api.moveTo(point, quaternion, printRobotPosition);
    
    int retryCount = 0;
    while (!result.hasSucceeded() && retryCount < maxRetries) {
        result = api.moveTo(point, quaternion, true); // 重試時使用true
        retryCount++;
    }
    
    return result;
}
```

**功能說明：**
- **初始移動**：首次嘗試移動到指定位置
- **失敗重試**：如果移動失敗，自動重試直到成功或達到最大重試次數
- **狀態追蹤**：追蹤重試次數避免無限迴圈
- **結果回傳**：回傳最終的移動結果

---

## 📏 **getdistance() - 計算兩點距離**

```java
private double getdistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}
```

**功能說明：**
- **歐幾里得距離計算**：使用標準的距離公式計算兩點間的直線距離
- **精度驗證支援**：主要用於驗證檢測到的角點與預期位置的偏差
- **簡潔實現**：提供簡單易用的距離計算工具方法

---

## 📊 **程式整體流程總結**

1. **初始化** → 啟動任務，移動到觀察位置
2. **影像獲取** → 從導航相機獲取影像
3. **標記檢測** → 使用ArUco檢測標記
4. **姿態估計** → 計算標記的3D位置和方向
5. **精度驗證** → 檢查檢測精度
6. **影像裁剪** → 精確提取標記周圍區域
7. **資料處理** → 為後續物品識別準備數據



&nbsp;

&nbsp;





# 📦 **程式逐行解析**

```java
package jp.jaxa.iss.kibo.rpc.sampleapk;
```
**第1行：** 定義程式包名

## **基礎匯入**
```java
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
```
**第3-5行：** 匯入Astrobee機器人控制API

```java
import android.util.Log;
```
**第10行：** Android日誌功能

```java
import java.util.List;
import java.util.ArrayList;
```
**第12-13行：** Java集合類別，用於儲存檢測到的標記

## **OpenCV電腦視覺函式庫**
```java
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
```
**第16-18行：** 
- `Dictionary` - ArUco標記字典
- `Aruco` - ArUco標記檢測功能
- `core.*` - OpenCV核心功能（Mat, Point, Size等）

```java
import org.opencv.imgproc.Imgproc;
```
**第26行：** 影像處理功能（顏色轉換、透視變換等）

```java
import org.opencv.calib3d.Calib3d;
import gov.nasa.arc.astrobee.Result;
```
**第29-30行：** 
- `Calib3d` - 相機標定和3D幾何運算
- `Result` - API呼叫結果

## 🚀 **主要任務執行（runPlan1方法）**

### **任務初始化**
```java
Log.i(TAG, "Start mission");
api.startMission();
```
**第44-46行：** 記錄日誌並啟動任務

### **機器人定位**
```java
Point point = new Point(10.9d, -9.92284d, 5.195d);
Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
api.moveTo(point, quaternion, false);
```
**第49-51行：** 移動到初始觀察位置

### **影像獲取**
```java
Mat image = api.getMatNavCam();
api.saveMatImage(image, "test.png");
```
**第54-56行：** 獲取相機影像並儲存

## 🎯 **ArUco標記檢測核心邏輯**

### **建立ArUco字典**
```java
Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
```
**第67行：** 
- 建立5x5大小的ArUco標記字典
- 包含250個不同的標記圖案
- 用於識別和追蹤標記

## **檢測標記**
```java
List<Mat> corners = new ArrayList<>();
Mat ids = new Mat();
Aruco.detectMarkers(image, dictionary, corners, ids);
```
**第73-75行：** 
- `corners` - 儲存檢測到的標記角點座標
- `ids` - 儲存標記的ID編號
- `detectMarkers` - 在影像中檢測ArUco標記

## **相機參數設定**
```java
if (corners.size() > 0) {
    double[][] intrinsics = api.getNavCamIntrinsics();
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
```
**第77-80行：** 
- 檢查是否檢測到標記
- 獲取相機內參數（焦距、主點、畸變係數）
- 建立相機矩陣和畸變係數矩陣

```java
cameraMatrix.put(0, 0, intrinsics[0]);
distCoeffs.put(0, 0, intrinsics[1]);
distCoeffs.convertTo(distCoeffs, CvType.CV_64F);
```
**第82-84行：** 
- 填入相機內參數
- 轉換為64位元浮點數格式

## **姿態估計**
```java
Mat rvecs = new Mat();
Mat tvecs = new Mat();
float markerLength = 0.05f; // 5cm markers
Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
```
**第87-91行：** 
- `rvecs` - 旋轉向量（標記方向）
- `tvecs` - 平移向量（標記位置）
- `markerLength` - 實際標記大小（5公分）
- 估計每個標記的3D姿態

### **影像處理與可視化**
```java
Mat imageWithFrame = image.clone();
Mat imagewithcroparea = image.clone();
Aruco.drawDetectedMarkers(imageWithFrame, corners, ids);
Log.i(TAG, "Detected " + corners.size() + " markers.");
```
**第94-98行：** 
- 複製原始影像
- 在影像上繪製檢測到的標記
- 記錄檢測到的標記數量

## 🔄 **標記處理迴圈**

```java
for (int i = 0; i < corners.size(); i++) {
    Mat currentCorners = corners.get(i);
    Mat UndistortImg = new Mat();
```
**第101-105行：** 遍歷每個檢測到的標記

### **姿態向量處理**
```java
if (rvecs.rows() > 0 && tvecs.rows() > 0) {
    Mat rvec = new Mat(3, 1, CvType.CV_64F);
    Mat tvec = new Mat(3, 1, CvType.CV_64F);
    
    rvecs.row(i).copyTo(rvec);
    tvecs.row(i).copyTo(tvec);
```
**第107-112行：** 
- 檢查姿態數據是否有效
- 提取當前標記的旋轉和平移向量

### **座標軸繪製**
```java
Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
```
**第114-116行：** 
- 轉換影像為RGB格式
- 繪製3D座標軸（Z=紅, Y=綠, X=藍）

### **精密裁剪處理**
```java
MatOfPoint2f cornerPoints = new MatOfPoint2f(currentCorners);
checkCornerAndCropRegion(imagewithcroparea, cameraMatrix, distCoeffs, rvec, tvec, cornerPoints);
api.saveMatImage(imagewithcroparea, "marker_" + i + "_crop_area.png");
```
**第118-121行：** 
- 轉換角點格式
- 呼叫精密裁剪方法
- 儲存裁剪區域影像

## 🔧 **輔助方法詳細解析**

### **checkCornerAndCropRegion方法**

```java
private void checkCornerAndCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, 
                                    Mat rvec, Mat tvec, MatOfPoint2f corners) {
```
**第194行：** 檢查角點位置並裁剪區域

#### **3D到2D投影驗證**
```java
org.opencv.core.Point3[] expectedPoint3D = {new org.opencv.core.Point3(-0.025, 0.025, 0)};
MatOfPoint3f expectedPointMat = new MatOfPoint3f(expectedPoint3D);
MatOfPoint2f projectedExpected = new MatOfPoint2f();
```
**第201-203行：** 
- 定義預期的3D座標點（-2.5cm, 2.5cm, 0）
- 準備投影計算

```java
Calib3d.projectPoints(expectedPointMat, rvec, tvec, cameraMatrix, distCoeffsDouble, projectedExpected);
```
**第211行：** 將3D點投影到2D影像座標

#### **精度驗證**
```java
double distance = getdistance(expectedCorner, actualCorner);
if (distance > 10.0) {
    Log.i(TAG, String.format("Corner[0] not at expected position..."));
} else {
    Log.i(TAG, "Corner[0] is close to expected position (-2.5, 2.5, 0)");
}
```
**第218-225行：** 
- 計算預期位置與實際位置的距離
- 如果誤差超過10像素則記錄警告

#### **裁剪區域定義**
```java
org.opencv.core.Point3[] cropCorners3D = {
    new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
    new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
    new org.opencv.core.Point3(-0.2325, -0.1125, 0), // Bottom-right
    new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
};
```
**第234-239行：** 定義裁剪區域的四個3D角點（單位：公尺）

### **cropAndSaveRegion方法**

```java
org.opencv.core.Point[] dstPoints = {
    new org.opencv.core.Point(0, 0),       // Top-left
    new org.opencv.core.Point(639, 0),     // Top-right
    new org.opencv.core.Point(639, 479),   // Bottom-right
    new org.opencv.core.Point(0, 479)      // Bottom-left
};
```
**第260-265行：** 定義目標矩形（640x480像素）

```java
Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
```
**第272-276行：** 
- 計算透視變換矩陣
- 執行透視變換，矯正傾斜的影像區域

### **cropMarkerRegion方法**

```java
float[] cornerData = new float[(int)(markerCorners.total() * markerCorners.channels())];
markerCorners.get(0, 0, cornerData);
```
**第295-296行：** 提取標記角點數據

```java
for (int i = 0; i < cornerData.length; i += 2) {
    float x = cornerData[i];
    float y = cornerData[i + 1];
    
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
}
```
**第301-309行：** 找出包圍標記的最小矩形

```java
float padding = Math.max(maxX - minX, maxY - minY) * 0.2f;
```
**第312行：** 添加20%的邊距

### **sureMoveToPoint方法**

```java
private Result sureMoveToPoint(Point point, Quaternion quaternion, boolean printRobotPosition, int maxRetries) {
    Result result = api.moveTo(point, quaternion, printRobotPosition);
    
    int retryCount = 0;
    while (!result.hasSucceeded() && retryCount < maxRetries) {
        result = api.moveTo(point, quaternion, true);
        retryCount++;
    }
    return result;
}
```
**第365-375行：** 
- 可靠的移動方法
- 如果移動失敗會自動重試
- 最多重試maxRetries次

## 📊 **程式整體流程總結**

1. **初始化** → 啟動任務，移動到觀察位置
2. **影像獲取** → 從導航相機獲取影像
3. **標記檢測** → 使用ArUco檢測標記
4. **姿態估計** → 計算標記的3D位置和方向
5. **精度驗證** → 檢查檢測精度
6. **影像裁剪** → 精確提取標記周圍區域
7. **資料處理** → 為後續物品識別準備數據
