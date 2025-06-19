package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Result;

import android.util.Log;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.photo.Photo;

public class YourService extends KiboRpcService {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private final String TAG = this.getClass().getSimpleName();

    // Instance variables to store detection results across areas
    private Set<String> foundTreasures = new HashSet<>();
    private Set<String> foundLandmarks = new HashSet<>(); // Add this line
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<>();

    // Area coordinates and orientations for all 4 areas
    private final Point[] AREA_POINTS = {
            new Point(10.95d, -9.78d, 5.195d), // Area 1
            new Point(10.925d, -8.35d, 4.56203d), // Area 2
            new Point(10.925d, -7.925d, 4.56093d), // Area 3
            new Point(10.666984d, -6.8525d, 4.945d) // Area 4
    };

    private final Quaternion[] AREA_QUATERNIONS = {
            new Quaternion(0f, 0f, -0.707f, 0.707f), // Area 1
            new Quaternion(0f, 0.707f, 0f, 0.707f), // Area 2
            new Quaternion(0f, 0.707f, 0f, 0.707f), // Area 3
            new Quaternion(0f, 0f, 1f, 0f) // Area 4
    };

    // 在类成员变量区域添加
    private Set<String> reportedLandmarks = new HashSet<>();
    private final String[] ALL_POSSIBLE_LANDMARKS = { "red_bull", "chips", "battery", "pickel", "lemon", "candy",
            "wine", "coin" };
    private Map<Integer, Set<String>> areaLandmarkTypes = new HashMap<>();

    @Override
    protected void runPlan1() {
        // Log the start of the mission.
        Log.i(TAG, "Start mission");

        // The mission starts.
        api.startMission();

        // Initialize area treasure tracking
        Map<Integer, Set<String>> areaTreasure = new HashMap<>();
        for (int i = 1; i <= 4; i++) {
            areaTreasure.put(i, new HashSet<String>());
            areaLandmarkTypes.put(i, new HashSet<>()); // 添加此行初始化 areaLandmarkTypes
        }

        // ========================================================================
        // CONFIGURABLE IMAGE PROCESSING PARAMETERS - EDIT HERE
        // ========================================================================

        Size cropWarpSize = new Size(640, 480); // Size for cropped/warped image
        Size resizeSize = new Size(320, 320); // Size for final processing

        // ========================================================================
        // PROCESS ALL 4 AREAS
        // ========================================================================

        // Loop through all 4 areas
        for (int areaIndex = 0; areaIndex < 4; areaIndex++) {
            int areaId = areaIndex + 1; // Area IDs are 1, 2, 3, 4

            Log.i(TAG, "=== Processing Area " + areaId + " ===");

            // 区域2特殊处理 - 同时处理区域2和区域3
            if (areaId == 2) {
                processAreaWithDualTags(areaId, areaTreasure);
                continue; // 继续下一个循环，跳过普通处理
            }

            // 区域3特殊处理 - 如果区域2时已经处理过，则跳过
            if (areaId == 3 && areaLandmarks.containsKey("area3")) {
                Log.i(TAG, "区域3已在处理区域2时提前处理，跳过");
                continue; // 跳过处理，因为在区域2时已经处理过了
            }

            // Move to the area
            Point targetPoint = AREA_POINTS[areaIndex];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaIndex];

            Log.i(TAG, String.format("Moving to Area %d: Point(%.3f, %.3f, %.3f)",
                    areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            switch (areaIndex) {
                case 0:
                    api.moveTo(new Point(10.925d, -9.85d, 4.695d), targetQuaternion, false); // Oasis1 cnter
                    api.moveTo(new Point(10.95d, -9.78d, 4.945d), targetQuaternion, false); // Scan point1
                    break;
                case 1:
                    api.moveTo(new Point(10.938d, -9.5d, 4.945d), targetQuaternion, false); // Oasis1&2 cnter
                    // api.moveTo(new Point(10.925d, -8.875d, 4.94d), targetQuaternion, false); //
                    // Scan point2
                    api.moveTo(new Point(11.175d, -8.975d, 5.195d), targetQuaternion, false); // Oasis2 cnter
                    api.moveTo(new Point(10.925d, -8.3d, 4.94d), targetQuaternion, false); // Scan point2
                    break;
                case 2:
                    // api.moveTo(new Point(11.175d, -8.975d, 5.195d), targetQuaternion, false); //
                    // Oasis2 cnter
                    // api.moveTo(new Point(10.7d, -7.925d, 5.195d), targetQuaternion, false); //
                    // Oasis3 cnter
                    // api.moveTo(new Point(10.925d, -7.925d, 4.94d), targetQuaternion, false); //
                    // Scan point3
                    break;
                case 3:
                    api.moveTo(new Point(10.7d, -7.925d, 5.195d), targetQuaternion, false); // Oasis3 cnter
                    api.moveTo(new Point(10.925d, -7.4d, 4.945d), targetQuaternion, false); // Oasis3&4 cnter
                    api.moveTo(new Point(11.175d, -6.875d, 4.685d), targetQuaternion, false); // Oasis4 cnter
                    api.moveTo(new Point(10.925d, -6.8525d, 4.945d), targetQuaternion, false);// Scan point4
                    break;
                default:
                    break;
            }

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace(); // 處理異常
            }

            // Get a camera image
            Mat image = api.getMatNavCam();

            // Process the image for this area (may contain multiple markers)
            List<Mat> claHeBinImages = imageEnhanceAndCrop(image, cropWarpSize, resizeSize, areaId);

            // Initialize detection results for this area
            Map<String, Integer> landmark_items = new HashMap<>();
            Set<String> treasure_types = new HashSet<>();

            if (claHeBinImages != null && !claHeBinImages.isEmpty()) {
                Log.i(TAG, "Area " + areaId + ": Image enhancement and cropping successful for " + claHeBinImages.size()
                        + " markers");

                // Detect items on each cropped image
                for (Mat claHeBinImage : claHeBinImages) {
                    Object[] detected_items = detectitemfromcvimg(
                            claHeBinImage,
                            0.3f, // conf_threshold
                            "lost", // img_type ("lost" or "target")
                            0.45f, // standard_nms_threshold
                            0.8f, // overlap_nms_threshold
                            320 // img_size
                    );

                    // Extract and merge results
                    Map<String, Integer> singleLandmarks = (Map<String, Integer>) detected_items[0];
                    Set<String> singleTreasures = (Set<String>) detected_items[1];

                    // Custom strategy: if multiple coins detected, report half the count
                    if (landmark_items.containsKey("coin")) {
                        int coinCount = landmark_items.get("coin");
                        if (coinCount > 1) {
                            landmark_items.put("coin", coinCount / 2);
                        }
                    }

                    for (Map.Entry<String, Integer> entry : singleLandmarks.entrySet()) {
                        landmark_items.put(entry.getKey(),
                                landmark_items.getOrDefault(entry.getKey(), 0) + entry.getValue());
                    }
                    treasure_types.addAll(singleTreasures);

                }

                Log.i(TAG, "Area " + areaId + " - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Area " + areaId + " - Treasure types: " + treasure_types);

                // Store results for later use
                areaLandmarks.put("area" + areaId, landmark_items);
                foundTreasures.addAll(treasure_types);

                // Add this line to store landmark types
                foundLandmarks.addAll(landmark_items.keySet());

                // Store treasure types for this area
                areaTreasure.get(areaId).addAll(treasure_types);

                Log.i(TAG, "Area " + areaId + " treasure types: " + areaTreasure.get(areaId));

            } else {
                Log.w(TAG, "Area " + areaId + ": Image enhancement failed - no markers detected or processing error");
            }

            // Clean up original image
            image.release();

            // ========================================================================
            // SET AREA INFO FOR THIS AREA
            // ========================================================================

            // Use the detected landmark items for area info
            String[] firstLandmark = getFirstLandmarkItem(landmark_items);
            if (firstLandmark != null) {
                String currentlandmark_items = firstLandmark[0];
                int landmarkCount = Integer.parseInt(firstLandmark[1]);

                // 记录真实检测到的 Landmark
                reportedLandmarks.add(currentlandmark_items);

                // Set the area info with detected landmarks
                api.setAreaInfo(areaId, currentlandmark_items, landmarkCount);
                Log.i(TAG, String.format("Area %d: %s x %d", areaId, currentlandmark_items, landmarkCount));
            } else {
                Log.w(TAG, "Area " + areaId + ": No landmark items detected, using fallback");
                // 使用备用方案
                String[] fallbackLandmark = getFallbackLandmark();
                api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
                Log.i(TAG, String.format("Area %d (FALLBACK): %s x %d", areaId, fallbackLandmark[0],
                        Integer.parseInt(fallbackLandmark[1])));
            }

            // Short delay between areas to ensure stability
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted");
            }

            // 记录该区域的 Landmark 类型
            recordAreaLandmarkTypes(areaId, landmark_items);
        }

        // ========================================================================
        // LOG SUMMARY OF ALL AREAS
        // ========================================================================

        Log.i(TAG, "=== AREA PROCESSING SUMMARY ===");
        for (int i = 1; i <= 4; i++) {
            Log.i(TAG, "Area " + i + " treasures: " + areaTreasure.get(i));
            Log.i(TAG, "Area " + i + " landmarks: " + areaLandmarks.get("area" + i));
        }
        Log.i(TAG, "All found treasures: " + foundTreasures);
        Log.i(TAG, "All found landmarks: " + foundLandmarks); // Add this line

        // ========================================================================
        // ASTRONAUT INTERACTION
        // ========================================================================

        // Move to the front of the astronaut and report rounding completion
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);

        Log.i(TAG, "Moving to astronaut position");
        api.moveTo(astronautPoint, astronautQuaternion, false);

        api.reportRoundingCompletion();

        // 停止Kibo原地0.5秒
        try {
            Thread.sleep(500); // 暫停0.5秒
        } catch (InterruptedException e) {
            e.printStackTrace(); // 處理異常
        }

        // Error handling verify markers are visible before proceeding
        boolean astronautMarkersOk = waitForMarkersDetection(2000, 200, "astronaut");

        if (astronautMarkersOk) {
            Log.i(TAG, "Astronaut markers confirmed - proceeding with target detection");
        } else {
            Log.w(TAG, "Astronaut markers not detected - proceeding anyway");
        }

        // ========================================================================
        // TARGET ITEM RECOGNITION
        // ========================================================================

        // Get target item image from astronaut
        Mat targetImage = api.getMatNavCam();

        // Process target image to identify what the astronaut is holding
        String targetTreasureType = processTargetImage(targetImage, resizeSize);

        if (targetTreasureType != null && !targetTreasureType.equals("unknown")) {
            Log.i(TAG, "Target treasure identified: " + targetTreasureType);

            // Find which area contains this treasure
            int targetAreaId = findTreasureInArea(targetTreasureType, areaTreasure);

            if (targetAreaId > 0) {
                Log.i(TAG, "Target treasure '" + targetTreasureType + "' found in Area " + targetAreaId);

                // Notify recognition
                api.notifyRecognitionItem();

                // Move back to the target area
                Point targetAreaPoint = AREA_POINTS[targetAreaId - 1];
                Quaternion targetAreaQuaternion = AREA_QUATERNIONS[targetAreaId - 1];

                Log.i(TAG, "Moving back to Area " + targetAreaId + " to get the treasure");
                // 先移动到目标区域默认位置
                api.moveTo(targetAreaPoint, targetAreaQuaternion, false);

                // 然后尝试移动到理想的拍照位置
                if (moveToIdealPhotoPosition(targetAreaId)) {
                    Log.i(TAG, "Successfully moved to ideal photo position");
                } else {
                    Log.i(TAG, "Using default position for photo");
                }

                // Take a snapshot of the target item
                api.takeTargetItemSnapshot();

                Log.i(TAG, "Mission completed successfully!");
            } else {
                Log.w(TAG, "Target treasure '" + targetTreasureType + "' not found in any area, using inference");
                // 使用备用推断方法
                targetAreaId = inferTargetAreaFromLandmarks(targetImage, resizeSize, areaTreasure);
                Log.i(TAG, "推断目标区域: " + targetAreaId);

                // 继续后续流程...
                api.notifyRecognitionItem();

                // Move back to the target area
                if (targetAreaId > 0) {
                    Point targetAreaPoint = AREA_POINTS[targetAreaId - 1];
                    Quaternion targetAreaQuaternion = AREA_QUATERNIONS[targetAreaId - 1];

                    Log.i(TAG, "Moving back to inferred Area " + targetAreaId);
                    api.moveTo(targetAreaPoint, targetAreaQuaternion, false);
                }

                api.takeTargetItemSnapshot();
            }
        } else {
            Log.w(TAG, "Could not identify target treasure from astronaut, using inference");

            // 使用备用推断方法
            int inferredAreaId = inferTargetAreaFromLandmarks(targetImage, resizeSize, areaTreasure);
            Log.i(TAG, "推断目标区域: " + inferredAreaId);

            api.notifyRecognitionItem();

            if (inferredAreaId > 0) {
                // 移动到推断的区域
                Point inferredAreaPoint = AREA_POINTS[inferredAreaId - 1];
                Quaternion inferredAreaQuaternion = AREA_QUATERNIONS[inferredAreaId - 1];

                Log.i(TAG, "Moving back to inferred Area " + inferredAreaId);
                api.moveTo(inferredAreaPoint, inferredAreaQuaternion, false);
            }

            api.takeTargetItemSnapshot();
        }

        // Clean up target image
        targetImage.release();
    }

    private void processAreaWithDualTags(int areaId, Map<Integer, Set<String>> areaTreasure) {
        try {
            // 确保是区域2
            if (areaId != 2) {
                Log.w(TAG, "此方法只应在处理区域2时调用");
                return;
            }

            // 移动到区域2
            Point targetPoint = AREA_POINTS[areaId - 1];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaId - 1];

            Log.i(TAG, String.format("Moving to Area %d: Point(%.3f, %.3f, %.3f)",
                    areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            api.moveTo(new Point(10.938d, -9.5d, 4.945d), targetQuaternion, false);
            api.moveTo(new Point(10.925d, -8.875d, 4.94d), targetQuaternion, false);

            // 获取相机图像
            Mat image = api.getMatNavCam();

            // 保存原始图像用于调试
            api.saveMatImage(image, "area_2_dual_processing_raw.png");

            // 初始化ArUco检测
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();

            // 检测标记
            Aruco.detectMarkers(image, dictionary, corners, ids);

            if (corners.size() >= 2) {
                Log.i(TAG, "检测到" + corners.size() + "个标记，将分别处理为区域2和区域3");

                // 获取最接近中心的两个标记
                Object[] filtered = keepClosestMarker(corners, ids, image, 2);
                List<Mat> filteredCorners = (List<Mat>) filtered[0];
                Mat filteredIds = (Mat) filtered[1];

                if (filteredCorners.size() >= 2) {
                    // 配置图像处理参数
                    Size cropWarpSize = new Size(640, 480);
                    Size resizeSize = new Size(320, 320);

                    // 处理相机参数
                    double[][] intrinsics = api.getNavCamIntrinsics();
                    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                    Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);

                    cameraMatrix.put(0, 0, intrinsics[0]);
                    distCoeffs.put(0, 0, intrinsics[1]);

                    // 估计标记位姿
                    Mat rvecs = new Mat();
                    Mat tvecs = new Mat();
                    float markerLength = 0.05f;

                    Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs,
                            tvecs);

                    if (rvecs.rows() >= 2) {
                        // 处理区域2的标记（第一个最靠近中心的标记）
                        processAreaTag(image, cameraMatrix, distCoeffs, rvecs, tvecs, 0, cropWarpSize, resizeSize, 2,
                                areaTreasure);

                        // 处理区域3的标记（第二个最靠近中心的标记）
                        processAreaTag(image, cameraMatrix, distCoeffs, rvecs, tvecs, 1, cropWarpSize, resizeSize, 3,
                                areaTreasure);

                        Log.i(TAG, "区域2和区域3的标记已成功处理");
                    } else {
                        Log.w(TAG, "无法估计两个标记的位姿");
                        processNormalArea(2, areaTreasure); // 回退到正常处理
                    }

                    // 释放资源
                    // Release all resources individually
                    cameraMatrix.release();
                    distCoeffs.release();
                    rvecs.release();
                    tvecs.release();
                    filteredIds.release();
                } else {
                    Log.w(TAG, "筛选后没有足够的标记，回退到正常处理");
                    processNormalArea(2, areaTreasure); // 回退到正常处理
                }

                // 释放角点资源
                for (Mat corner : filteredCorners) {
                    if (corner != null && !corner.empty())
                        corner.release();
                }

            } else {
                Log.w(TAG, "未检测到足够的标记（至少需要2个），回退到正常处理");
                processNormalArea(2, areaTreasure); // 回退到正常处理
            }

            // 释放资源
            for (Mat corner : corners) {
                corner.release();
            }
            ids.release();
            image.release();

        } catch (Exception e) {
            Log.e(TAG, "处理双标记区域时出错: " + e.getMessage());
            processNormalArea(2, areaTreasure); // 回退到正常处理
        }
    }

    /**
     * Process target image to identify the treasure type the astronaut is holding
     * 
     * @param targetImage Image from astronaut
     * @param resizeSize  Processing size
     * @return Treasure type name or "unknown"
     */
    private String processTargetImage(Mat targetImage, Size resizeSize) {
        try {
            Log.i(TAG, "Processing target image from astronaut");

            // Save the target image for debugging
            api.saveMatImage(targetImage, "target_astronaut_raw.png");

            // Use the SAME processing pipeline as areas (ArUco detection + cropping +
            // enhancement)
            Size cropWarpSize = new Size(640, 480); // Same as area processing
            List<Mat> processedTargets = imageEnhanceAndCrop(targetImage, cropWarpSize, resizeSize, 0); // Use 0 for
                                                                                                        // target

            if (processedTargets != null && !processedTargets.isEmpty()) {
                Log.i(TAG, "Target image processing successful - markers detected and cropped: "
                        + processedTargets.size());

                // We expect only one marker but handle multiple just in case
                Map<String, Integer> landmark_items = new HashMap<>();
                Set<String> treasure_types = new HashSet<>();
                for (Mat processedTarget : processedTargets) {
                    Object[] detected_items = detectitemfromcvimg(
                            processedTarget,
                            0.3f, // Lower confidence for target detection
                            "target", // img_type for target
                            0.45f, // standard_nms_threshold
                            0.8f, // overlap_nms_threshold
                            320 // img_size
                    );

                    Map<String, Integer> singleLandmarks = (Map<String, Integer>) detected_items[0];
                    Set<String> singleTreasures = (Set<String>) detected_items[1];

                    for (Map.Entry<String, Integer> entry : singleLandmarks.entrySet()) {
                        landmark_items.put(entry.getKey(),
                                landmark_items.getOrDefault(entry.getKey(), 0) + entry.getValue());
                    }
                    treasure_types.addAll(singleTreasures);

                    processedTarget.release();
                }

                Log.i(TAG, "Target - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Target - Treasure types: " + treasure_types);

                if (!treasure_types.isEmpty()) {
                    String targetTreasure = treasure_types.iterator().next();
                    Log.i(TAG, "Target treasure detected: " + targetTreasure);
                    return targetTreasure;
                }

            } else {
                Log.w(TAG, "Target image processing failed - no markers detected or processing error");
            }

            Log.w(TAG, "No treasure detected in target image");
            return "unknown";

        } catch (Exception e) {
            Log.e(TAG, "Error processing target image: " + e.getMessage());
            return "unknown";
        }
    }

    /**
     * Basic enhancement for target image (simpler than area processing)
     */
    private Mat enhanceTargetImage(Mat image, Size resizeSize) {
        try {
            // 调整大小
            Mat resized = new Mat();
            Imgproc.resize(image, resized, resizeSize);

            // 使用直接加强 - 直方图均衡化
            Mat enhanced = new Mat();
            Imgproc.cvtColor(resized, enhanced, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(enhanced, enhanced);

            // 保存增强后的图像
            api.saveMatImage(enhanced, "target_astronaut_enhanced.png");

            // 释放资源
            resized.release();

            return enhanced;
        } catch (Exception e) {
            Log.e(TAG, "Error enhancing target image: " + e.getMessage());
            return null;
        }
    }

    /**
     * Find which area contains the specified treasure type
     * 
     * @param treasureType The treasure type to find
     * @param areaTreasure Map of area treasures
     * @return Area ID (1-4) or 0 if not found
     */
    private int findTreasureInArea(String treasureType, Map<Integer, Set<String>> areaTreasure) {
        for (int areaId = 1; areaId <= 4; areaId++) {
            Set<String> treasures = areaTreasure.get(areaId);
            if (treasures != null && treasures.contains(treasureType)) {
                return areaId;
            }
        }
        return 0; // Not found
    }

    /**
     * Method to detect items from CV image using YOLO - matches Python
     * testcallyololib.py functionality
     * 
     * @param image                  Input OpenCV Mat image
     * @param conf                   Confidence threshold (e.g., 0.3f)
     * @param imgtype                Image type: "lost" or "target"
     * @param standard_nms_threshold Standard NMS threshold (e.g., 0.45f)
     * @param overlap_nms_threshold  Overlap NMS threshold for intelligent NMS
     *                               (e.g., 0.6f)
     * @param img_size               Image size for processing (e.g., 320)
     * @return Object array: [landmark_quantities (Map<String, Integer>),
     *         treasure_types (Set<String>)]
     */
    private Object[] detectitemfromcvimg(Mat image, float conf, String imgtype,
            float standard_nms_threshold, float overlap_nms_threshold, int img_size) {
        YOLODetectionService yoloService = null;
        try {
            Log.i(TAG, String.format("Starting YOLO detection - type: %s, conf: %.2f", imgtype, conf));

            // Initialize YOLO detection service
            yoloService = new YOLODetectionService(this);

            // Enhance frame before running YOLO
            Mat frame = EnhanceUtils.enhance(image);

            // Call detection with all parameters (matches Python simple_detection_example)
            YOLODetectionService.EnhancedDetectionResult result = yoloService.DetectfromcvImage(
                    frame, imgtype, conf, standard_nms_threshold, overlap_nms_threshold);
            frame.release();

            // Get Python-like result with class names
            Map<String, Object> pythonResult = result.getPythonLikeResult();

            // Extract landmark quantities (Map<String, Integer>) - matches Python
            // detection['landmark_quantities']
            Map<String, Integer> landmarkQuantities = (Map<String, Integer>) pythonResult.get("landmark_quantities");
            if (landmarkQuantities == null) {
                landmarkQuantities = new HashMap<>();
            }

            // Extract treasure quantities and get the keys (types) - matches Python
            // detection['treasure_quantities'].keys()
            Map<String, Integer> treasureQuantities = (Map<String, Integer>) pythonResult.get("treasure_quantities");
            if (treasureQuantities == null) {
                treasureQuantities = new HashMap<>();
            }
            Set<String> treasureTypes = new HashSet<>(treasureQuantities.keySet());

            // Log results (matches Python print statements)
            Log.i(TAG, "Landmark quantities: " + landmarkQuantities);
            Log.i(TAG, "Treasure types: " + treasureTypes);

            // Return as array: [landmark_quantities, treasure_types]
            // This matches Python: report_landmark.append(detection['landmark_quantities'])
            // store_treasure.append(detection['treasure_quantities'].keys())
            return new Object[] { landmarkQuantities, treasureTypes };

        } catch (Exception e) {
            Log.e(TAG, "Error in detectitemfromcvimg: " + e.getMessage(), e);
            // Return empty results on error
            return new Object[] { new HashMap<String, Integer>(), new HashSet<String>() };
        } finally {
            // Clean up YOLO service
            if (yoloService != null) {
                yoloService.close();
            }
        }
    }

    /**
     * Helper method to get the first landmark item and its count (matches Python
     * usage pattern)
     * 
     * @param landmarkQuantities Map of landmark quantities
     * @return String array: [landmark_name, count_as_string] or null if empty
     */
    private String[] getFirstLandmarkItem(Map<String, Integer> landmarkQuantities) {
        if (landmarkQuantities != null && !landmarkQuantities.isEmpty()) {
            // Get first entry (matches Python landmark_items.keys()[0])
            Map.Entry<String, Integer> firstEntry = landmarkQuantities.entrySet().iterator().next();
            String landmarkName = firstEntry.getKey();
            Integer count = firstEntry.getValue();
            return new String[] { landmarkName, String.valueOf(count) };
        }
        return null;
    }

    /**
     * Enhanced image processing method that detects ArUco markers, crops region,
     * applies CLAHE enhancement, and binarizes the image
     * 
     * @param image        Input image from NavCam
     * @param cropWarpSize Size for the cropped/warped image (e.g., 640x480)
     * @param resizeSize   Size for the final processed image (e.g., 320x320)
     * @param areaId       Area identifier for filename generation
     * @return List of processed CLAHE + Otsu binarized images (one for each
     *         selected marker)
     */
    private List<Mat> imageEnhanceAndCrop(Mat image, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            // Save original test image with area ID
            String rawImageFilename = "area_" + areaId + "_raw.png";
            api.saveMatImage(image, rawImageFilename);
            Log.i(TAG, "Raw image saved as " + rawImageFilename);

            // Initialize ArUco detection
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();

            // Detect markers
            Aruco.detectMarkers(image, dictionary, corners, ids);

            if (corners.size() > 0) {
                Log.i(TAG, "Detected " + corners.size() + " markers.");

                // Keep up to two markers closest to image center
                Object[] filtered = keepClosestMarker(corners, ids, image, 2);
                List<Mat> filteredCorners = (List<Mat>) filtered[0];
                Mat filteredIds = (Mat) filtered[1];

                // Clean up original corners and ids (now safe since we cloned the data)
                for (Mat corner : corners) {
                    corner.release();
                }
                ids.release();

                Log.i(TAG, "Using closest markers. Remaining markers: " + filteredCorners.size());

                // Get camera parameters
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);

                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);
                distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

                // Estimate pose for selected markers
                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f; // 5cm markers

                Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

                // Prepare result list
                List<Mat> processedImages = new ArrayList<>();

                // Draw all markers once
                Mat imageWithFrame = image.clone();
                Aruco.drawDetectedMarkers(imageWithFrame, filteredCorners, filteredIds);

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    for (int i = 0; i < filteredCorners.size(); i++) {
                        Mat rvec = new Mat(3, 1, CvType.CV_64F);
                        Mat tvec = new Mat(3, 1, CvType.CV_64F);

                        rvecs.row(i).copyTo(rvec);
                        tvecs.row(i).copyTo(tvec);

                        // Draw axis for each marker
                        Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);

                        // Save marker with frame using area ID and index
                        String markerFilename = "area_" + areaId + "_marker_" + i + "_with_frame.png";
                        api.saveMatImage(imageWithFrame, markerFilename);

                        // Process crop region and store enhanced image
                        Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec,
                                cropWarpSize, resizeSize, areaId);
                        if (processedImage != null) {
                            processedImages.add(processedImage);
                        }

                        rvec.release();
                        tvec.release();
                    }

                    imageWithFrame.release();
                    cameraMatrix.release();
                    distCoeffs.release();
                    rvecs.release();
                    tvecs.release();

                    // Clean up filtered corners and ids
                    filteredIds.release();
                    for (Mat corner : filteredCorners) {
                        corner.release();
                    }

                    return processedImages;
                }

                // Clean up if pose estimation failed
                imageWithFrame.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
                filteredIds.release();
                for (Mat corner : filteredCorners) {
                    corner.release();
                }
            } else {
                Log.w(TAG, "No ArUco markers detected in image");
                // Clean up empty lists
                ids.release();
            }

            return new ArrayList<>(); // No markers detected

        } catch (Exception e) {
            Log.e(TAG, "Error in imageEnhanceAndCrop: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    /**
     * Helper method to process the crop region and apply CLAHE + binarization
     */
    private Mat processCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize,
            Size resizeSize, int areaId) {
        try {
            // Define crop area corners in 3D (manually adjusted)
            org.opencv.core.Point3[] cropCorners3D = {
                    new org.opencv.core.Point3(-0.0265, 0.0420, 0), // Top-left
                    new org.opencv.core.Point3(-0.2385, 0.0420, 0), // Top-right
                    new org.opencv.core.Point3(-0.2385, -0.1170, 0), // Bottom-right
                    new org.opencv.core.Point3(-0.0265, -0.1170, 0) // Bottom-left
            };

            MatOfPoint3f cropCornersMat = new MatOfPoint3f(cropCorners3D);
            MatOfPoint2f cropCorners2D = new MatOfPoint2f();

            // Convert distortion coefficients
            double[] distData = new double[5];
            distCoeffs.get(0, 0, distData);
            MatOfDouble distCoeffsDouble = new MatOfDouble();
            distCoeffsDouble.fromArray(distData);

            // Project crop corners to 2D
            Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
            org.opencv.core.Point[] cropPoints2D = cropCorners2D.toArray();

            if (cropPoints2D.length == 4) {
                // Create perspective transformation and get processed image with custom sizes
                Mat processedImage = cropEnhanceAndBinarize(image, cropPoints2D, cropWarpSize, resizeSize, areaId);

                // Clean up
                cropCornersMat.release();
                cropCorners2D.release();
                distCoeffsDouble.release();

                return processedImage;
            }

            // Clean up if crop failed
            cropCornersMat.release();
            cropCorners2D.release();
            distCoeffsDouble.release();

            return null;

        } catch (Exception e) {
            Log.e(TAG, "Error in processCropRegion: " + e.getMessage());
            return null;
        }
    }

    /**
     * Helper method to crop, enhance with CLAHE, and binarize the image
     * 
     * @param image        Input image
     * @param cropPoints2D 2D points for perspective transformation
     * @param cropWarpSize Size for the cropped/warped image (configurable)
     * @param resizeSize   Size for the final processed image (configurable)
     * @param areaId       Area identifier for filename generation
     */
    private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize,
            Size resizeSize, int areaId) {
        try {
            // STEP 1: 创建裁剪图像
            org.opencv.core.Point[] dstPointsCrop = {
                    new org.opencv.core.Point(0, 0),
                    new org.opencv.core.Point(cropWarpSize.width - 1, 0),
                    new org.opencv.core.Point(cropWarpSize.width - 1, cropWarpSize.height - 1),
                    new org.opencv.core.Point(0, cropWarpSize.height - 1)
            };

            MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
            MatOfPoint2f dstPointsMatCrop = new MatOfPoint2f(dstPointsCrop);
            Mat perspectiveMatrixCrop = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMatCrop);
            Mat croppedImage = new Mat();
            Imgproc.warpPerspective(image, croppedImage, perspectiveMatrixCrop, cropWarpSize);

            // 记录裁剪图像信息
            Core.MinMaxLocResult minMaxResultCrop = Core.minMaxLoc(croppedImage);
            Log.i(TAG, String.format("Cropped image %.0fx%.0f - Min: %.2f, Max: %.2f",
                    cropWarpSize.width, cropWarpSize.height, minMaxResultCrop.minVal, minMaxResultCrop.maxVal));

            // 保存裁剪图像
            String cropFilename = String.format("area_%d_cropped_region_%.0fx%.0f.png", areaId, cropWarpSize.width,
                    cropWarpSize.height);
            api.saveMatImage(croppedImage, cropFilename);

            // STEP 2: 调整大小
            Mat resizedImage = new Mat();
            Imgproc.resize(croppedImage, resizedImage, resizeSize);

            // 保存调整大小后的图像
            String resizeFilename = String.format("area_%d_yolo_original_%.0fx%.0f.png", areaId, resizeSize.width,
                    resizeSize.height);
            api.saveMatImage(resizedImage, resizeFilename);

            // STEP 3: 直接应用Otsu二值化
            Mat binarizedOtsu = new Mat();
            double otsuThreshold = Imgproc.threshold(resizedImage, binarizedOtsu, 0, 255,
                    Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

            // 记录Otsu二值化图像信息
            Core.MinMaxLocResult binaryOtsuResult = Core.minMaxLoc(binarizedOtsu);
            Log.i(TAG, String.format("Binary Otsu (%.1f) - Min: %.2f, Max: %.2f",
                    otsuThreshold, binaryOtsuResult.minVal, binaryOtsuResult.maxVal));

            // 保存二值化图像
            String binaryFilename = String.format("area_%d_yolo_binary_otsu_%.0fx%.0f.png", areaId, resizeSize.width,
                    resizeSize.height);
            api.saveMatImage(binarizedOtsu, binaryFilename);

            // 清理中间图像
            srcPointsMat.release();
            dstPointsMatCrop.release();
            perspectiveMatrixCrop.release();
            croppedImage.release();
            resizedImage.release();

            return binarizedOtsu;

        } catch (Exception e) {
            Log.e(TAG, "Error in cropEnhanceAndBinarize: " + e.getMessage());
            return null;
        }
    }

    /**
     * 处理指定区域标记
     */
    private void processAreaTag(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvecs, Mat tvecs,
            int markerIndex, Size cropWarpSize, Size resizeSize, int areaId,
            Map<Integer, Set<String>> areaTreasure) {
        try {
            // 获取此标记的旋转和平移向量
            Mat rvec = new Mat(3, 1, CvType.CV_64F);
            Mat tvec = new Mat(3, 1, CvType.CV_64F);

            rvecs.row(markerIndex).copyTo(rvec);
            tvecs.row(markerIndex).copyTo(tvec);

            // 处理此标记
            Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec,
                    cropWarpSize, resizeSize, areaId);

            // 如果处理成功，进行物品检测
            if (processedImage != null) {
                Map<String, Integer> landmark_items = new HashMap<>();
                Set<String> treasure_types = new HashSet<>();

                // 检测物品
                Object[] detected_items = detectitemfromcvimg(
                        processedImage,
                        0.3f, // conf_threshold
                        "lost", // img_type
                        0.45f, // standard_nms_threshold
                        0.8f, // overlap_nms_threshold
                        320 // img_size
                );

                // 提取检测结果
                Map<String, Integer> singleLandmarks = (Map<String, Integer>) detected_items[0];
                Set<String> singleTreasures = (Set<String>) detected_items[1];

                // 保存区域结果
                areaLandmarks.put("area" + areaId, singleLandmarks);
                foundTreasures.addAll(singleTreasures);
                foundLandmarks.addAll(singleLandmarks.keySet());
                areaTreasure.get(areaId).addAll(singleTreasures);

                // 记录地标类型
                recordAreaLandmarkTypes(areaId, singleLandmarks);

                // 设置区域信息
                String[] firstLandmark = getFirstLandmarkItem(singleLandmarks);
                if (firstLandmark != null) {
                    String landmarkName = firstLandmark[0];
                    int landmarkCount = Integer.parseInt(firstLandmark[1]);

                    // 记录已报告的地标
                    reportedLandmarks.add(landmarkName);

                    // 设置区域信息
                    api.setAreaInfo(areaId, landmarkName, landmarkCount);
                    Log.i(TAG, String.format("区域 %d: %s x %d", areaId, landmarkName, landmarkCount));
                } else {
                    // 使用备用地标
                    String[] fallbackLandmark = getFallbackLandmark();
                    api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
                    Log.i(TAG, String.format("区域 %d (备用): %s x %d",
                            areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1])));
                }

                // 释放资源
                processedImage.release();
            } else {
                Log.w(TAG, "区域 " + areaId + " 处理失败");

                // 使用备用地标
                String[] fallbackLandmark = getFallbackLandmark();
                api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
            }

            // 释放资源
            rvec.release();
            tvec.release();

        } catch (Exception e) {
            Log.e(TAG, "处理区域 " + areaId + " 标记时出错: " + e.getMessage());

            // 出错时使用备用地标
            String[] fallbackLandmark = getFallbackLandmark();
            api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
        }
    }

    /**
     * 正常处理单个区域
     */
    private void processNormalArea(int areaId, Map<Integer, Set<String>> areaTreasure) {
        try {
            Log.i(TAG, "执行区域 " + areaId + " 的正常处理流程");

            // 移动到区域位置
            Point targetPoint = AREA_POINTS[areaId - 1];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaId - 1];

            Log.i(TAG, String.format("移动到区域 %d: 坐标(%.3f, %.3f, %.3f)",
                    areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            api.moveTo(targetPoint, targetQuaternion, false);

            // 稳定等待
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // 获取图像
            Mat image = api.getMatNavCam();

            // 处理参数
            Size cropWarpSize = new Size(640, 480);
            Size resizeSize = new Size(320, 320);

            // 处理图像
            List<Mat> processedImages = imageEnhanceAndCrop(image, cropWarpSize, resizeSize, areaId);

            // 初始化结果容器
            Map<String, Integer> landmark_items = new HashMap<>();
            Set<String> treasure_types = new HashSet<>();

            if (processedImages != null && !processedImages.isEmpty()) {
                Log.i(TAG, "区域 " + areaId + ": 图像处理成功，处理 " + processedImages.size() + " 个标记");

                // 处理每个图像
                for (Mat processedImage : processedImages) {
                    Object[] detected_items = detectitemfromcvimg(
                            processedImage,
                            0.3f, // conf_threshold
                            "lost", // img_type
                            0.45f, // standard_nms_threshold
                            0.8f, // overlap_nms_threshold
                            320 // img_size
                    );

                    // 合并结果
                    Map<String, Integer> singleLandmarks = (Map<String, Integer>) detected_items[0];
                    Set<String> singleTreasures = (Set<String>) detected_items[1];

                    for (Map.Entry<String, Integer> entry : singleLandmarks.entrySet()) {
                        landmark_items.put(entry.getKey(),
                                landmark_items.getOrDefault(entry.getKey(), 0) + entry.getValue());
                    }
                    treasure_types.addAll(singleTreasures);

                    // 释放资源
                    processedImage.release();
                }

                // 保存检测结果
                areaLandmarks.put("area" + areaId, landmark_items);
                foundTreasures.addAll(treasure_types);
                foundLandmarks.addAll(landmark_items.keySet());
                areaTreasure.get(areaId).addAll(treasure_types);

                // 记录地标
                recordAreaLandmarkTypes(areaId, landmark_items);
            }

            // 清理原始图像
            image.release();

            // 设置区域信息
            String[] firstLandmark = getFirstLandmarkItem(landmark_items);
            if (firstLandmark != null) {
                String landmarkName = firstLandmark[0];
                int landmarkCount = Integer.parseInt(firstLandmark[1]);

                // 记录已报告地标
                reportedLandmarks.add(landmarkName);

                // 设置区域信息
                api.setAreaInfo(areaId, landmarkName, landmarkCount);
                Log.i(TAG, String.format("区域 %d: %s x %d", areaId, landmarkName, landmarkCount));
            } else {
                // 使用备用地标
                String[] fallbackLandmark = getFallbackLandmark();
                api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
                Log.i(TAG, String.format("区域 %d (备用): %s x %d",
                        areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1])));
            }

        } catch (Exception e) {
            Log.e(TAG, "处理区域 " + areaId + " 时出错: " + e.getMessage());

            // 错误时使用备用地标
            String[] fallbackLandmark = getFallbackLandmark();
            api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
        }
    }

    // 使用系统时间戳代替随机数生成备用地标
    private String[] getFallbackLandmark() {
        // 创建未报告过的Landmark列表
        List<String> unreportedLandmarks = new ArrayList<>();
        for (String landmark : ALL_POSSIBLE_LANDMARKS) {
            if (!reportedLandmarks.contains(landmark)) {
                unreportedLandmarks.add(landmark);
            }
        }
        
        // 如果所有Landmark都已报告过，则从所有可能的Landmark中选择第一个
        if (unreportedLandmarks.isEmpty()) {
            unreportedLandmarks = new ArrayList<>();
            for (String landmark : ALL_POSSIBLE_LANDMARKS) {
                unreportedLandmarks.add(landmark);
            }
        }
        
        // 使用System.currentTimeMillis()代替Random
        int randomIndex = (int)(System.currentTimeMillis() % unreportedLandmarks.size());
        String selectedLandmark = unreportedLandmarks.get(randomIndex);
        
        // 记录已报告的Landmark
        reportedLandmarks.add(selectedLandmark);
        
        Log.i(TAG, "选择了备用Landmark: " + selectedLandmark);
        return new String[] {selectedLandmark, "1"};
    }

    // 选择包含未发现宝藏的随机区域
    private int getRandomAreaWithTreasure(Map<Integer, Set<String>> areaTreasure) {
        List<Integer> areasWithTreasure = new ArrayList<>();

        for (int areaId = 1; areaId <= 4; areaId++) {
            if (areaTreasure.get(areaId) != null && !areaTreasure.get(areaId).isEmpty()) {
                areasWithTreasure.add(areaId);
            }
        }

        Random rand = new Random();

        if (areasWithTreasure.isEmpty()) {
            // Choose a random area between 1 and 4
            return rand.nextInt(4) + 1;
        } else {
            // Choose randomly from the areas that still have treasure
            int randomIndex = rand.nextInt(areasWithTreasure.size());
            return areasWithTreasure.get(randomIndex);
        }
    }

    /**
     * 优化版本的图像增强工具，不依赖OpenCV的Photo和Imgcodecs
     */
    public static final class EnhanceUtils {

        /** 传回处理后的Mat，方便串接在实时影像流程中 */
        public static Mat enhance(Mat src) {
            // 1. 使用高斯模糊替代降噪
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(src, blurred, new Size(5, 5), 0);
            
            // 2. 使用直方图均衡化增强对比度
            Mat enhanced = new Mat();
            if (src.channels() == 1) {
                // 灰度图像直接均衡化
                Imgproc.equalizeHist(blurred, enhanced);
            } else {
                // 彩色图像转为HSV空间后均衡化亮度通道
                Mat hsv = new Mat();
                Imgproc.cvtColor(blurred, hsv, Imgproc.COLOR_BGR2HSV);
                List<Mat> channels = new ArrayList<>();
                Core.split(hsv, channels);
                
                // 对亮度通道(V)进行均衡化
                Imgproc.equalizeHist(channels.get(2), channels.get(2));
                
                // 重新合并通道
                Core.merge(channels, hsv);
                Imgproc.cvtColor(hsv, enhanced, Imgproc.COLOR_HSV2BGR);
                
                // 释放资源
                hsv.release();
                for (Mat channel : channels) {
                    channel.release();
                }
            }
            
            // 3. 锐化处理
            Mat sharpened = new Mat();
            Mat kernel = new Mat(3, 3, CvType.CV_32F);
            // 拉普拉斯锐化核
            float[] kernelData = {
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            };
            kernel.put(0, 0, kernelData);
            Imgproc.filter2D(enhanced, sharpened, -1, kernel);
            
            // 释放资源
            blurred.release();
            enhanced.release();
            kernel.release();
            
            return sharpened;
        }
        
        private EnhanceUtils() {} // 私有构造函数
    }

    /**
     * 记录该区域的地标类型
     */
    private void recordAreaLandmarkTypes(int areaId, Map<String, Integer> landmark_items) {
        if (landmark_items != null && !landmark_items.isEmpty()) {
            Set<String> landmarkTypes = landmark_items.keySet();
            if (areaLandmarkTypes.get(areaId) != null) {
                areaLandmarkTypes.get(areaId).addAll(landmarkTypes);
                Log.i(TAG, "区域 " + areaId + " 的 Landmark 类型: " + landmarkTypes);
            }
        }
    }

    /**
     * 在指定时间内等待检测到标记
     * @param totalWaitTime 总等待时间(毫秒)
     * @param checkInterval 检查间隔(毫秒)
     * @param purpose 等待目的(日志用)
     * @return 是否在超时前检测到标记
     */
    private boolean waitForMarkersDetection(int totalWaitTime, int checkInterval, String purpose) {
        try {
            int elapsedTime = 0;
            while (elapsedTime < totalWaitTime) {
                // 获取图像
                Mat image = api.getMatNavCam();
                if (image == null) {
                    Thread.sleep(checkInterval);
                    elapsedTime += checkInterval;
                    continue;
                }
                
                // 检测ArUco标记
                Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                List<Mat> corners = new ArrayList<>();
                Mat ids = new Mat();
                
                Aruco.detectMarkers(image, dictionary, corners, ids);
                
                // 清理资源
                image.release();
                ids.release();
                
                // 检查是否检测到标记
                if (!corners.isEmpty()) {
                    Log.i(TAG, purpose + " 标记检测: 发现 " + corners.size() + " 个标记");
                    // 释放corners
                    for (Mat corner : corners) {
                        if (corner != null && !corner.empty()) {
                            corner.release();
                        }
                    }
                    return true;
                }
                
                // 释放corners
                for (Mat corner : corners) {
                    if (corner != null && !corner.empty()) {
                        corner.release();
                    }
                }
                
                Log.i(TAG, purpose + " 标记检测: 尚未发现标记, 已等待 " + elapsedTime + "ms");
                Thread.sleep(checkInterval);
                elapsedTime += checkInterval;
            }
            
            Log.w(TAG, purpose + " 标记检测: 超时 (" + totalWaitTime + "ms), 未检测到标记");
            return false;
        } catch (Exception e) {
            Log.e(TAG, purpose + " 标记检测出错: " + e.getMessage());
            return false;
        }
    }

    /**
     * 保留最接近图像中心的指定数量的标记
     * @param corners 所有检测到的标记角点
     * @param ids 所有检测到的标记ID
     * @param image 原始图像
     * @param keepCount 要保留的标记数量
     * @return 对象数组：[filteredCorners, filteredIds]
     */
    private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image, int keepCount) {
        try {
            if (corners.isEmpty() || ids.empty()) {
                return new Object[] { new ArrayList<Mat>(), new Mat() };
            }
            
            // 获取图像中心
            double centerX = image.cols() / 2.0;
            double centerY = image.rows() / 2.0;
            
            // 计算每个标记中心到图像中心的距离
            List<double[]> markerDistances = new ArrayList<>();
            
            for (int i = 0; i < corners.size(); i++) {
                Mat corner = corners.get(i);
                float[] cornerData = new float[8];
                corner.get(0, 0, cornerData);
                
                // 计算标记中心
                double markerCenterX = 0;
                double markerCenterY = 0;
                for (int j = 0; j < 4; j++) {
                    markerCenterX += cornerData[j*2];
                    markerCenterY += cornerData[j*2+1];
                }
                markerCenterX /= 4.0;
                markerCenterY /= 4.0;
                
                // 计算到中心的距离
                double distance = Math.sqrt(
                    Math.pow(markerCenterX - centerX, 2) + 
                    Math.pow(markerCenterY - centerY, 2)
                );
                
                markerDistances.add(new double[] { i, distance });
            }
            
            // 按距离排序(不使用lambda表达式，避免兼容性问题)
            java.util.Collections.sort(markerDistances, new java.util.Comparator<double[]>() {
                @Override
                public int compare(double[] a, double[] b) {
                    return Double.compare(a[1], b[1]);
                }
            });
            
            // 保留指定数量的最近标记
            List<Mat> filteredCorners = new ArrayList<>();
            Mat filteredIds = new Mat(Math.min(keepCount, corners.size()), 1, CvType.CV_32SC1);
            
            for (int i = 0; i < Math.min(keepCount, markerDistances.size()); i++) {
                int idx = (int) markerDistances.get(i)[0];
                filteredCorners.add(corners.get(idx).clone());
                
                int[] idValue = new int[1];
                ids.get(idx, 0, idValue);
                filteredIds.put(i, 0, idValue);
                
                Log.i(TAG, String.format("保留标记 #%d (ID: %d), 距离中心 %.1f 像素", 
                    i, idValue[0], markerDistances.get(i)[1]));
            }
            
            return new Object[] { filteredCorners, filteredIds };
            
        } catch (Exception e) {
            Log.e(TAG, "筛选最近标记时出错: " + e.getMessage());
            return new Object[] { new ArrayList<Mat>(), new Mat() };
        }
    }
    
    /**
     * 移动到指定区域的最佳拍照位置，确保满足距离和角度要求
     * @param areaId 目标区域ID (1-4)
     * @return 是否成功移动到理想位置
     */
    private boolean moveToIdealPhotoPosition(int areaId) {
        try {
            // 初始化尝试计数
            int attempts = 0;
            int maxAttempts = 3;
            boolean conditionsMet = false;
            
            // 保存最佳尝试结果
            boolean anySuccess = false;
            double bestAngleDiff = Double.MAX_VALUE;
            double bestDistance = Double.MAX_VALUE;
            Point bestPosition = null;
            Quaternion bestOrientation = null;
            
            while (!conditionsMet && attempts < maxAttempts) {
                attempts++;
                Log.i(TAG, "开始第 " + attempts + " 次尝试移动到理想拍照位置");
                
                // 拍摄一张图像分析当前场景
                Mat image = api.getMatNavCam();
                if (image == null) {
                    Log.e(TAG, "无法获取相机图像");
                    continue;
                }
                
                // 检测ArUco标记
                Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                List<Mat> corners = new ArrayList<>();
                Mat ids = new Mat();
                
                Aruco.detectMarkers(image, dictionary, corners, ids);
                
                if (corners.isEmpty()) {
                    Log.w(TAG, "尝试 " + attempts + ": 未检测到标记");
                    image.release();
                    ids.release();
                    continue;
                }
                
                // 获取最近的标记
                Object[] filtered = keepClosestMarker(corners, ids, image, 1);
                List<Mat> filteredCorners = (List<Mat>) filtered[0];
                Mat filteredIds = (Mat) filtered[1];

                // Get camera parameters
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);

                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);

                // Estimate pose for selected markers
                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f; // 5cm markers

                Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

                if (rvecs.empty() || tvecs.empty() || rvecs.rows() == 0) {
                    Log.w(TAG, "尝试 " + attempts + ": 无法估计标记的位姿");
                    releaseAll(image, ids, cameraMatrix, distCoeffs, rvecs, tvecs, filteredIds);
                    for (Mat corner : filteredCorners) {
                        if (corner != null && !corner.empty()) corner.release();
                    }
                    continue;
                }
                
                // 获取旋转和平移向量
                double[] rvecData = new double[3];
                double[] tvecData = new double[3];
                rvecs.get(0, 0, rvecData);
                tvecs.get(0, 0, tvecData);
                
                // --- 计算理想位置 ---
                // 计算当前与标记的距离
                double currentDistance = Math.sqrt(
                    tvecData[0] * tvecData[0] +
                    tvecData[1] * tvecData[1] +
                    tvecData[2] * tvecData[2]);
                
                Log.i(TAG, String.format("尝试 %d: 当前与标记距离 %.2f 米", attempts, currentDistance));
                
                // 将旋转向量转换为旋转矩阵
                Mat rotMatrix = new Mat();
                Calib3d.Rodrigues(new MatOfDouble(rvecData), rotMatrix);
                
                // 计算标记法线（Z轴方向）
                double[] zAxis = new double[3];
                double[] rotData = new double[9];
                rotMatrix.get(0, 0, rotData);
                
                zAxis[0] = rotData[2];  // 第一列第三个元素
                zAxis[1] = rotData[5];  // 第二列第三个元素
                zAxis[2] = rotData[8];  // 第三列第三个元素
                
                // 归一化
                double zNorm = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
                zAxis[0] /= zNorm;
                zAxis[1] /= zNorm;
                zAxis[2] /= zNorm;
                
                // 计算理想的拍照距离（约0.6米）
                float targetDistance = 0.6f;
                
                // 获取当前机器人位置
                Point currentPos = api.getRobotKinematics().getPosition();
                
                // 计算理想位置（当前位置 + 调整向量）
                Point idealPos = new Point(
                    currentPos.getX() + (tvecData[0] / currentDistance - zAxis[0]) * targetDistance,
                    currentPos.getY() + (tvecData[1] / currentDistance - zAxis[1]) * targetDistance,
                    currentPos.getZ() + (tvecData[2] / currentDistance - zAxis[2]) * targetDistance);
                
                // 计算面向标记的四元数
                double[] forwardVector = {-zAxis[0], -zAxis[1], -zAxis[2]};
                
                // 上向量（默认用z轴）
                double[] upVector = {0, 0, 1};
                
                // 计算右向量（叉积）
                double[] rightVector = {
                    forwardVector[1] * upVector[2] - forwardVector[2] * upVector[1],
                    forwardVector[2] * upVector[0] - forwardVector[0] * upVector[2],
                    forwardVector[0] * upVector[1] - forwardVector[1] * upVector[0]
                };
                
                // 归一化右向量
                double rightLen = Math.sqrt(rightVector[0]*rightVector[0] + rightVector[1]*rightVector[1] + rightVector[2]*rightVector[2]);
                if (rightLen < 1e-6) {
                    // 如果前向向量与上向量平行，使用另一个上向量
                    upVector[0] = 0; upVector[1] = 1; upVector[2] = 0;
                    
                    rightVector[0] = forwardVector[1] * upVector[2] - forwardVector[2] * upVector[1];
                    rightVector[1] = forwardVector[2] * upVector[0] - forwardVector[0] * upVector[2];
                    rightVector[2] = forwardVector[0] * upVector[1] - forwardVector[1] * upVector[0];
                    
                    rightLen = Math.sqrt(rightVector[0]*rightVector[0] + 
                                     rightVector[1]*rightVector[1] + 
                                     rightVector[2]*rightVector[2]);
                }
                
                rightVector[0] /= rightLen;
                rightVector[1] /= rightLen;
                rightVector[2] /= rightLen;
                
                // 重新计算上向量（叉积）以确保正交
                double[] realUp = {
                    rightVector[1] * forwardVector[2] - rightVector[2] * forwardVector[1],
                    rightVector[2] * forwardVector[0] - rightVector[0] * forwardVector[2],
                    rightVector[0] * forwardVector[1] - rightVector[1] * forwardVector[0]
                };
                
                // 从旋转矩阵构建四元数
                double trace = rightVector[0] + realUp[1] + forwardVector[2];
                
                float qw, qx, qy, qz;
                
                if (trace > 0) {
                    float s = 0.5f / (float)Math.sqrt(trace + 1.0);
                    qw = 0.25f / s;
                    qx = (float)((realUp[2] - forwardVector[1]) * s);
                    qy = (float)((forwardVector[0] - rightVector[2]) * s);
                    qz = (float)((rightVector[1] - realUp[0]) * s);
                } else if (rightVector[0] > realUp[1] && rightVector[0] > forwardVector[2]) {
                    float s = 2.0f * (float)Math.sqrt(1.0 + rightVector[0] - realUp[1] - forwardVector[2]);
                    qw = (float)((realUp[2] - forwardVector[1]) / s);
                    qx = 0.25f * s;
                    qy = (float)((realUp[0] + rightVector[1]) / s);
                    qz = (float)((forwardVector[0] + rightVector[2]) / s);
                } else if (realUp[1] > forwardVector[2]) {
                    float s = 2.0f * (float)Math.sqrt(1.0 + realUp[1] - rightVector[0] - forwardVector[2]);
                    qw = (float)((forwardVector[0] - rightVector[2]) / s);
                    qx = (float)((realUp[0] + rightVector[1]) / s);
                    qy = 0.25f * s;
                    qz = (float)((forwardVector[1] + realUp[2]) / s);
                } else {
                    float s = 2.0f * (float)Math.sqrt(1.0 + forwardVector[2] - rightVector[0] - realUp[1]);
                    qw = (float)((rightVector[1] - realUp[0]) / s);
                    qx = (float)((forwardVector[0] + rightVector[2]) / s);
                    qy = (float)((forwardVector[1] + realUp[2]) / s);
                    qz = 0.25f * s;
                }
                
                Quaternion idealQuat = new Quaternion(qx, qy, qz, qw);
                
                Log.i(TAG, String.format("尝试 %d: 理想拍照位置 (%.3f, %.3f, %.3f)",
                    attempts, idealPos.getX(), idealPos.getY(), idealPos.getZ()));
                
                // 执行移动
                Result moveResult = api.moveTo(idealPos, idealQuat, false);
                
                if (!moveResult.hasSucceeded()) {
                    Log.w(TAG, "尝试 " + attempts + ": 移动失败");
                    releaseAll(image, ids, cameraMatrix, distCoeffs, rvecs, tvecs, filteredIds, rotMatrix);
                    for (Mat corner : filteredCorners) {
                        if (corner != null && !corner.empty()) corner.release();
                    }
                    continue;
                }
                
                // --- 验证移动后的位置是否满足条件 ---
                
                // 等待机器人稳定
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                
                // 获取新图像检查位置和朝向
                Mat verifyImage = api.getMatNavCam();
                List<Mat> verifyCorners = new ArrayList<>();
                Mat verifyIds = new Mat();
                
                Aruco.detectMarkers(verifyImage, dictionary, verifyCorners, verifyIds);
                
                if (verifyCorners.isEmpty()) {
                    Log.w(TAG, "尝试 " + attempts + ": 验证时未检测到标记");
                    anySuccess = true;  // 至少移动成功了，即使无法验证
                    bestPosition = idealPos;
                    bestOrientation = idealQuat;
                    
                    releaseAll(verifyImage, verifyIds, image, ids, cameraMatrix, distCoeffs, 
                            rvecs, tvecs, filteredIds, rotMatrix);
                    for (Mat corner : filteredCorners) {
                        if (corner != null && !corner.empty()) corner.release();
                    }
                    continue;
                }
                
                // 估计移动后的标记位姿
                Mat verifyRvecs = new Mat();
                Mat verifyTvecs = new Mat();
                
                Aruco.estimatePoseSingleMarkers(verifyCorners, markerLength,
                        cameraMatrix, distCoeffs, verifyRvecs, verifyTvecs);
                
                if (!verifyRvecs.empty() && !verifyTvecs.empty() && verifyRvecs.rows() > 0) {
                    // 检查距离
                    double[] vtvecData = new double[3];
                    double[] vrvecData = new double[3];
                    verifyTvecs.get(0, 0, vtvecData);
                    verifyRvecs.get(0, 0, vrvecData);
                    
                    double finalDistance = Math.sqrt(
                        vtvecData[0] * vtvecData[0] +
                        vtvecData[1] * vtvecData[1] +
                        vtvecData[2] * vtvecData[2]);
                    
                    // 检查角度 - 计算旋转向量的模长，转换为角度
                    double angleDiff = Math.sqrt(
                        vrvecData[0] * vrvecData[0] +
                        vrvecData[1] * vrvecData[1] +
                        vrvecData[2] * vrvecData[2]) * 180.0 / Math.PI;
                    
                    Log.i(TAG, String.format("尝试 %d 验证结果: 距离=%.2f米, 角度=%.2f度", 
                            attempts, finalDistance, angleDiff));
                    
                    // 检查条件是否满足
                    boolean distanceOk = finalDistance <= 0.9;
                    boolean angleOk = angleDiff <= 30.0;
                    
                    // 更新最佳尝试
                    if ((distanceOk && angleOk) || 
                        (finalDistance < bestDistance && angleDiff < bestAngleDiff)) {
                        bestDistance = finalDistance;
                        bestAngleDiff = angleDiff;
                        bestPosition = idealPos;
                        bestOrientation = idealQuat;
                        anySuccess = true;
                    }
                    
                    // 如果所有条件都满足，跳出循环
                    if (distanceOk && angleOk) {
                        Log.i(TAG, String.format("✓ 尝试 %d: 条件满足! 距离=%.2f米(≤0.9米), 角度=%.2f度(≤30度)", 
                                attempts, finalDistance, angleDiff));
                        conditionsMet = true;
                    } else {
                        Log.w(TAG, String.format("✗ 尝试 %d: 条件不满足。距离=%.2f米(需≤0.9米):%s, 角度=%.2f度(需≤30度):%s", 
                                attempts, finalDistance, distanceOk ? "满足" : "不满足", 
                                angleDiff, angleOk ? "满足" : "不满足"));
                    }
                    
                    verifyRvecs.release();
                    verifyTvecs.release();
                }
                
                // 清理验证资源
                for (Mat corner : verifyCorners) {
                    if (corner != null && !corner.empty()) corner.release();
                }
                verifyIds.release();
                verifyImage.release();
                
                // 清理主循环资源
                releaseAll(image, ids, cameraMatrix, distCoeffs, rvecs, tvecs, filteredIds, rotMatrix);
                for (Mat corner : filteredCorners) {
                    if (corner != null && !corner.empty()) corner.release();
                }
            }
            
            // 如果没有满足条件但至少有一次成功
            if (!conditionsMet && anySuccess && bestPosition != null) {
                Log.w(TAG, String.format("未能完全满足条件，使用最佳尝试：距离=%.2f米，角度=%.2f度", 
                        bestDistance, bestAngleDiff));
                
                // 如果最后一次尝试不是最佳的，移动到最佳位置
                Point currentPos = api.getRobotKinematics().getPosition();
                Quaternion currentQuat = api.getRobotKinematics().getOrientation();
                
                if (!currentPos.equals(bestPosition) || !currentQuat.equals(bestOrientation)) {
                    api.moveTo(bestPosition, bestOrientation, false);
                }
                
                // 拍照
                Mat finalImage = api.getMatNavCam();
                api.saveMatImage(finalImage, "area_" + areaId + "_treasure_photo.png");
                finalImage.release();
                
                return true;
            }
            
            return conditionsMet;
            
        } catch (Exception e) {
            Log.e(TAG, "移动到理想拍照位置时发生错误: " + e.getMessage(), e);
            return false;
        }
    }

    /**
     * 释放多个Mat资源
     */
    private void releaseAll(Mat... mats) {
        for (Mat m : mats) {
            if (m != null && !m.empty()) {
                m.release();
            }
        }
    }

    /**
     * 当无法识别目标宝物时，根据宇航员图像中的Landmark和之前的区域数据推测可能的宝物区域
     */
    private int inferTargetAreaFromLandmarks(Mat targetImage, Size resizeSize, Map<Integer, Set<String>> areaTreasure) {
        try {
            // 检测宇航员图像中的Landmark
            Mat enhancedImage = enhanceTargetImage(targetImage, resizeSize);
            if (enhancedImage == null) return getRandomAreaWithTreasure(areaTreasure);
            
            Object[] detected_items = detectitemfromcvimg(
                enhancedImage,
                0.3f,      // 降低阈值以增加检测几率
                "target",  // 目标图像类型
                0.45f,     // NMS阈值
                0.8f,      // 重叠NMS阈值
                320        // 图像尺寸
            );
            
            if (enhancedImage != null && !enhancedImage.empty()) {
                enhancedImage.release();
            }
            
            // 提取检测到的Landmark
            Map<String, Integer> detected_landmarks = (Map<String, Integer>) detected_items[0];
            
            if (detected_landmarks == null || detected_landmarks.isEmpty()) {
                Log.w(TAG, "在目标图像中未检测到任何Landmark，随机选择区域");
                return getRandomAreaWithTreasure(areaTreasure);
            }
            
            Log.i(TAG, "在目标图像中检测到的Landmark: " + detected_landmarks.keySet());
            
            // 分析每个区域与检测到的Landmark的匹配程度
            Map<Integer, Integer> areaMatchCount = new HashMap<>();
            for (int areaId = 1; areaId <= 4; areaId++) {
                Set<String> areaLandmarks = areaLandmarkTypes.get(areaId);
                Set<String> treasures = areaTreasure.get(areaId);
                
                if (areaLandmarks == null) continue;
                
                int matchCount = 0;
                for (String landmark : detected_landmarks.keySet()) {
                    if (areaLandmarks.contains(landmark)) {
                        matchCount++;
                    }
                }
                
                // 只考虑有宝藏的区域
                if (treasures != null && !treasures.isEmpty()) {
                    areaMatchCount.put(areaId, matchCount);
                }
            }
            
            Log.i(TAG, "各区域与目标图像Landmark的匹配程度: " + areaMatchCount);
            
            // 找出匹配程度最高的区域
            int bestAreaId = 0;
            int highestMatch = -1;
            
            for (Map.Entry<Integer, Integer> entry : areaMatchCount.entrySet()) {
                if (entry.getValue() > highestMatch) {
                    highestMatch = entry.getValue();
                    bestAreaId = entry.getKey();
                }
            }
            
            if (bestAreaId > 0) {
                Log.i(TAG, "根据Landmark分析，最可能的目标区域是: " + bestAreaId + " (匹配度: " + highestMatch + ")");
                return bestAreaId;
            } else {
                return getRandomAreaWithTreasure(areaTreasure);
            }
            
        } catch (Exception e) {
            Log.e(TAG, "推断目标区域时出错: " + e.getMessage());
            return getRandomAreaWithTreasure(areaTreasure);
        }
    }
}