package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.util.Log;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    private final String TAG = this.getClass().getSimpleName();
    
    // Instance variables to store detection results across areas
    private Set<String> foundTreasures = new HashSet<>();
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<>();
    // Reusable YOLO detection service
    private YOLODetectionService yoloService;

    @Override
    protected void runPlan1(){
        // Log the start of the mission.
        Log.i(TAG, "Start mission");
        
        // The mission starts.
        api.startMission();

        // Initialize YOLO service once
        yoloService = new YOLODetectionService(this);

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image = api.getMatNavCam();

        // ========================================================================
        // CONFIGURABLE IMAGE PROCESSING PARAMETERS - EDIT HERE
        // ========================================================================
        
        Size cropWarpSize = new Size(640, 480);   // Size for cropped/warped image
        Size resizeSize = new Size(320, 320);     // Size for final processing
        
        // Send the image into image enhancement and cropping with custom sizes
        Mat claHeBinImage = imageEnhanceAndCrop(image, cropWarpSize, resizeSize);
        
        // Initialize detection results
        Map<String, Integer> landmark_items = new HashMap<>();
        Set<String> treasure_types = new HashSet<>();
        
        if (claHeBinImage != null) {
            Log.i(TAG, "Image enhancement and cropping successful");
            
            // Detect items using YOLO - matches Python testcallyololib.py functionality
            Object[] detected_items = detectitemfromcvimg(
                claHeBinImage, 
                0.3f,      // conf_threshold (same as Python)
                "lost",    // img_type ("lost" or "target") 
                0.45f,     // standard_nms_threshold (same as Python)
                0.8f,      // overlap_nms_threshold (same as Python)
                320        // img_size (same as Python)
            );
            
            // Extract results (matches Python report_landmark and store_treasure)
            landmark_items = (Map<String, Integer>) detected_items[0];
            treasure_types = (Set<String>) detected_items[1];
            
            Log.i(TAG, "Report landmark quantities: " + landmark_items);
            Log.i(TAG, "Store treasure types: " + treasure_types);
            
            // Store results for later use
            areaLandmarks.put("area1", landmark_items);
            foundTreasures.addAll(treasure_types);
            
            // Clean up the processed image when done
            claHeBinImage.release();
        } else {
            Log.w(TAG, "Image enhancement failed - no markers detected or processing error");
        }

        // Clean up original image
        image.release();

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // Use the detected landmark items (matches Python currentlandmark_items = landmark_items.keys()[0])
        String[] firstLandmark = getFirstLandmarkItem(landmark_items);
        if (firstLandmark != null) {
            String currentlandmark_items = firstLandmark[0];
            int landmarkCount = Integer.parseInt(firstLandmark[1]);
            
            // When you recognize landmark items, let's set the type and number.
            api.setAreaInfo(1, currentlandmark_items, landmarkCount);
            Log.i(TAG, String.format("Area 1: %s x %d", currentlandmark_items, landmarkCount));
        } else {
            Log.w(TAG, "No landmark items detected for area 1");
            // Set default if no detection
            api.setAreaInfo(1, "unknown", 0);
        }

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

        // Close YOLO service at mission end
        if (yoloService != null) {
            yoloService.close();
        }
    }

    @Override
    protected void runPlan2(){
       // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }


    /**
     * Method to detect items from CV image using YOLO - matches Python testcallyololib.py functionality
     * @param image Input OpenCV Mat image
     * @param conf Confidence threshold (e.g., 0.3f)
     * @param imgtype Image type: "lost" or "target"
     * @param standard_nms_threshold Standard NMS threshold (e.g., 0.45f)
     * @param overlap_nms_threshold Overlap NMS threshold for intelligent NMS (e.g., 0.8f)
     * @param img_size Image size for processing (e.g., 320)
     * @return Object array: [landmark_quantities (Map<String, Integer>), treasure_types (Set<String>)]
     */
    private Object[] detectitemfromcvimg(Mat image, float conf, String imgtype,
                                       float standard_nms_threshold, float overlap_nms_threshold, int img_size) {
        try {
            Log.i(TAG, String.format("Starting YOLO detection - type: %s, conf: %.2f", imgtype, conf));

            // Reuse initialized YOLO service
            if (yoloService == null) {
                Log.w(TAG, "YOLO service not initialized");
                yoloService = new YOLODetectionService(this);
            }
            
            // Call detection with all parameters (matches Python simple_detection_example)
            YOLODetectionService.EnhancedDetectionResult result = yoloService.DetectfromcvImage(
                image, imgtype, conf, standard_nms_threshold, overlap_nms_threshold
            );
            
            // Get Python-like result with class names
            Map<String, Object> pythonResult = result.getPythonLikeResult();
            
            // Extract landmark quantities (Map<String, Integer>) - matches Python detection['landmark_quantities']
            Map<String, Integer> landmarkQuantities = (Map<String, Integer>) pythonResult.get("landmark_quantities");
            if (landmarkQuantities == null) {
                landmarkQuantities = new HashMap<>();
            }
            
            // Extract treasure quantities and get the keys (types) - matches Python detection['treasure_quantities'].keys()
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
            //                     store_treasure.append(detection['treasure_quantities'].keys())
            return new Object[]{landmarkQuantities, treasureTypes};
            
        } catch (Exception e) {
            Log.e(TAG, "Error in detectitemfromcvimg: " + e.getMessage(), e);
            // Return empty results on error
            return new Object[]{new HashMap<String, Integer>(), new HashSet<String>()};
        }
    }

    /**
     * Helper method to get the first landmark item and its count (matches Python usage pattern)
     * @param landmarkQuantities Map of landmark quantities
     * @return String array: [landmark_name, count_as_string] or null if empty
     */
    private String[] getFirstLandmarkItem(Map<String, Integer> landmarkQuantities) {
        if (landmarkQuantities != null && !landmarkQuantities.isEmpty()) {
            // Get first entry (matches Python landmark_items.keys()[0])
            Map.Entry<String, Integer> firstEntry = landmarkQuantities.entrySet().iterator().next();
            String landmarkName = firstEntry.getKey();
            Integer count = firstEntry.getValue();
            return new String[]{landmarkName, String.valueOf(count)};
        }
        return null;
    }

    /**
     * Enhanced image processing method that detects ArUco markers, crops region, 
     * applies CLAHE enhancement, and binarizes the image
     * @param image Input image from NavCam
     * @param cropWarpSize Size for the cropped/warped image (e.g., 640x480)
     * @param resizeSize Size for the final processed image (e.g., 320x320)
     * @return Processed CLAHE + Otsu binarized image, or null if no markers detected
     */
    private Mat imageEnhanceAndCrop(Mat image, Size cropWarpSize, Size resizeSize) {
        try {
            // Save original test image
            api.saveMatImage(image, "test.png");
            
            // Initialize ArUco detection
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            
            // Detect markers
            Aruco.detectMarkers(image, dictionary, corners, ids);
            
            if (corners.size() > 0) {
                Log.i(TAG, "Detected " + corners.size() + " markers.");
                
                // Get camera parameters
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
                
                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);
                distCoeffs.convertTo(distCoeffs, CvType.CV_64F);
                
                // Estimate pose for first marker
                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f; // 5cm markers
                
                Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
                
                // Process first marker only
                Mat imageWithFrame = image.clone();
                Aruco.drawDetectedMarkers(imageWithFrame, corners, ids);
                
                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = new Mat(3, 1, CvType.CV_64F);
                    Mat tvec = new Mat(3, 1, CvType.CV_64F);
                    
                    rvecs.row(0).copyTo(rvec);
                    tvecs.row(0).copyTo(tvec);
                    
                    // Convert to RGB and draw axis
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
                    
                    // Save marker with frame
                    api.saveMatImage(imageWithFrame, "marker_0_with_frame.png");
                    
                    // Process crop region and return enhanced image with custom sizes
                    Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec, cropWarpSize, resizeSize);
                    
                    // Clean up
                    rvec.release();
                    tvec.release();
                    imageWithFrame.release();
                    cameraMatrix.release();
                    distCoeffs.release();
                    rvecs.release();
                    tvecs.release();
                    
                    // Clean up corners and ids
                    ids.release();
                    for (Mat corner : corners) {
                        corner.release();
                    }
                    
                    return processedImage;
                }
                
                // Clean up if pose estimation failed
                imageWithFrame.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
            } else {
                Log.w(TAG, "No ArUco markers detected in image");
            }
            
            // Clean up
            ids.release();
            for (Mat corner : corners) {
                corner.release();
            }
            
            return null; // No markers detected
            
        } catch (Exception e) {
            Log.e(TAG, "Error in imageEnhanceAndCrop: " + e.getMessage());
            return null;
        }
    }

    /**
     * Helper method to process the crop region and apply CLAHE + binarization
     */
    private Mat processCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize, Size resizeSize) {
        try {
            // Define crop area corners in 3D (manually adjusted)
            // org.opencv.core.Point3[] cropCorners3D = {
            //     new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
            //     new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
            //     new org.opencv.core.Point3(-0.2325, -0.1125, 0),  // Bottom-right
            //     new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
            // };
            org.opencv.core.Point3[] cropCorners3D = {
                new org.opencv.core.Point3(-0.0265, 0.0420, 0),    // Top-left
                new org.opencv.core.Point3(-0.2385, 0.0420, 0),   // Top-right  
                new org.opencv.core.Point3(-0.2385, -0.1170, 0),  // Bottom-right
                new org.opencv.core.Point3(-0.0265, -0.1170, 0)   // Bottom-left
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
                Mat processedImage = cropEnhanceAndBinarize(image, cropPoints2D, cropWarpSize, resizeSize);
                
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
     * @param image Input image
     * @param cropPoints2D 2D points for perspective transformation
     * @param cropWarpSize Size for the cropped/warped image (configurable)
     * @param resizeSize Size for the final processed image (configurable)
     */
    private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize) {
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
            
            // Save the cropped image with dynamic filename
            String cropFilename = String.format("cropped_region_%.0fx%.0f.png", cropWarpSize.width, cropWarpSize.height);
            api.saveMatImage(croppedImage, cropFilename);
            Log.i(TAG, "Cropped region saved as " + cropFilename);
            
            // ========================================================================
            // STEP 2: Resize to final processing size (configurable)
            // ========================================================================
            
            // Resize the cropped image to final size
            Mat resizedImage = new Mat();
            Imgproc.resize(croppedImage, resizedImage, resizeSize);
            
            // Save resized image
            String resizeFilename = String.format("yolo_original_%.0fx%.0f.png", resizeSize.width, resizeSize.height);
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
            
            // Save the Otsu binarized image with dynamic filename
            String binaryFilename = String.format("yolo_binary_otsu_%.0fx%.0f.png", resizeSize.width, resizeSize.height);
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

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }
}