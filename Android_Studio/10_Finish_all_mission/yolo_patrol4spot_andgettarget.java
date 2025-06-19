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
import java.util.Collections; // Added for sorting
import java.util.Random; // Added for fault tolerance

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;

// Enum for Area Types
enum AreaEnum {
    AREA1(1), AREA2(2), AREA3(3), AREA4(4), UNKNOWN(0);

    private final int areaId;

    AreaEnum(int areaId) {
        this.areaId = areaId;
    }

    public int getAreaId() {
        return areaId;
    }
}

// Class to hold ArTag detection data
class ArTagDetectionData {
    Mat corners;
    int markerId;

    public ArTagDetectionData(Mat corners, int markerId) {
        this.corners = corners.clone(); // Clone to ensure data integrity
        this.markerId = markerId;
    }

    public void releaseCorners() {
        if (corners != null) {
            corners.release();
        }
    }
}
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    private final String TAG = this.getClass().getSimpleName();
    private Random randomGenerator; // Added for fault tolerance

    // Constants for landmark detection
    private static final String[] ALL_LANDMARK_TYPES = {
            "coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box"
    };
    private static final float LANDMARK_CONFIDENCE_THRESHOLD = 0.5f;

    private final int ASTRONAUT_AR_TAG_ID = 50; // Placeholder ID

    private final AreaEnum[] areaEnumsByIndex = {AreaEnum.AREA1, AreaEnum.AREA2, AreaEnum.AREA3, AreaEnum.AREA4};
    // private Map<Integer, AreaEnum> markerToAreaMap = new HashMap<>(); // Removed

    // Instance variables to store detection results across areas
    private Set<String> foundTreasures = new HashSet<>();
    private Set<String> foundLandmarks = new HashSet<>();  // Add this line
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<>();


    // Area coordinates and orientations for all 4 areas 
    private final Point[] AREA_POINTS = {
            new Point(10.95d, -9.78d, 5.195d),         // Area 1
            new Point(10.925d, -8.875d, 4.56203d),     // Area 2 
            new Point(10.925d, -7.925d, 4.56093d),     // Area 3
            new Point(10.666984d, -6.8525d, 4.945d)    // Area 4
    };

    private final Quaternion[] AREA_QUATERNIONS = {
            new Quaternion(0f, 0f, -0.707f, 0.707f), // Area 1
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 2
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 3
            new Quaternion(0f, 0f, 1f, 0f)           // Area 4
    };

    // Oasis Zone coordinates and orientations
    private final Point[] OASIS_ZONE_POINTS = {
            new Point(11.42, -9.50, 4.94),         // Oasis Zone 1
            new Point(11.42, -8.50, 4.94),         // Oasis Zone 2 (Placeholder)
            new Point(11.00, -7.50, 4.50),         // Oasis Zone 3 (Placeholder)
            new Point(10.70, -6.50, 4.50)          // Oasis Zone 4 (Placeholder)
    };

    private final Quaternion[] OASIS_ZONE_QUATERNIONS = {
            AREA_QUATERNIONS[1],                   // OZ1 facing Area 2
            AREA_QUATERNIONS[2],                   // OZ2 facing Area 3
            AREA_QUATERNIONS[3],                   // OZ3 facing Area 4
            new Quaternion(0f, 0f, 0.707f, 0.707f) // OZ4 facing Astronaut
    };

    public static class LandmarkDetectionResult {
        public String bestLandmarkName;
        public float bestLandmarkConfidence;
        public int bestLandmarkCount;
        public List<YOLODetectionService.FinalDetection> allLandmarkDetections;
        public Map<String, Integer> rawLandmarkQuantities; // Original structure from YOLO
        public Set<String> treasureTypes;

        public LandmarkDetectionResult() {
            this.bestLandmarkName = null;
            this.bestLandmarkConfidence = 0.0f;
            this.bestLandmarkCount = 0;
            this.allLandmarkDetections = new ArrayList<>();
            this.rawLandmarkQuantities = new HashMap<>();
            this.treasureTypes = new HashSet<>();
        }
    }

    @Override
    protected void runPlan1(){
        // markerToAreaMap removed, no longer needed for initialization here.

        Set<AreaEnum> completedAreas = new HashSet<>();

        // Log the start of the mission.
        Log.i(TAG, "Start mission");

        // IMPORTANT: Mission time is critical (5-minute limit).
        // The delays added for Oasis Zones (e.g., Thread.sleep(500)) should be adjusted
        // based on testing to maximize Oasis Bonus without exceeding the total time.
        // Monitor the overall mission duration carefully.

        // The mission starts.
        api.startMission();

        // Initialize area treasure tracking
        Map<Integer, Set<String>> areaTreasure = new HashMap<>();
        for (int i = 1; i <= 4; i++) {
            areaTreasure.put(i, new HashSet<String>());
        }

        // ========================================================================
        // CONFIGURABLE IMAGE PROCESSING PARAMETERS - EDIT HERE
        // ========================================================================

        Size cropWarpSize = new Size(640, 480);   // Size for cropped/warped image
        Size resizeSize = new Size(320, 320);     // Size for final processing

        // ========================================================================
        // PROCESS ALL 4 AREAS
        // ========================================================================

        // Loop through all 4 areas
        for (int areaIndex = 0; areaIndex < 4; areaIndex++) {
            AreaEnum currentViewpointArea = areaEnumsByIndex[areaIndex]; // Current area based on loop index
            int currentViewpointApiId = currentViewpointArea.getAreaId(); // API ID for current viewpoint area

            Log.i(TAG, "=== Considering Viewpoint for " + currentViewpointArea + " (API ID: " + currentViewpointApiId + ") ===");

            if (completedAreas.contains(currentViewpointArea)) {
                Log.i(TAG, currentViewpointArea + " (API ID: " + currentViewpointApiId + ") is already completed. Skipping navigation and processing for this viewpoint.");
                continue;
            }

            // Move to the area
            Point targetPoint = AREA_POINTS[areaIndex];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaIndex];

            Log.i(TAG, String.format("Moving to viewpoint of %s: Point(%.3f, %.3f, %.3f)",
                    currentViewpointArea, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            api.moveTo(targetPoint, targetQuaternion, false);

            // Get a camera image
            Mat image = api.getMatNavCam();

            // Process the image for the currentViewpointArea
            Mat claHeBinImage = imageEnhanceAndCrop(image, cropWarpSize, resizeSize, currentViewpointArea);

            if (claHeBinImage != null) {
                Log.i(TAG, "Image enhancement and cropping successful for " + currentViewpointArea);

                // Detect items using YOLO
                LandmarkDetectionResult landmarkDetectionResult = detectitemfromcvimg(
                        claHeBinImage,
                        0.5f,      // conf_threshold
                        "lost",    // img_type ("lost" or "target")
                        0.45f,     // standard_nms_threshold
                        0.8f,      // overlap_nms_threshold
                        320        // img_size
                );

                Map<String, Integer> landmark_items = landmarkDetectionResult.rawLandmarkQuantities; // Using raw quantities for areaLandmarks
                Set<String> treasure_types = landmarkDetectionResult.treasureTypes;
                String detectedLandmarkName = landmarkDetectionResult.bestLandmarkName;
                float detectedLandmarkConfidence = landmarkDetectionResult.bestLandmarkConfidence;
                int detectedLandmarkCount = landmarkDetectionResult.bestLandmarkCount;
                List<YOLODetectionService.FinalDetection> allFoundLandmarkCandidates = landmarkDetectionResult.allLandmarkDetections;

                Log.i(TAG, currentViewpointArea + " - Best Landmark: " + detectedLandmarkName + " (Conf: " + detectedLandmarkConfidence + ", Count: " + detectedLandmarkCount + ")");
                Log.i(TAG, currentViewpointArea + " - Raw Landmark quantities: " + landmark_items);
                Log.i(TAG, currentViewpointArea + " - Treasure types: " + treasure_types);
                Log.i(TAG, currentViewpointArea + " - All landmark candidates: " + (allFoundLandmarkCandidates != null ? allFoundLandmarkCandidates.size() : 0));


                // Store results for later use
                areaLandmarks.put("area" + currentViewpointApiId, landmark_items); // Use currentViewpointApiId
                foundTreasures.addAll(treasure_types);
                // foundLandmarks.addAll(landmark_items.keySet()); // REMOVED - will be populated by new logic

                // Ensure areaTreasure has an entry for this areaId
                if (!areaTreasure.containsKey(currentViewpointApiId)) {
                     areaTreasure.put(currentViewpointApiId, new HashSet<String>());
                }
                areaTreasure.get(currentViewpointApiId).addAll(treasure_types);

                Log.i(TAG, currentViewpointArea + " treasure types: " + areaTreasure.get(currentViewpointApiId));

                // Clean up the processed image
                claHeBinImage.release();

                // SET AREA INFO FOR THIS AREA - New Fault Tolerance Logic
                String landmarkToReport = "unknown"; // Default
                int countToReport = 0; // Default

                if (detectedLandmarkName != null && detectedLandmarkConfidence >= LANDMARK_CONFIDENCE_THRESHOLD) {
                    landmarkToReport = detectedLandmarkName;
                    countToReport = detectedLandmarkCount > 0 ? detectedLandmarkCount : 1; // Ensure count is at least 1
                    Log.i(TAG, "High confidence detection for " + currentViewpointArea + ": " + landmarkToReport + " (conf: " + detectedLandmarkConfidence + "), count: " + countToReport);
                } else {
                    Log.w(TAG, currentViewpointArea + ": Landmark detection failed, confidence low (" + detectedLandmarkConfidence + " for " + detectedLandmarkName + "), or no landmark detected. Activating fault tolerance.");

                    String guessType = null;

                    // Strategy B: Highest confidence candidate from current image, if not already reported
                    if (allFoundLandmarkCandidates != null && !allFoundLandmarkCandidates.isEmpty()) {
                        // allFoundLandmarkCandidates is already sorted by confidence
                        for (YOLODetectionService.FinalDetection candidate : allFoundLandmarkCandidates) {
                            String candidateName = YOLODetectionService.getClassName(candidate.classId);
                            if (candidateName != null && !foundLandmarks.contains(candidateName)) {
                                guessType = candidateName;
                                Log.i(TAG, "Fault Tolerance (Strategy B) for " + currentViewpointArea + ": Selected non-reported candidate '" + guessType + "' (conf: " + candidate.confidence + ")");
                                break;
                            }
                        }
                    }

                    // Strategy A: Random unreported if Strategy B failed or yielded no new item
                    if (guessType == null) {
                        List<String> unreportedLandmarks = getUnreportedLandmarkTypes(foundLandmarks);
                        if (!unreportedLandmarks.isEmpty()) {
                            if (this.randomGenerator == null) { // Initialize if null
                                this.randomGenerator = new Random();
                            }
                            guessType = unreportedLandmarks.get(this.randomGenerator.nextInt(unreportedLandmarks.size()));
                            Log.i(TAG, "Fault Tolerance (Strategy A) for " + currentViewpointArea + ": Selected random unreported landmark '" + guessType + "' from " + unreportedLandmarks.size() + " options.");
                        }
                    }

                    // Apply guess or fallback
                    if (guessType != null) {
                        landmarkToReport = guessType;
                        countToReport = 1; // Guesses are always reported with count 1
                    } else if (detectedLandmarkName != null) {
                        // Fallback to original low-confidence detection if no guess could be made
                        landmarkToReport = detectedLandmarkName;
                        countToReport = detectedLandmarkCount > 0 ? detectedLandmarkCount : 1;
                        Log.w(TAG, "Fault Tolerance (Fallback) for " + currentViewpointArea + ": No suitable guess. Using original low-confidence detection '" + landmarkToReport + "'");
                    } else {
                        // Absolute fallback: if no detection and no guess, pick the first from ALL_LANDMARK_TYPES or a default.
                        if (ALL_LANDMARK_TYPES.length > 0) {
                            landmarkToReport = ALL_LANDMARK_TYPES[0]; // Default to the first known landmark
                            countToReport = 1;
                            Log.w(TAG, "Fault Tolerance (Absolute Fallback) for " + currentViewpointArea + ": No detection and no guess. Defaulting to '" + landmarkToReport + "'");
                        } else {
                            landmarkToReport = "unknown"; // Final fallback
                            countToReport = 0;
                            Log.e(TAG, "Fault Tolerance (Critical Fallback): ALL_LANDMARK_TYPES is empty! Reporting 'unknown'.");
                        }
                    }
                }

                // Now, report the chosen landmark and count
                if (!"unknown".equals(landmarkToReport) || countToReport > 0) {
                     api.setAreaInfo(currentViewpointApiId, landmarkToReport, countToReport);
                     Log.i(TAG, String.format("Set AreaInfo for %s (API ID: %d): %s x %d. Confidence was: %.2f",
                             currentViewpointArea.name(), currentViewpointApiId, landmarkToReport, countToReport, detectedLandmarkConfidence));
                     foundLandmarks.add(landmarkToReport); // Add the *actually reported* landmark to foundLandmarks
                } else {
                     api.setAreaInfo(currentViewpointApiId, "unknown", 0); // Explicitly report unknown, 0
                     Log.w(TAG, currentViewpointArea + ": All detection and fallback strategies failed. Reporting 'unknown' for AreaInfo.");
                }
                completedAreas.add(currentViewpointArea); // Mark this area as completed
                Log.i(TAG, currentViewpointArea + " (API ID: " + currentViewpointApiId + ") marked as completed.");

            } else {
                Log.w(TAG, currentViewpointArea + ": Image enhancement failed or no suitable marker found.");
                // If processing failed for this viewpoint area and it's not yet completed,
                // set default info and mark as completed (attempted).
                if (!completedAreas.contains(currentViewpointArea)) {
                    api.setAreaInfo(currentViewpointApiId, "unknown_viewpoint_miss", 0);
                    completedAreas.add(currentViewpointArea);
                    Log.i(TAG, currentViewpointArea + " (API ID: " + currentViewpointApiId + ") marked as completed (attempted/missed from own viewpoint).");
                }
            }

            // Clean up original image
            image.release();

            // Short delay between areas to ensure stability
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted");
            }

            // Navigate to Oasis Zones after respective Area processing
            if (areaIndex == 0) {
                // Navigate to Oasis Zone 1
                Log.i(TAG, "Moving to Oasis Zone 1");
                api.moveTo(OASIS_ZONE_POINTS[0], OASIS_ZONE_QUATERNIONS[0], false);
                Log.i(TAG, "Arrived at Oasis Zone 1, pausing for bonus.");
                try {
                    Thread.sleep(500); // Configurable delay for Oasis Zone bonus
                } catch (InterruptedException e) {
                    Log.w(TAG, "Sleep interrupted in Oasis Zone 1");
                }
            } else if (areaIndex == 1) {
                // Navigate to Oasis Zone 2
                Log.i(TAG, "Moving to Oasis Zone 2");
                api.moveTo(OASIS_ZONE_POINTS[1], OASIS_ZONE_QUATERNIONS[1], false);
                Log.i(TAG, "Arrived at Oasis Zone 2, pausing for bonus.");
                try {
                    Thread.sleep(500); // Configurable delay for Oasis Zone bonus
                } catch (InterruptedException e) {
                    Log.w(TAG, "Sleep interrupted in Oasis Zone 2");
                }
            } else if (areaIndex == 2) {
                // Navigate to Oasis Zone 3
                Log.i(TAG, "Moving to Oasis Zone 3");
                api.moveTo(OASIS_ZONE_POINTS[2], OASIS_ZONE_QUATERNIONS[2], false);
                Log.i(TAG, "Arrived at Oasis Zone 3, pausing for bonus.");
                try {
                    Thread.sleep(500); // Configurable delay for Oasis Zone bonus
                } catch (InterruptedException e) {
                    Log.w(TAG, "Sleep interrupted in Oasis Zone 3");
                }
            }
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
        Log.i(TAG, "All found landmarks: " + foundLandmarks);  // Add this line

        // Navigate to Oasis Zone 4 before moving to Astronaut
        Log.i(TAG, "Moving to Oasis Zone 4");
        api.moveTo(OASIS_ZONE_POINTS[3], OASIS_ZONE_QUATERNIONS[3], false);
        Log.i(TAG, "Arrived at Oasis Zone 4, pausing for bonus.");
        try {
            Thread.sleep(500); // Configurable delay for Oasis Zone bonus
        } catch (InterruptedException e) {
            Log.w(TAG, "Sleep interrupted in Oasis Zone 4");
        }

        // ========================================================================
        // ASTRONAUT INTERACTION
        // ========================================================================

        // Move to the front of the astronaut and report rounding completion
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d); // Defined earlier in the code
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f); // Defined earlier

        Log.i(TAG, "Moving to astronaut position with precision enhancement...");
        // api.moveTo(astronautPoint, astronautQuaternion, false); // Original call replaced by loop

        int maxMoveAttempts = 3; // Max attempts for iterative moveTo
        double targetTolerance = 0.10; // Target tolerance in meters (10cm) before AR tag tuning
        double currentDistance = Double.MAX_VALUE;
        boolean initialMoveSuccessful = false;

        for (int attempt = 0; attempt < maxMoveAttempts; attempt++) {
            Log.i(TAG, String.format("Attempt %d to reach astronaut position.", attempt + 1));
            // Use a blocking call for moveTo
            api.moveTo(astronautPoint, astronautQuaternion, true); // Blocking call

            // Get current position after move
            Point currentPosition = api.getRobotKinematics().getPosition();
            if (currentPosition == null) {
                Log.w(TAG, "Failed to get robot position after move attempt.");
                // Consider a short sleep and retry or break
                try { Thread.sleep(200); } catch (InterruptedException e) { Log.w(TAG, "Sleep interrupted"); }
                continue;
            }

            // Calculate distance to target
            double dx = astronautPoint.getX() - currentPosition.getX();
            double dy = astronautPoint.getY() - currentPosition.getY();
            double dz = astronautPoint.getZ() - currentPosition.getZ();
            currentDistance = Math.sqrt(dx*dx + dy*dy + dz*dz);

            Log.i(TAG, String.format("Distance to astronaut target: %.3fm", currentDistance));

            if (currentDistance <= targetTolerance) {
                Log.i(TAG, "Reached astronaut position within tolerance.");
                initialMoveSuccessful = true;
                break; // Exit loop if within tolerance
            } else {
                Log.i(TAG, String.format("Not within tolerance (%.2fm). Current distance: %.3fm. Attempting relative move.", targetTolerance, currentDistance));
                // Calculate relative move needed (robot's frame of reference for relativeMoveTo is forward, left, up)
                // This requires transforming the world delta (dx, dy, dz) to robot's frame or simply moving along world axes if Astrobee's relativeMoveTo behaves that way for small adjustments without orientation change.
                // For simplicity, assuming small adjustments along world axes if robot is already facing astronaut.
                // A more precise relative move would require transforming dx,dy,dz to robot's coordinate system.
                // However, the issue description implies using relativeMoveTo with world-based delta if already close: "api.getRobotPosition() 獲取自身位置，也能計算與Astronaut的差向量，用 relativeMoveTo() 做最後的修正移動"
                // Assuming api.relativeMoveTo(dx, dy, dz, ...) for small corrections means dx (world X), dy (world Y), dz (world Z) if Astrobee is roughly aligned.
                // More robust: dx_robot_frame, dy_robot_frame, dz_robot_frame.
                // Given the context, let's try a direct relative move with world differences.
                // The `api.relativeMoveTo` expects dx (forward), dy (left), dz (up) in robot frame.
                // The calculated dx, dy, dz are in world frame.
                // A proper solution would get robot's current quaternion, invert it, and rotate the world vector (dx,dy,dz).
                // For now, as a simpler heuristic for small corrections when already mostly facing the target:
                // If we are mostly facing the astronaut (which we should be after moveTo),
                // a positive world Z difference (dz) might mean we need to move forward (positive relative dx).
                // a positive world X difference (dx) might mean we are to the "left" of target, so need to move "right" (negative relative dy).
                // a positive world Y difference (dy) might mean we are "behind" target (in world Y), so need to move "forward" in Y (could be relative dx or dy based on orientation)

                // Given the complexity of frame transformation, let's use a simpler approach:
                // Re-issue a moveTo. If the first moveTo with 'true' doesn't get us close enough,
                // a relative move without proper frame transformation might be risky.
                // The issue example `result = api.moveTo(astronautPoint, astronautQuat, true); loopCounter++; } while(!result.hasSucceeded() && loopCounter < 3);`
                // suggests just retrying moveTo. Let's follow that more closely.

                // So, instead of relativeMoveTo here, we just let the loop retry the absolute moveTo.
                Log.i(TAG, "Retrying absolute moveTo.");
                if (attempt < maxMoveAttempts -1) { // If not the last attempt
                     try { Thread.sleep(500); } catch (InterruptedException e) { Log.w(TAG, "Sleep interrupted during retry delay"); }
                }
            }
        }

        if (!initialMoveSuccessful) {
            Log.w(TAG, String.format("Failed to reach astronaut position within tolerance after %d attempts. Last distance: %.3fm", maxMoveAttempts, currentDistance));
        }

        // The AR Tag fine-tuning logic added in Step 1 will follow this block.
        // AR Tag fine-tuning logic START (This line is already present from previous step)
        Log.i(TAG, "Attempting AR Tag fine-tuning for astronaut position.");
        Mat astronautViewImage = api.getMatNavCam(); // Get image for AR detection

        if (astronautViewImage != null && !astronautViewImage.empty()) {
            Dictionary arucoDictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            Aruco.detectMarkers(astronautViewImage, arucoDictionary, corners, ids);

            boolean astronautTagFound = false;
            int targetTagIndex = -1;

            if (ids.rows() > 0) {
                for (int i = 0; i < ids.rows(); i++) {
                    if (ids.get(i,0) != null && (int)ids.get(i,0)[0] == ASTRONAUT_AR_TAG_ID) {
                        astronautTagFound = true;
                        targetTagIndex = i;
                        Log.i(TAG, "Astronaut AR Tag ID " + ASTRONAUT_AR_TAG_ID + " found.");
                        break;
                    }
                }
            }

            if (astronautTagFound) {
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                cameraMatrix.put(0, 0, intrinsics[0]); // fx, fy, cx, cy
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
                distCoeffs.put(0, 0, intrinsics[1]);   // k1, k2, p1, p2, k3

                List<Mat> targetCornerList = new ArrayList<>();
                targetCornerList.add(corners.get(targetTagIndex).clone()); // Clone to avoid modifying original list elements

                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f; // Assumed marker length in meters

                Aruco.estimatePoseSingleMarkers(targetCornerList, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat tvec = tvecs.row(0); // Translation vector: [x, y, z] of tag in camera frame
                    // Camera frame: +X right, +Y down, +Z forward (into scene)

                    double tag_x_cam = tvec.get(0,0)[0]; // Tag's X position in camera frame
                    double tag_y_cam = tvec.get(0,0)[1]; // Tag's Y position in camera frame
                    double tag_z_cam = tvec.get(0,0)[2]; // Tag's Z position (depth) in camera frame

                    double desired_z_distance_to_tag = 0.2; // Target 0.2m from the tag

                    // Robot's relativeMoveTo(forward, left, up, ...)
                    // To center the tag in view (tag_x_cam = 0, tag_y_cam = 0) and achieve desired_z_distance_to_tag:
                    // 1. Move forward by (tag_z_cam - desired_z_distance_to_tag)
                    //    - If tag_z_cam > desired_z_distance_to_tag, robot is too far, move forward (positive dx_robot_relative).
                    double dx_robot_relative = tag_z_cam - desired_z_distance_to_tag;

                    // 2. Move left by tag_x_cam
                    //    - If tag_x_cam > 0 (tag is to the right in image), robot needs to move left (positive dy_robot_relative).
                    double dy_robot_relative = tag_x_cam;

                    // 3. Move up by -tag_y_cam
                    //    - If tag_y_cam > 0 (tag is down in image), robot needs to move up (positive dz_robot_relative).
                    double dz_robot_relative = -tag_y_cam;

                    Log.i(TAG, String.format("AR Tag raw offsets: tag_x_cam=%.3f, tag_y_cam=%.3f, tag_z_cam=%.3f", tag_x_cam, tag_y_cam, tag_z_cam));
                    Log.i(TAG, String.format("AR Tag Adjustment: Calculated relative move: dx=%.3f (forward), dy=%.3f (left), dz=%.3f (up)", dx_robot_relative, dy_robot_relative, dz_robot_relative));

                    // Apply safety limits to the relative move distances
                    dx_robot_relative = Math.max(-0.3, Math.min(0.3, dx_robot_relative)); // Limit forward/backward
                    dy_robot_relative = Math.max(-0.2, Math.min(0.2, dy_robot_relative)); // Limit left/right
                    dz_robot_relative = Math.max(-0.2, Math.min(0.2, dz_robot_relative)); // Limit up/down

                    Log.i(TAG, String.format("AR Tag Adjustment: Performing clamped relative move: dx=%.3f, dy=%.3f, dz=%.3f", dx_robot_relative, dy_robot_relative, dz_robot_relative));

                    if (Math.abs(dx_robot_relative) > 0.02 || Math.abs(dy_robot_relative) > 0.02 || Math.abs(dz_robot_relative) > 0.02) { // Only move if significant
                        api.relativeMoveTo(dx_robot_relative, dy_robot_relative, dz_robot_relative, 0, 0, 0, true); // Blocking relative move
                        Log.i(TAG, "Relative move for AR Tag adjustment completed.");
                    } else {
                        Log.i(TAG, "AR Tag adjustment too small, skipping relative move.");
                    }

                } else {
                    Log.w(TAG, "Could not estimate pose of Astronaut AR Tag ID " + ASTRONAUT_AR_TAG_ID);
                }
                // Release OpenCV Mats
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
                for(Mat c : targetCornerList) c.release(); // Release the cloned corner
            } else {
                Log.i(TAG, "Astronaut AR Tag ID " + ASTRONAUT_AR_TAG_ID + " not found in current view. Skipping AR fine-tuning.");
            }

            // Release general Mats from detection
            for(Mat c : corners) c.release();
            ids.release();
            astronautViewImage.release();
        } else {
            Log.w(TAG, "Failed to get NavCam image for Astronaut AR Tag fine-tuning or image was empty.");
        }
        // AR Tag fine-tuning logic END

        api.reportRoundingCompletion(); // This call remains, now after the adjustment logic.

        // Commented out old astronaut marker check, as AR fine-tuning is now done before reporting.
        // Log.i(TAG, "Moving to astronaut position");
        // api.moveTo(astronautPoint, astronautQuaternion, false); // This is where the new code is inserted after
        // api.reportRoundingCompletion(); // This is after the new code

        // // Error handling verify markers are visible before proceeding <<<< START OF BLOCK TO COMMENT
        // boolean astronautMarkersOk = waitForMarkersDetection(2000, 200, "astronaut");

        // if (astronautMarkersOk) {
        //     Log.i(TAG, "Astronaut markers confirmed - proceeding with target detection");
        // } else {
        //     Log.w(TAG, "Astronaut markers not detected - proceeding anyway");
        // } // <<<< END OF BLOCK TO COMMENT

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
                api.moveTo(targetAreaPoint, targetAreaQuaternion, false);

                // Take a snapshot of the target item
                api.takeTargetItemSnapshot();

                Log.i(TAG, "Mission completed successfully!");
            } else {
                Log.w(TAG, "Target treasure '" + targetTreasureType + "' not found in any area");
                api.notifyRecognitionItem();
                api.takeTargetItemSnapshot();
            }
        } else {
            Log.w(TAG, "Could not identify target treasure from astronaut");
            api.notifyRecognitionItem();
            api.takeTargetItemSnapshot();
        }

        // Clean up target image
        targetImage.release();
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
     * Process target image to identify the treasure type the astronaut is holding
     * @param targetImage Image from astronaut
     * @param resizeSize Processing size
     * @return Treasure type name or "unknown"
     */
    private String processTargetImage(Mat targetImage, Size resizeSize) {
        try {
            Log.i(TAG, "Processing target image from astronaut");

            // Save the target image for debugging
            api.saveMatImage(targetImage, "target_astronaut_raw.png");

            // Use the SAME processing pipeline as areas (ArUco detection + cropping + enhancement)
            Size cropWarpSize = new Size(640, 480);   // Same as area processing
            // Pass AreaEnum.UNKNOWN for target processing, as it's not a numbered area.
            // imageEnhanceAndCrop now returns a single Mat.
            Mat processedTarget = imageEnhanceAndCrop(targetImage, cropWarpSize, resizeSize, AreaEnum.UNKNOWN);

            if (processedTarget != null) {
                Log.i(TAG, "Target image processing successful - marker detected and cropped (associated with UNKNOWN viewpoint).");

                // Detect items using YOLO with "target" type - SAME as area processing
                Object[] detected_items = detectitemfromcvimg(
                        processedTarget,
                        0.3f,      // Lower confidence for target detection
                        "target",  // img_type for target
                        0.45f,     // standard_nms_threshold
                        0.8f,      // overlap_nms_threshold
                        320        // img_size
                );

                // Extract results - SAME as area processing
                Map<String, Integer> landmark_items = (Map<String, Integer>) detected_items[0];
                Set<String> treasure_types = (Set<String>) detected_items[1];

                Log.i(TAG, "Target - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Target - Treasure types: " + treasure_types);

                if (!treasure_types.isEmpty()) {
                    String targetTreasure = treasure_types.iterator().next(); // Assuming only one treasure type on target
                    Log.i(TAG, "Target treasure detected: " + targetTreasure);
                    // processedTarget is released below
                    // No need to return here, let it flow to final release
                }
                // processedTarget is released outside this if/else block if it's not null
            }
            // Removed the specific 'else' for processedTarget == null here,
            // as the outer 'if (processedTarget != null)' handles the case where it's null from the start.
            // The log "Target image processing - no markers detected or map was empty." covers this.

            if (processedTarget != null) {
                processedTarget.release(); // General release for processedTarget if it was obtained
            }

            // Determine return value based on treasure_types from YOLO
            // This part of the logic for extracting treasure_types from detected_items remains,
            // but it's now inside the if (processedTarget != null) block.
            // If processedTarget was null, treasure_types would not have been populated.
            Map<String, Integer> landmark_items = new HashMap<>(); // ensure initialized
            Set<String> treasure_types = new HashSet<>(); // ensure initialized

            if (processedTarget != null) { // This check is redundant if YOLO call is inside, but good for clarity
                 Object[] detected_items = detectitemfromcvimg(
                        processedTarget,
                        0.3f,      // Lower confidence for target detection
                        "target",  // img_type for target
                        0.45f,     // standard_nms_threshold
                        0.8f,      // overlap_nms_threshold
                        320        // img_size
                );
                landmark_items = (Map<String, Integer>) detected_items[0]; //
                treasure_types = (Set<String>) detected_items[1];
                Log.i(TAG, "Target - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Target - Treasure types: " + treasure_types);
                processedTarget.release(); // Release after YOLO
            }


            if (treasure_types != null && !treasure_types.isEmpty()){
                return treasure_types.iterator().next();
            } else {
                Log.w(TAG, "No treasure detected in target image after all processing.");
                return "unknown";
            }

        } catch (Exception e) {
            Log.e(TAG, "Error processing target image: " + e.getMessage(), e);
            // If processedTarget was obtained but an error occurred later, ensure it's released.
            // This is a simplified catch; a more robust version would handle this more gracefully.
            // if (processedTarget != null && !processedTarget.empty()) { // Check if Mat is valid before release
            //     processedTarget.release();
            // }
            return "unknown";
        }
    }

    /**
     * Basic enhancement for target image (simpler than area processing)
     */
    private Mat enhanceTargetImage(Mat image, Size resizeSize) {
        try {
            // Resize to processing size
            Mat resized = new Mat();
            Imgproc.resize(image, resized, resizeSize);

            // Apply basic CLAHE enhancement
            Mat enhanced = new Mat();
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);
            clahe.setTilesGridSize(new Size(8, 8));
            clahe.apply(resized, enhanced);

            // Save enhanced target for debugging
            api.saveMatImage(enhanced, "target_astronaut_enhanced.png");

            resized.release();
            return enhanced;

        } catch (Exception e) {
            Log.e(TAG, "Error enhancing target image: " + e.getMessage());
            return null;
        }
    }

    /**
     * Find which area contains the specified treasure type
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
     * Method to detect items from CV image using YOLO - matches Python testcallyololib.py functionality
     * @param image Input OpenCV Mat image
     * @param conf Confidence threshold (e.g., 0.3f)
     * @param imgtype Image type: "lost" or "target"
     * @param standard_nms_threshold Standard NMS threshold (e.g., 0.45f)
     * @param overlap_nms_threshold Overlap NMS threshold for intelligent NMS (e.g., 0.8f)
     * @param img_size Image size for processing (e.g., 320)
     * @return LandmarkDetectionResult object containing detailed detection information.
     */
    private LandmarkDetectionResult detectitemfromcvimg(Mat image, float conf, String imgtype,
                                                        float standard_nms_threshold, float overlap_nms_threshold, int img_size) {
        YOLODetectionService yoloService = null;
        LandmarkDetectionResult landmarkDetectionResult = new LandmarkDetectionResult(); // Renamed to avoid conflict

        try {
            Log.i(TAG, String.format("Starting YOLO detection - type: %s, conf: %.2f", imgtype, conf));

            yoloService = new YOLODetectionService(this);
            // Assign to yoloServiceResult, and use landmarkDetectionResult for the method's return object
            YOLODetectionService.EnhancedDetectionResult yoloServiceResult = yoloService.DetectfromcvImage(
                    image, imgtype, conf, standard_nms_threshold, overlap_nms_threshold
            );

            Map<String, Object> pythonResult = yoloServiceResult.getPythonLikeResult();

            // Populate rawLandmarkQuantities
            Map<String, Integer> rawLandmarks = (Map<String, Integer>) pythonResult.get("landmark_quantities");
            if (rawLandmarks != null) {
                landmarkDetectionResult.rawLandmarkQuantities.putAll(rawLandmarks);
            }

            // Populate treasureTypes
            Map<String, Integer> treasureQuantitiesMap = (Map<String, Integer>) pythonResult.get("treasure_quantities");
            if (treasureQuantitiesMap != null) {
                landmarkDetectionResult.treasureTypes.addAll(treasureQuantitiesMap.keySet());
            }

            // Populate allLandmarkDetections
            if (yoloServiceResult.getDetections() != null) {
                for (YOLODetectionService.FinalDetection detection : yoloServiceResult.getDetections()) {
                    if (YOLODetectionService.LANDMARK_IDS.contains(detection.classId)) {
                        landmarkDetectionResult.allLandmarkDetections.add(detection);
                    }
                }
            }

            // Sort landmarks by confidence (descending)
            if (landmarkDetectionResult.allLandmarkDetections != null) {
                 Collections.sort(landmarkDetectionResult.allLandmarkDetections, (d1, d2) -> Float.compare(d2.confidence, d1.confidence));
            }

            // Determine best landmark and its count
            if (landmarkDetectionResult.allLandmarkDetections != null && !landmarkDetectionResult.allLandmarkDetections.isEmpty()) {
                YOLODetectionService.FinalDetection topLandmarkDetection = landmarkDetectionResult.allLandmarkDetections.get(0);
                landmarkDetectionResult.bestLandmarkName = YOLODetectionService.getClassName(topLandmarkDetection.classId);
                landmarkDetectionResult.bestLandmarkConfidence = topLandmarkDetection.confidence;

                int count = 0;
                for (YOLODetectionService.FinalDetection ld : landmarkDetectionResult.allLandmarkDetections) {
                    if (ld.classId == topLandmarkDetection.classId) {
                        count++;
                    }
                }
                landmarkDetectionResult.bestLandmarkCount = count;
            }

            Log.i(TAG, "Raw Landmark quantities: " + landmarkDetectionResult.rawLandmarkQuantities);
            Log.i(TAG, "Treasure types: " + landmarkDetectionResult.treasureTypes);
            Log.i(TAG, "All landmark detections count: " + (landmarkDetectionResult.allLandmarkDetections != null ? landmarkDetectionResult.allLandmarkDetections.size() : 0));
            if (landmarkDetectionResult.bestLandmarkName != null) {
                Log.i(TAG, String.format("Best landmark: %s, Confidence: %.2f, Count: %d",
                        landmarkDetectionResult.bestLandmarkName, landmarkDetectionResult.bestLandmarkConfidence, landmarkDetectionResult.bestLandmarkCount));
            } else {
                Log.i(TAG, "No best landmark identified.");
            }

            return landmarkDetectionResult;

        } catch (Exception e) {
            Log.e(TAG, "Error in detectitemfromcvimg: " + e.getMessage(), e);
            return new LandmarkDetectionResult(); // Return a new, empty object
        } finally {
            if (yoloService != null) {
                yoloService.close();
            }
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
     * @param currentViewpointArea The AreaEnum representing the robot's current viewpoint.
     * @return Processed Mat image for the selected AR tag, or null if none processed.
     */
    private Mat imageEnhanceAndCrop(Mat image, Size cropWarpSize, Size resizeSize, AreaEnum currentViewpointArea) {
        // Initialize ArUco detection related objects once
        Dictionary dictionary = null;
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat(); // Mat to store marker IDs

        try {
            String rawImageFilename = "area" + currentViewpointArea.getAreaId() + "_raw_capture.png";
            api.saveMatImage(image, rawImageFilename);
            Log.i(TAG, "Raw image for " + currentViewpointArea + " saved as " + rawImageFilename);

            dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            Aruco.detectMarkers(image, dictionary, corners, ids);

            if (ids.rows() > 0) {
                Log.i(TAG, "Detected " + ids.rows() + " markers for " + currentViewpointArea);

                List<Mat> selectedCornersList = new ArrayList<>();
                Mat selectedIdsMat = new Mat(); // Will store the ID of the selected marker

                if (ids.rows() == 1) {
                    selectedCornersList.add(corners.get(0).clone()); // Clone to be safe for release later
                    ids.row(0).copyTo(selectedIdsMat); // Copy the first (and only) ID row
                    Log.i(TAG, "Single marker detected, using it. ID: " + selectedIdsMat.get(0,0)[0]);
                } else {
                    // Multiple markers detected, find the one closest to the image center
                    Log.i(TAG, "Multiple markers (" + ids.rows() + ") detected, selecting closest to center for " + currentViewpointArea);
                    double imageCenterX = image.cols() / 2.0;
                    double imageCenterY = image.rows() / 2.0;
                    int closestIndex = -1;
                    double minDistance = Double.MAX_VALUE;

                    for (int i = 0; i < corners.size(); i++) {
                        Mat cornerSet = corners.get(i);
                        if (cornerSet.rows() != 1 || cornerSet.cols() != 4 || cornerSet.channels() != 2) continue; // Should be 1x4 with 2 channels (x,y pairs)

                        float[] cornerData = new float[8]; // 4 points * 2 coords
                        cornerSet.get(0,0, cornerData);
                        double markerCenterX = (cornerData[0] + cornerData[2] + cornerData[4] + cornerData[6]) / 4.0;
                        double markerCenterY = (cornerData[1] + cornerData[3] + cornerData[5] + cornerData[7]) / 4.0;

                        double distance = Math.sqrt(Math.pow(markerCenterX - imageCenterX, 2) + Math.pow(markerCenterY - imageCenterY, 2));
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestIndex = i;
                        }
                    }

                    if (closestIndex != -1) {
                        selectedCornersList.add(corners.get(closestIndex).clone());
                        ids.row(closestIndex).copyTo(selectedIdsMat);
                        Log.i(TAG, "Closest marker selected. Index: " + closestIndex + ", ID: " + selectedIdsMat.get(0,0)[0] + ", Distance: " + minDistance);
                    } else {
                        Log.w(TAG, "Could not select a closest marker for " + currentViewpointArea);
                        // Clean up original corners and ids if no marker is selected
                        for (Mat corner : corners) corner.release();
                        ids.release();
                        return null;
                    }
                }

                // Clean up original corners and ids Mat as they are processed or cloned
                for (Mat corner : corners) corner.release();
                ids.release(); // ids Mat is fully processed or copied

                // Now process the single selected marker (selectedCornersList, selectedIdsMat)
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);
                distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f;

                Aruco.estimatePoseSingleMarkers(selectedCornersList, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

                Mat imageWithFrame = image.clone();
                Aruco.drawDetectedMarkers(imageWithFrame, selectedCornersList, selectedIdsMat); // Draw only the selected marker
                Mat processedImage = null;

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = rvecs.row(0);
                    Mat tvec = tvecs.row(0);

                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
                    // Use currentViewpointArea.getAreaId() for logging and filenames
                    String markerFilename = "area" + currentViewpointArea.getAreaId() + "_marker_" + (selectedIdsMat.empty() ? "unknown" : (int)selectedIdsMat.get(0,0)[0]) + "_with_frame.png";
                    api.saveMatImage(imageWithFrame, markerFilename);
                    Log.i(TAG, "Marker image for " + currentViewpointArea + " saved as " + markerFilename);

                    processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec, cropWarpSize, resizeSize, currentViewpointArea.getAreaId());
                } else {
                    Log.w(TAG, "Pose estimation failed for selected marker in " + currentViewpointArea);
                }

                // Clean up Mats specific to this marker's processing
                for(Mat c : selectedCornersList) c.release(); // Release cloned corners
                selectedIdsMat.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
                imageWithFrame.release();

                return processedImage; // Return the single processed image or null

            } else {
                Log.w(TAG, "No ArUco markers detected in image for " + currentViewpointArea);
                if(ids != null) ids.release();
                for (Mat corner : corners) corner.release();
            }
            return null; // No markers detected

        } catch (Exception e) {
            Log.e(TAG, "Error in imageEnhanceAndCrop for " + currentViewpointArea + ": " + e.getMessage(), e);
            if (ids != null) ids.release();
            for (Mat corner : corners) corner.release();
            return null; // Return null on error
        }
    }

    // Removed inferAreaFromMarker method

    /**
     * Helper method to process the crop region and apply CLAHE + binarization
     */
    private Mat processCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            // Define crop area corners in 3D (manually adjusted)
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

    /**
     * FIXED: Keep only the marker closest to the image center
     * This version properly handles corner data format for ArUco
     * @param corners List of detected marker corners
     * @param ids Mat containing marker IDs
     * @param image Original image (to get center coordinates)
     * @return Object array: [filtered_corners, filtered_ids]
     */
    /*
    private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image) {
        if (corners.size() == 0) {
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }

        if (corners.size() == 1) {
            // For single marker, still clone the data to avoid memory issues
            List<Mat> clonedCorners = new ArrayList<>();
            clonedCorners.add(corners.get(0).clone());

            Mat clonedIds = new Mat();
            if (ids.rows() > 0) {
                ids.copyTo(clonedIds);
            }

            Log.i(TAG, "Single marker detected, using it.");
            return new Object[]{clonedCorners, clonedIds};
        }

        Log.i(TAG, "Multiple markers detected (" + corners.size() + "), finding closest to center...");

        // Calculate image center
        double imageCenterX = image.cols() / 2.0;
        double imageCenterY = image.rows() / 2.0;

        int closestIndex = 0;
        double minDistance = Double.MAX_VALUE;

        // Find the marker closest to image center
        for (int i = 0; i < corners.size(); i++) {
            Mat corner = corners.get(i);

            // Validate corner data format
            if (corner.rows() != 1 || corner.cols() != 4 || corner.channels() != 2) {
                Log.w(TAG, String.format("Invalid corner format for marker %d: %dx%d channels=%d",
                        i, corner.rows(), corner.cols(), corner.channels()));
                continue;
            }

            // Extract the 4 corner points safely
            float[] cornerData = new float[8]; // 4 points * 2 coordinates
            corner.get(0, 0, cornerData);

            // Calculate marker center (average of 4 corners)
            double markerCenterX = 0;
            double markerCenterY = 0;

            for (int j = 0; j < 4; j++) {
                markerCenterX += cornerData[j * 2];     // x coordinates
                markerCenterY += cornerData[j * 2 + 1]; // y coordinates
            }

            markerCenterX /= 4.0;
            markerCenterY /= 4.0;

            // Calculate distance to image center
            double distance = Math.sqrt(
                    Math.pow(markerCenterX - imageCenterX, 2) +
                            Math.pow(markerCenterY - imageCenterY, 2)
            );

            Log.i(TAG, String.format("Marker %d center: (%.1f, %.1f), distance: %.1f",
                    i, markerCenterX, markerCenterY, distance));

            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        }

        Log.i(TAG, "Closest marker: index " + closestIndex + ", distance: " + minDistance);

        // Create filtered results with properly cloned data
        List<Mat> filteredCorners = new ArrayList<>();
        Mat selectedCorner = corners.get(closestIndex);

        // Ensure the corner data is in the correct format and clone it
        if (selectedCorner.rows() == 1 && selectedCorner.cols() == 4 && selectedCorner.channels() == 2) {
            Mat clonedCorner = selectedCorner.clone();
            filteredCorners.add(clonedCorner);
        } else {
            Log.e(TAG, String.format("Selected corner has invalid format: %dx%d channels=%d",
                    selectedCorner.rows(), selectedCorner.cols(), selectedCorner.channels()));
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }

        // Also filter the IDs to match
        Mat filteredIds = new Mat();
        if (ids.rows() > closestIndex) {
            // Create a 1x1 matrix with the selected ID
            int[] idData = new int[1];
            ids.get(closestIndex, 0, idData);
            filteredIds = new Mat(1, 1, CvType.CV_32S);
            filteredIds.put(0, 0, idData);
        }

        return new Object[]{filteredCorners, filteredIds};
    }
    */

    /**
     * Verifies that ArUco markers are visible by taking pictures at regular intervals
     * @param maxWaitTimeMs Maximum time to wait (e.g., 2000)
     * @param intervalMs Interval between attempts (e.g., 200)
     * @param debugPrefix Prefix for saved debug images (e.g., "astronaut")
     * @return true if markers detected, false if timeout
     */
    private boolean waitForMarkersDetection(int maxWaitTimeMs, int intervalMs, String debugPrefix) {
        boolean markersDetected = false;
        int maxAttempts = maxWaitTimeMs / intervalMs;
        int attempts = 0;
        long startTime = System.currentTimeMillis();

        Log.i(TAG, String.format("Starting marker detection verification - max %dms, interval %dms",
                maxWaitTimeMs, intervalMs));

        while (!markersDetected && attempts < maxAttempts) {
            try {
                // Take a picture
                Mat testImage = api.getMatNavCam();

                if (testImage != null) {
                    // Initialize ArUco detection
                    Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                    List<Mat> corners = new ArrayList<>();
                    Mat ids = new Mat();

                    // Detect markers
                    Aruco.detectMarkers(testImage, dictionary, corners, ids);

                    if (corners.size() > 0) {
                        markersDetected = true;
                        long elapsedTime = System.currentTimeMillis() - startTime;
                        Log.i(TAG, String.format("SUCCESS: %d markers detected after %d attempts (%.1fs)",
                                corners.size(), attempts + 1, elapsedTime / 1000.0));

                        // Save successful image for debugging
                        api.saveMatImage(testImage, debugPrefix + "_markers_detected.png");
                    } else {
                        Log.d(TAG, String.format("Attempt %d/%d: No markers detected", attempts + 1, maxAttempts));
                    }

                    // Clean up ArUco detection resources
                    for (Mat corner : corners) {
                        corner.release();
                    }
                    ids.release();

                    // Clean up test image
                    testImage.release();
                } else {
                    Log.w(TAG, "Failed to get image from camera on attempt " + (attempts + 1));
                }

                attempts++;

                // Wait before next attempt (only if not the last attempt)
                if (!markersDetected && attempts < maxAttempts) {
                    Thread.sleep(intervalMs);
                }

            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted during marker detection");
                break;
            } catch (Exception e) {
                Log.e(TAG, "Error during marker detection attempt " + (attempts + 1) + ": " + e.getMessage());
                attempts++;

                // Still wait before next attempt
                if (attempts < maxAttempts) {
                    try {
                        Thread.sleep(intervalMs);
                    } catch (InterruptedException ie) {
                        Log.w(TAG, "Sleep interrupted after error");
                        break;
                    }
                }
            }
        }

        // Log final result
        long totalTime = System.currentTimeMillis() - startTime;
        if (markersDetected) {
            Log.i(TAG, String.format("%s position verified - markers visible", debugPrefix));
            return true;
        } else {
            Log.w(TAG, String.format("WARNING: No markers detected at %s after %d attempts (%.1fs)",
                    debugPrefix, attempts, totalTime / 1000.0));
            return false;
        }
    }





    /**
     * Returns a list of landmark types from ALL_LANDMARK_TYPES that are not present in the provided set.
     * @param reportedLandmarks A set of landmark type strings that have already been reported.
     * @return A list of unreported landmark type strings.
     */
    private List<String> getUnreportedLandmarkTypes(Set<String> reportedLandmarks) {
        List<String> unreported = new ArrayList<>();
        for (String landmarkType : ALL_LANDMARK_TYPES) {
            if (!reportedLandmarks.contains(landmarkType)) {
                unreported.add(landmarkType);
            }
        }
        return unreported;
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }



}

