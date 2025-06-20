package jp.jaxa.iss.kibo.rpc.sampleapk;

import ai.onnxruntime.*;
import android.content.Context;
import android.util.Log;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * 增强型YOLO对象检测服务，具有智能非极大值抑制功能
 * 匹配Python的yoloraw_postprocessing.py功能
 */
public class YOLODetectionService {
    private static final String TAG = "YOLODetectionService";
    private static final String MODEL_NAME = "yolo_v8n_400.onnx";
    private static final int INPUT_SIZE = 320;
    private static final float DEFAULT_CONF_THRESHOLD = 0.3f;
    private static final float DEFAULT_STANDARD_NMS_THRESHOLD = 0.45f;
    private static final float DEFAULT_OVERLAP_NMS_THRESHOLD = 0.8f;

    // 与Python代码匹配的类定义
    private static final String[] CLASS_NAMES = {
            "coin", "compass", "coral", "crystal", "diamond", "emerald",
            "fossil", "key", "letter", "shell", "treasure_box"
    };

    private static final Set<Integer> TREASURE_IDS = new HashSet<>(Arrays.asList(3, 4, 5)); // 水晶，钻石，翡翠
    private static final Set<Integer> LANDMARK_IDS = new HashSet<>(Arrays.asList(0, 1, 2, 6, 7, 8, 9, 10)); // 硬币，指南针，珊瑚，化石，钥匙，信件，贝壳，宝箱

    private OrtEnvironment env;
    private OrtSession session;
    private Context context;
    private boolean isInitialized = false;

    public YOLODetectionService(Context context) {
        this.context = context;
        initializeModel();
    }

    /**
     * Check if the ONNX model has been loaded successfully.
     */
    public boolean isModelReady() {
        return isInitialized;
    }

    private void initializeModel() {
        try {
            Log.i(TAG, "正在初始化YOLO模型...");

            env = OrtEnvironment.getEnvironment();
            File modelFile = copyAssetToFile(MODEL_NAME);

            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

            session = env.createSession(modelFile.getAbsolutePath(), sessionOptions);
            isInitialized = true;
            Log.i(TAG, "YOLO模型初始化成功");

        } catch (Exception e) {
            Log.e(TAG, "初始化YOLO模型失败: " + e.getMessage(), e);
            isInitialized = false;
        }
    }

    private File copyAssetToFile(String assetName) throws IOException {
        InputStream inputStream = context.getAssets().open(assetName);
        File outputFile = new File(context.getFilesDir(), assetName);

        FileOutputStream outputStream = new FileOutputStream(outputFile);
        byte[] buffer = new byte[4096];
        int length;
        while ((length = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, length);
        }

        outputStream.close();
        inputStream.close();

        return outputFile;
    }

    /**
     * 主检测方法，匹配Python的simple_detection_example功能
     * @param image OpenCV Mat图像
     * @param imageType "lost"或"target"
     * @param confThreshold 置信度阈值（默认：0.3）
     * @param standardNmsThreshold 标准NMS阈值（默认：0.45）
     * @param overlapNmsThreshold 智能NMS的重叠阈值（默认：0.8）
     * @return 包含宝藏和地标数量的增强检测结果
     */
    public EnhancedDetectionResult DetectfromcvImage(Mat image, String imageType,
                                                     float confThreshold,
                                                     float standardNmsThreshold,
                                                     float overlapNmsThreshold) {
        if (!isInitialized) {
            Log.e(TAG, "YOLO模型未初始化");
            return new EnhancedDetectionResult();
        }

        try {
            Log.i(TAG, "开始对图像类型进行检测: " + imageType);

            // 预处理图像
            Mat preprocessedImage = preprocessImage(image);
            float[][][][] inputData = matToFloatArray(preprocessedImage);

            // 运行推理获取原始张量
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
            inputMap.put("images", inputTensor);

            OrtSession.Result result = session.run(inputMap);
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            float[][][] rawOutput = (float[][][]) outputTensor.getValue();

            // 应用智能后处理管道
            EnhancedDetectionResult detectionResult = yoloPostprocessPipeline(
                    rawOutput, confThreshold, standardNmsThreshold, overlapNmsThreshold,
                    INPUT_SIZE, imageType, image.width(), image.height()
            );

            // 清理资源
            inputTensor.close();
            result.close();
            preprocessedImage.release();

            Log.i(TAG, String.format("完成%s图像的检测", imageType));
            detectionResult.logResults(TAG);

            return detectionResult;

        } catch (Exception e) {
            Log.e(TAG, "检测失败: " + e.getMessage(), e);
            return new EnhancedDetectionResult();
        }
    }

    /**
     * 使用默认参数的便捷方法
     */
    public EnhancedDetectionResult DetectfromcvImage(Mat image, String imageType) {
        return DetectfromcvImage(image, imageType, DEFAULT_CONF_THRESHOLD,
                DEFAULT_STANDARD_NMS_THRESHOLD, DEFAULT_OVERLAP_NMS_THRESHOLD);
    }

    /**
     * 兼容现有代码的方法，返回简单类计数
     * @param image OpenCV Mat图像
     * @return 使用"lost"检测逻辑的类ID到计数的映射
     */
    public Map<Integer, Integer> getItemCounts(Mat image) {
        EnhancedDetectionResult result = DetectfromcvImage(image, "lost");
        return result.getAllQuantities();
    }

    /**
     * 获取类名数组供外部使用
     * @return 类名数组
     */
    public static String[] getClassNames() {
        return CLASS_NAMES.clone();
    }

    /**
     * 通过ID获取类名
     * @param classId 类ID（从0开始）
     * @return 类名，如果ID无效则返回null
     */
    public static String getClassName(int classId) {
        if (classId >= 0 && classId < CLASS_NAMES.length) {
            return CLASS_NAMES[classId];
        }
        return null;
    }

    /**
     * 增强型后处理管道，匹配Python逻辑
     */
    private EnhancedDetectionResult yoloPostprocessPipeline(float[][][] rawTensor,
                                                            float confThreshold,
                                                            float standardNmsThreshold,
                                                            float overlapNmsThreshold,
                                                            int imgSize,
                                                            String imgType,
                                                            int originalWidth,
                                                            int originalHeight) {
        Log.i(TAG, String.format("原始张量形状: [%d, %d, %d]",
                rawTensor.length, rawTensor[0].length, rawTensor[0][0].length));

        // ====================================================================
        // 关键修复: 将张量从[1, 15, 2100]转置为[1, 2100, 15]
        // 这与Python相匹配: processed_tensor = raw_tensor.transpose(1, 2)
        // ====================================================================

        float[][] processed;
        int numDetections, numFeatures;

        // 检查是否需要转置（匹配Python逻辑）
        if (rawTensor[0].length < rawTensor[0][0].length) {
            // 需要从[15, 2100]转置为[2100, 15]
            Log.i(TAG, "将张量从[15, 2100]转置为[2100, 15]");

            numDetections = rawTensor[0][0].length;  // 2100
            numFeatures = rawTensor[0].length;       // 15

            processed = new float[numDetections][numFeatures];

            // 转置: processed[detection][feature] = rawTensor[0][feature][detection]
            for (int det = 0; det < numDetections; det++) {
                for (int feat = 0; feat < numFeatures; feat++) {
                    processed[det][feat] = rawTensor[0][feat][det];
                }
            }

        } else {
            // 已经是正确格式[2100, 15]
            Log.i(TAG, "张量已经是正确格式");
            processed = rawTensor[0];
            numDetections = processed.length;
            numFeatures = processed[0].length;
        }

        Log.i(TAG, String.format("处理%d个检测提案，每个有%d个特征",
                numDetections, numFeatures));

        // 记录每个特征在所有检测中的最小/最大值（匹配Python的第0-14层）
        Log.i(TAG, "所有检测中特征的最小/最大值:");
        for (int featIdx = 0; featIdx < numFeatures; featIdx++) {
            float minValue = Float.MAX_VALUE;
            float maxValue = Float.MIN_VALUE;

            // 在所有检测中找出这个特征的最小/最大值
            for (int detIdx = 0; detIdx < numDetections; detIdx++) {
                float value = processed[detIdx][featIdx];
                minValue = Math.min(minValue, value);
                maxValue = Math.max(maxValue, value);
            }

            // 这现在应该匹配Python的"Layer X: min=..., max=..."
            Log.i(TAG, String.format("层 %d: 最小值=%.6f, 最大值=%.6f",
                    featIdx, minValue, maxValue));
        }

        List<DetectionCandidate> candidates = new ArrayList<>();

        // 步骤1：提取所有高于置信度阈值的检测候选者
        for (int i = 0; i < processed.length; i++) {
            float[] prediction = processed[i];

            if (prediction.length < 5) continue;

            // 提取边界框和类别分数
            float centerX = prediction[0];
            float centerY = prediction[1];
            float width = prediction[2];
            float height = prediction[3];

            // 检查所有类别分数
            for (int classId = 0; classId < CLASS_NAMES.length; classId++) {
                float classScore = prediction[4 + classId];

                if (classScore > confThreshold) {
                    // 将坐标缩放回原始图像大小
                    float scaleX = (float) originalWidth / imgSize;
                    float scaleY = (float) originalHeight / imgSize;

                    float scaledCenterX = centerX * scaleX;
                    float scaledCenterY = centerY * scaleY;
                    float scaledWidth = width * scaleX;
                    float scaledHeight = height * scaleY;

                    candidates.add(new DetectionCandidate(
                            scaledCenterX, scaledCenterY, scaledWidth, scaledHeight,
                            classScore, classId
                    ));
                }
            }
        }

        Log.i(TAG, String.format("总检测候选者: %d", candidates.size()));

        // 步骤2：分离宝藏和地标候选者
        List<DetectionCandidate> treasureCandidates = new ArrayList<>();
        List<DetectionCandidate> landmarkCandidates = new ArrayList<>();

        for (DetectionCandidate candidate : candidates) {
            if (TREASURE_IDS.contains(candidate.classId)) {
                treasureCandidates.add(candidate);
            } else if (LANDMARK_IDS.contains(candidate.classId)) {
                landmarkCandidates.add(candidate);
            }
        }

        Log.i(TAG, String.format("宝藏候选者: %d, 地标候选者: %d",
                treasureCandidates.size(), landmarkCandidates.size()));

        // 步骤3：应用图像类型约束与智能NMS
        return applyImageTypeConstraints(treasureCandidates, landmarkCandidates,
                imgType, standardNmsThreshold, overlapNmsThreshold);
    }

    private EnhancedDetectionResult applyImageTypeConstraints(List<DetectionCandidate> treasureCandidates,
                                                              List<DetectionCandidate> landmarkCandidates,
                                                              String imgType,
                                                              float standardNmsThreshold,
                                                              float overlapNmsThreshold) {
        List<FinalDetection> finalDetections = new ArrayList<>();
        Map<Integer, Integer> treasureQuantities = new HashMap<>();
        Map<Integer, Integer> landmarkQuantities = new HashMap<>();
        Map<Integer, Integer> allQuantities = new HashMap<>();

        if ("target".equals(imgType)) {
            Log.i(TAG, "目标物品逻辑 - 应用标准NMS");

            // 对宝藏和地标应用标准NMS
            List<FinalDetection> treasureFinal = applyStandardNMS(treasureCandidates, standardNmsThreshold);
            List<FinalDetection> landmarkFinal = applyStandardNMS(landmarkCandidates, standardNmsThreshold);

            // NMS后计算数量
            countQuantities(treasureFinal, treasureQuantities, allQuantities);
            countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

            // 按置信度排序
            treasureFinal.sort((a, b) -> Float.compare(b.confidence, a.confidence));
            landmarkFinal.sort((a, b) -> Float.compare(b.confidence, a.confidence));

            // 选择恰好1个宝藏 + 2个不同地标类型
            if (!treasureFinal.isEmpty() && landmarkFinal.size() >= 2) {
                finalDetections.add(treasureFinal.get(0));
                Log.i(TAG, String.format("选择的宝藏: %s (置信度: %.3f)",
                        CLASS_NAMES[treasureFinal.get(0).classId], treasureFinal.get(0).confidence));

                Set<Integer> selectedLandmarkClasses = new HashSet<>();
                for (FinalDetection landmark : landmarkFinal) {
                    if (!selectedLandmarkClasses.contains(landmark.classId)) {
                        finalDetections.add(landmark);
                        selectedLandmarkClasses.add(landmark.classId);
                        Log.i(TAG, String.format("选择的地标: %s (置信度: %.3f)",
                                CLASS_NAMES[landmark.classId], landmark.confidence));

                        if (selectedLandmarkClasses.size() == 2) break;
                    }
                }
            }

        } else if ("lost".equals(imgType)) {
            Log.i(TAG, "失物逻辑 - 应用智能NMS");

            if (!treasureCandidates.isEmpty()) {
                // 情况1: 1个地标 + 1个宝藏
                Log.i(TAG, "情况1: 检测到宝藏 + 地标");

                List<FinalDetection> treasureFinal = applyStandardNMS(treasureCandidates, standardNmsThreshold);
                List<FinalDetection> landmarkFinal = applyLandmarkIntelligentNMS(landmarkCandidates, overlapNmsThreshold);

                countQuantities(treasureFinal, treasureQuantities, allQuantities);
                countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

                treasureFinal.sort((a, b) -> Float.compare(b.confidence, a.confidence));
                landmarkFinal.sort((a, b) -> Float.compare(b.confidence, a.confidence));

                if (!treasureFinal.isEmpty()) {
                    finalDetections.add(treasureFinal.get(0));
                    Log.i(TAG, String.format("选择的宝藏: %s (置信度: %.3f)",
                            CLASS_NAMES[treasureFinal.get(0).classId], treasureFinal.get(0).confidence));
                }

                if (!landmarkFinal.isEmpty()) {
                    finalDetections.add(landmarkFinal.get(0));
                    Log.i(TAG, String.format("选择的地标: %s (置信度: %.3f)",
                            CLASS_NAMES[landmarkFinal.get(0).classId], landmarkFinal.get(0).confidence));
                }

            } else {
                // 情况2: 只有地标
                Log.i(TAG, "情况2: 只检测到地标");

                List<FinalDetection> landmarkFinal = applyLandmarkIntelligentNMS(landmarkCandidates, overlapNmsThreshold);
                countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

                landmarkFinal.sort((a, b) -> Float.compare(b.confidence, a.confidence));

                if (!landmarkFinal.isEmpty()) {
                    finalDetections.add(landmarkFinal.get(0));
                    Log.i(TAG, String.format("选择的地标: %s (置信度: %.3f)",
                            CLASS_NAMES[landmarkFinal.get(0).classId], landmarkFinal.get(0).confidence));
                }
            }
        }

        return new EnhancedDetectionResult(finalDetections, allQuantities, treasureQuantities, landmarkQuantities);
    }

    private List<FinalDetection> applyStandardNMS(List<DetectionCandidate> candidates, float nmsThreshold) {
        if (candidates.size() <= 1) {
            return convertToFinalDetections(candidates);
        }

        // 按置信度排序
        candidates.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        List<DetectionCandidate> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[candidates.size()];

        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;

            DetectionCandidate current = candidates.get(i);
            kept.add(current);

            for (int j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j]) continue;

                DetectionCandidate other = candidates.get(j);
                if (calculateIoU(current, other) > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return convertToFinalDetections(kept);
    }

    private List<FinalDetection> applyLandmarkIntelligentNMS(List<DetectionCandidate> candidates, float overlapThreshold) {
        if (candidates.size() <= 1) {
            return convertToFinalDetections(candidates);
        }

        Log.i(TAG, String.format("对%d个地标检测应用智能NMS", candidates.size()));

        // 找到最高置信度的检测及其类别
        DetectionCandidate highest = candidates.stream()
                .max(Comparator.comparingDouble(c -> c.confidence))
                .orElse(null);

        if (highest == null) return new ArrayList<>();

        int selectedClass = highest.classId;
        Log.i(TAG, String.format("选择的类别: %d (%s) 置信度: %.3f",
                selectedClass, CLASS_NAMES[selectedClass], highest.confidence));

        // 仅过滤出所选类别的检测
        List<DetectionCandidate> sameClassCandidates = new ArrayList<>();
        for (DetectionCandidate candidate : candidates) {
            if (candidate.classId == selectedClass) {
                sameClassCandidates.add(candidate);
            }
        }

        Log.i(TAG, String.format("选择类别的检测数: %d/%d",
                sameClassCandidates.size(), candidates.size()));

        // 使用重叠阈值对相同类别的检测应用标准NMS
        List<FinalDetection> result = applyStandardNMS(sameClassCandidates, overlapThreshold);

        Log.i(TAG, String.format("智能NMS后保留的地标: %d/%d 类别 %s",
                result.size(), sameClassCandidates.size(), CLASS_NAMES[selectedClass]));

        return result;
    }

    private List<FinalDetection> convertToFinalDetections(List<DetectionCandidate> candidates) {
        List<FinalDetection> result = new ArrayList<>();
        for (DetectionCandidate candidate : candidates) {
            result.add(new FinalDetection(
                    candidate.centerX, candidate.centerY, candidate.width, candidate.height,
                    candidate.confidence, candidate.classId
            ));
        }
        return result;
    }

    private void countQuantities(List<FinalDetection> detections,
                                 Map<Integer, Integer> specificQuantities,
                                 Map<Integer, Integer> allQuantities) {
        for (FinalDetection detection : detections) {
            specificQuantities.put(detection.classId,
                    specificQuantities.getOrDefault(detection.classId, 0) + 1);
            allQuantities.put(detection.classId,
                    allQuantities.getOrDefault(detection.classId, 0) + 1);
        }
    }

    private float calculateIoU(DetectionCandidate a, DetectionCandidate b) {
        float x1_a = a.centerX - a.width / 2;
        float y1_a = a.centerY - a.height / 2;
        float x2_a = a.centerX + a.width / 2;
        float y2_a = a.centerY + a.height / 2;

        float x1_b = b.centerX - b.width / 2;
        float y1_b = b.centerY - b.height / 2;
        float x2_b = b.centerX + b.width / 2;
        float y2_b = b.centerY + b.height / 2;

        float intersectionX1 = Math.max(x1_a, x1_b);
        float intersectionY1 = Math.max(y1_a, y1_b);
        float intersectionX2 = Math.min(x2_a, x2_b);
        float intersectionY2 = Math.min(y2_a, y2_b);

        if (intersectionX2 <= intersectionX1 || intersectionY2 <= intersectionY1) {
            return 0.0f;
        }

        float intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float unionArea = areaA + areaB - intersectionArea;

        return intersectionArea / unionArea;
    }

    private Mat preprocessImage(Mat image) {
        Mat processedImage = new Mat();

        // 如果需要，转换为RGB
        if (image.channels() == 1) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_BGRA2RGB);
        } else if (image.channels() == 3) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_BGR2RGB);
        } else {
            image.copyTo(processedImage);
        }

        // 调整大小至模型输入尺寸
        Mat resizedImage = new Mat();
        Imgproc.resize(processedImage, resizedImage, new Size(INPUT_SIZE, INPUT_SIZE));

        processedImage.release();
        return resizedImage;
    }

    private float[][][][] matToFloatArray(Mat image) {
        float[][][][] inputData = new float[1][3][INPUT_SIZE][INPUT_SIZE];

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                double[] pixel = image.get(y, x);

                // 归一化至[0, 1]
                inputData[0][0][y][x] = (float) (pixel[0] / 255.0);
                inputData[0][1][y][x] = (float) (pixel[1] / 255.0);
                inputData[0][2][y][x] = (float) (pixel[2] / 255.0);
            }
        }

        return inputData;
    }

    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "关闭YOLO服务时出错: " + e.getMessage(), e);
        }
    }

    // 辅助类
    public static class DetectionCandidate {
        public final float centerX, centerY, width, height;
        public final float confidence;
        public final int classId;

        public DetectionCandidate(float centerX, float centerY, float width, float height,
                                  float confidence, int classId) {
            this.centerX = centerX;
            this.centerY = centerY;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
            this.classId = classId;
        }
    }

    public static class FinalDetection {
        public final float centerX, centerY, width, height;
        public final float confidence;
        public final int classId;

        public FinalDetection(float centerX, float centerY, float width, float height,
                              float confidence, int classId) {
            this.centerX = centerX;
            this.centerY = centerY;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
            this.classId = classId;
        }

        @Override
        public String toString() {
            return String.format("检测[类别=%s, 置信度=%.2f, 中心=(%.1f,%.1f), 大小=(%.1f,%.1f)]",
                    CLASS_NAMES[classId], confidence, centerX, centerY, width, height);
        }
    }

    public static class EnhancedDetectionResult {
        private List<FinalDetection> detections;
        private Map<Integer, Integer> allQuantities;
        private Map<Integer, Integer> treasureQuantities;
        private Map<Integer, Integer> landmarkQuantities;

        public EnhancedDetectionResult() {
            this.detections = new ArrayList<>();
            this.allQuantities = new HashMap<>();
            this.treasureQuantities = new HashMap<>();
            this.landmarkQuantities = new HashMap<>();
        }

        public EnhancedDetectionResult(List<FinalDetection> detections,
                                       Map<Integer, Integer> allQuantities,
                                       Map<Integer, Integer> treasureQuantities,
                                       Map<Integer, Integer> landmarkQuantities) {
            this.detections = detections;
            this.allQuantities = allQuantities;
            this.treasureQuantities = treasureQuantities;
            this.landmarkQuantities = landmarkQuantities;
        }

        public List<FinalDetection> getDetections() { return detections; }
        public Map<Integer, Integer> getAllQuantities() { return allQuantities; }
        public Map<Integer, Integer> getTreasureQuantities() { return treasureQuantities; }
        public Map<Integer, Integer> getLandmarkQuantities() { return landmarkQuantities; }

        /**
         * 获取Python格式的结果
         * @return 包含以类名为键的数量的映射
         */
        public Map<String, Object> getPythonLikeResult() {
            Map<String, Object> result = new HashMap<>();

            // 将所有数量转换为使用类名
            Map<String, Integer> allQuantitiesNamed = new HashMap<>();
            for (Map.Entry<Integer, Integer> entry : allQuantities.entrySet()) {
                allQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            // 将宝藏数量转换为使用类名
            Map<String, Integer> treasureQuantitiesNamed = new HashMap<>();
            for (Map.Entry<Integer, Integer> entry : treasureQuantities.entrySet()) {
                treasureQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            // 将地标数量转换为使用类名
            Map<String, Integer> landmarkQuantitiesNamed = new HashMap<>();
            for (Map.Entry<Integer, Integer> entry : landmarkQuantities.entrySet()) {
                landmarkQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            result.put("all_quantities", allQuantitiesNamed);
            result.put("treasure_quantities", treasureQuantitiesNamed);
            result.put("landmark_quantities", landmarkQuantitiesNamed);

            return result;
        }

        public void logResults(String tag) {
            Log.i(tag, String.format("总检测数: %d", detections.size()));

            Map<String, Object> pythonResult = getPythonLikeResult();
            Log.i(tag, "所有数量: " + pythonResult.get("all_quantities"));
            Log.i(tag, "宝藏数量: " + pythonResult.get("treasure_quantities"));
            Log.i(tag, "地标数量: " + pythonResult.get("landmark_quantities"));
        }
    }
}