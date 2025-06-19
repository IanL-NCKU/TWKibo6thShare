因為每個APK檔案實在太大，不能上傳至 Github 所以完整的 APK 檔們採用 Google 雲端的方式存放
但在每個資料夾中都有存放 APK 的壓縮檔 debug-apk.zip 經解壓縮上傳後與原 APK 檔成果一致，大家依然可以自行取用
連結: https://drive.google.com/drive/folders/12J0TEfFL4H_E_MA2l2DCwjLUUhorDsJD?usp=drive_link

For each APK file is too large to upload to GitHub, the complete APK files are stored on Google Drive instead. However, each folder contains compressed APK files saved as debug-apk.zip. After decompression and upload, these are identical to the original APK files, so everyone can still access and use them as needed.

Link: https://drive.google.com/drive/folders/12J0TEfFL4H_E_MA2l2DCwjLUUhorDsJD?usp=drive_link

### Additional Setup Notes

The Android projects expect the YOLO ONNX model and OpenCV native libraries to be packaged in the APK. If you see errors like:

```
E/OpenCV/StaticHelper: OpenCV error: Cannot load info library for OpenCV
E/YOLODetectionService: YOLO model not initialized
```

please confirm the following before building the APK:

1. Place `yolo_v8n_400.onnx` inside `app/src/main/assets/` of the Android module.
2. Include OpenCV via Gradle as shown in `onnxruntimegradle.txt`.
3. Clean and rebuild the project so that the libraries and model are bundled correctly.
