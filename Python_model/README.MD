所有的 Model Weight 都是採用 320*320 進行訓練，使用者請注意輸入尺寸

如果沒有 Python 環境，請至 <https://www.anaconda.com/download> 填寫信箱，點擊寄到信件中的 Download 下載安裝檔。

環境需求（如果執行程式碼時沒有 import 報錯則不需要）：

在 Anaconda 的 Environment 中創建新環境，建議安裝 Python 3.9 版本。

1. 先開啟 Anaconda Prompt，在環境中安裝：

    ```
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

2. 使用 `cd` 指令切換至 `\TWKibo6thShare\Python_model` 目錄：

    ```
    cd \TWKibo6thShare\Python_model
    ```

3. 安裝所需套件：

    ```
    pip install -r requirements.txt
    pip install ultralytics
    ```

即完成環境安裝。完成後可在 Anaconda Navigator 的 Home 中開啟 VS Code。