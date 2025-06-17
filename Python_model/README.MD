如果沒有 Python 環境請至 https://www.anaconda.com/download 填寫信箱
點擊寄到信件中 Download 下載安裝檔 


環境需求(如果執行程式碼不會 import 報錯則不需要)
在 Anaconda 的中 Enviroment 中創建新環境 建議安裝 3.9版本

1. 先開啟 Anaconda Prompt 在環境中安裝
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

2. 使用 cd 指令至 \TWKibo6thShare\Python_model
pip install -r requirements.txt