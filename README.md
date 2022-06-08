## Error Pattern Recognition and Learning Assistance System for Java Codes
### 系統原始檔說明
| 檔案(資料夾)名稱 | 說明 |
| :----- | :----- |
[student_codes](https://drive.google.com/drive/folders/1cPBZPclzEp8n9TXH3qr6CjnJWdXYqmx0?usp=sharing)(資料夾) | 於 student_codes 資料夾內，有全部學生樣本程式碼，分別再以1-901(資料夾)代表一個樣本程式碼，1-901 代表每一個程式的編號
程式錯誤總頻.xlsx | 判斷所有樣本程式碼所屬於哪一項錯誤，有該項錯誤標示為 1，無該項錯誤標示為 0。
sourcecodes.txt | 將所有樣本程式碼寫入單個文字檔方便類神經模型做讀取，並將樣本程式碼的總樣本數做擴增。
labels.txt | 將程式錯誤總頻.xlsx 轉換為文字檔方便類神經模型讀取。

### 系統執行檔說明:
| 檔案名稱 | 說明 |
| :----- | :----- |
73test.txt、73train.txt | 將所有樣本程式碼分為 7 成訓練 3 成測試兩個檔案。
91test.txt、91train.txt | 將所有樣本程式碼分為 9 成訓練 1 成測試兩個檔案。
trainCNN_7 成 3 成_有權重.py | 該檔案是讀入 73test.txt 與 73train.txt，並加入權重用 CNN 訓練模型去跑出程式碼訓練出的數據。
trainCNN_9 成 1 成_有權重.py | 該檔案是讀入 91test.txt 與 91train.txt，並加入權重用 CNN 訓練模型去跑出程式碼訓練出的數據。
trainCNN_7 成 3 成_沒有權重.py | 該檔案是讀入 73test.txt 與 73train.txt，用 CNN 訓練模型去跑出程式碼訓練出的數據。
trainCNN_9 成 1 成_沒有權重.py | 該檔案是讀入 91test.txt 與 91train.txt，用 CNN 訓練模型去跑出程式碼訓練出的數據。
trainLSTM_CloudVer_訓練測試分開 讀取.py | 該檔案是分次讀入 73test.txt、 73train.txt 與 91test.txt 、 91train.txt 兩種比例，用 LSTM 訓練模型去跑出程式碼訓練出的數據。
last_model | 為以上程式執行出數據做比較後，最終系統使用模型。
