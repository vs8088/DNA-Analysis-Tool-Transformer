# DNA Analysis Tool
### 🧬 AI-Powered DNA Classification & Repair Tool | AI 驅動的 DNA 分類與修復工具
A powerful tool utilizing Transformer model to analyze DNA sequences. It supports classifying text data as DNA and repairing sequences with missing bases ('N'). Available as a local Python desktop app and a Google Colab notebook.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

**English**: A powerful tool utilizing Transformer model to analyze DNA sequences. It supports classifying text data as DNA and repairing sequences with missing bases ('N'). Available as a local Python desktop app and a Google Colab notebook.

**中文**: 一款利用 Transformer 模型來分析 DNA 序列的強大工具。支持判斷文本數據是否為有效 DNA，以及修復缺失鹼基（'N'）的序列。提供本地 Python 桌面版和 Google Colab 雲端版。

---

## ✨ Features (功能特點)

| Feature | Description (English) | 說明 (中文) |
| :--- | :--- | :--- |
| **🧬 Classification** | Uses Perplexity (PPL) scores to determine if a sequence is valid DNA. | 利用困惑度 (PPL) 分數判斷輸入序列是否為有效的生物 DNA。 |
| **🔧 Deep Repair** | Detects 'N' gaps and predicts the most likely missing nucleotides using MLM inference. | 自動檢測序列中的 'N' 缺口，並利用 MLM 推論預測最可能的缺失鹼基。 |
| **⚡ Smart Chunking** | Automatically handles long sequences by splitting them into overlapping chunks. | 採用重疊分塊策略，自動處理超過模型長度限制的長序列。 |
| **💾 Low Memory Mode** | Supports FP16 (Half Precision) to run the 2.5B model on consumer GPUs (min 6GB VRAM). | 支持 FP16 半精度模式，讓 2.5B 大模型能在消費級顯卡（需 6GB+ VRAM）上運行。 |

---

## 🚀 Option 1: Run Locally (Desktop App)
### 選項一：本地運行 (桌面應用程式)

Suitable for users with a dedicated NVIDIA GPU. It provides a graphical user interface (GUI).
適合擁有 NVIDIA 顯卡的用戶，提供完整的圖形化介面。

### Prerequisites (前置需求)
*   Python 3.8+
*   NVIDIA GPU (Recommended) with CUDA installed.
*   RAM: 16GB+ (System RAM). Depands on model weight.

### Installation (安裝步驟)

1.  **Clone the repository (克隆項目)**:
    ```
    https://github.com/vs8088/DNA-Analysis-Tool-Transformer.git
    cd DNA-Analysis-Tool-Transformer
    ```

2.  **Install Dependencies (安裝依賴)**:
    ```
    pip install -q -r requirements.txt
    ```
    *(Note: For GPU support, ensure you install the CUDA version of PyTorch. Visit [pytorch.org](https://pytorch.org/) for the correct command.)*
    *(注意：如需 GPU 加速，請確保安裝了 CUDA 版本的 PyTorch，請參考 [pytorch.org](https://pytorch.org/) 的安裝指令)*

## .env Settings (.env 環境變數說明)

If you want to use cached/local models (avoid network timeouts), copy `.env.example` to `.env` and adjust the values.  
若要使用本機快取模型（避免下載逾時），請將 `.env.example` 複製為 `.env` 並修改以下設定。

| Variable | Description (EN) | 說明 (中文) | Example |
| --- | --- | --- | --- |
| `NT_MODEL_PATH` | Path to your downloaded model folder. Points the app to local weights instead of downloading. | 已下載模型資料夾路徑，用本機權重避免重新下載。 | `C:\Users\%USERNAME%\DNA-Analysis-Tool-Transformer\models\%model_name%` |
| `NT_LOCAL_ONLY` | `1/true/yes` to force offline/local-files-only mode. | 設為 `1/true/yes` 強制離線模式只讀本機檔案。 | `1` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Enable faster segmented downloads from Hugging Face. No effect if `NT_LOCAL_ONLY=1`. | 啟用 Hugging Face 分段加速下載（離線模式不受影響）。 | `1` |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Seconds to wait for hub responses when downloading. | 下載時的逾時秒數。 | `1800` |

Steps (簡要步驟):
1. Copy `.env.example` to `.env`. / 複製 `.env.example` 為 `.env`。
2. Set `NT_MODEL_PATH` to your cache folder; set `NT_LOCAL_ONLY=1` if you want offline-only. / 設定 `NT_MODEL_PATH` 為模型快取路徑，若要離線請設 `NT_LOCAL_ONLY=1`。
3. Launch the app; in UI, you can still toggle “Use local files only.” / 啟動程式後，可在介面勾選「Use local files only」。

Desktop app: select the model from the dropdown (or set `NT_MODEL_PATH` to a local copy).  
Colab: choose the model in the “Model” dropdown before initialization.

### Usage (使用方法)

1.  Run the script:
    ```
    python dna_analysis.py
    ```
2.  **Select CSV**: Choose your data file.
3.  **Configure**:
    *   **Column Name**: Enter the header name of the DNA column (e.g., `sequence_text`).
    *   **Mode**: Choose `Classification` or `Deep Repair`.
    *   **FP16**: Check this to save memory (Recommended).
4.  **Initialize Model**: Click to download (approx. 5GB) and load the model.
5.  **Start**: Click process and wait for the output CSV.

---

## ☁️ Option 2: Run on Google Colab (Cloud)
### 選項二：在 Google Colab 上運行 (雲端)

Suitable for users without a powerful local GPU. Runs entirely in the browser.
適合沒有高階顯卡的用戶，完全在瀏覽器中運行。

### Instructions (操作指南)

1.  Open [Google Colab](https://colab.research.google.com/).
    打開 [Google Colab](https://colab.research.google.com/)。
2.  **Create a New Notebook** and copy the code from `dna_colab.py` (or upload the notebook file).
    **新建筆記本** 並複製 `dna_colab.py` 中的代碼（或直接上傳 `.ipynb` 文件）。
3.  **Enable GPU Runtime (啟用 GPU)**:
    *   Go to menu: `Runtime` > `Change runtime type`.
    *   Select **T4 GPU** (Essential!).
    *   點擊選單：`執行階段` > `變更執行階段類型` > 選擇 **T4 GPU**（非常重要！）。
4.  **Run the Cell**: Click the Play button.
    **執行代碼格**：點擊播放按鈕。
5.  **Use the UI**: An interactive upload widget will appear. Upload your CSV and click "Start Analysis".
    **使用介面**：下方會出現互動式元件，上傳 CSV 並點擊 "Start Analysis" 即可，完成後會自動下載結果。

---

## 📊 Data Format (數據格式)

Input CSV file should look like this:
輸入的 CSV 文件應如下所示：

| id | sequence_text | description (Optional)|
| :--- | :--- | :--- |
| 1 | ATCGGCTAACGG | Valid DNA |
| 2 | ATCGNCTAACNN | DNA with gaps |
| 3 | RandomTextHere | Invalid Data |

**Output (輸出):**
*   **Classify Mode**: Adds `classification` (Likely Human/Likely Contamination/Non-Human DNA) and `perplexity` score.
*   **Repair Mode**: Adds `repaired_sequence` and `status`.

---

## 🧪 Test Data Description (測試數據說明)

The generated `test_dna_data.csv` contains various scenarios to evaluate the model's robustness in classification and repair tasks.
生成的 `test_dna_data.csv` 包含多種場景，用於評估模型在分類和修復任務中的穩健性。

| ID | Type (類型) | Description (描述) | Expected Result [Classify Mode] <br> (預期結果 [分類模式]) | Expected Result [Repair Mode] <br> (預期結果 [修復模式]) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Normal DNA** <br> (正常 DNA) | Standard valid DNA sequence (60bp). <br> 標準的有效 DNA 序列 (60bp)。 | **Likely Human DNA** <br> (Low Perplexity) | **No Change** <br> (原樣輸出) |
| **2** | **Long DNA** <br> (長序列 DNA) | Sequence > 1000bp to test chunking logic. <br> 長度超過 1000bp，用於測試分塊邏輯。 | **Likely Human DNA** <br> (Should handle chunks correctly) | **No Change** <br> (原樣輸出) |
| **3** | **Gapped DNA** <br> (缺失 DNA) | DNA sequence with random 'N' gaps. <br> 包含隨機 'N' 缺失的 DNA 序列。 | **Likely Human DNA** <br> (Model tolerates small gaps) | **Repaired** <br> ('N' replaced with A/T/C/G) |
| **4** | **Long Gapped DNA** <br> (長序列缺失 DNA) | Long sequence (>1000bp) with 'N' gaps. <br> 包含 'N' 缺失的長序列 (>1000bp)。 | **Likely Human DNA** | **Repaired** <br> (Predicts across chunks) |
| **5** | **English Text** <br> (英文文本) | Plain English sentence ("This is not..."). <br> 普通英文句子 ("This is not...")。 | **Likely Contamination/Non-Human** <br> (High Perplexity) | **No Change** <br> (No 'N' found) |
| **6** | **Random Noise** <br> (隨機噪聲) | Alphanumeric string ("RandomString..."). <br> 包含數字和字母的混合字符串。 | **Likely Contamination/Non-Human** | **No Change** |
| **7** | **Repetitive Pattern** <br> (重複模式) | Artificial repeats ("ATCGATCG..."). <br> 人工合成的重複序列 ("ATCGATCG...")。 | **Likely Contamination/Non-Human** <br> (Structurally valid but simple) | **No Change** |
| **8** | **Lowercase DNA** <br> (小寫 DNA) | Valid sequence in lowercase ("atcg..."). <br> 小寫的有效序列 ("atcg...")。 | **Likely Contamination/Non-Human** <br> (Tokenizer handles normalization) | **No Change** <br> (Output may be uppercased) |
| **9** | **Empty** <br> (空值) | Empty string or NaN. <br> 空字符串或 NaN。 | **Error / Empty** <br> (Skipped by logic) | **Empty** |
| **10** | **All Ns** <br> (全 N 序列) | Sequence consisting entirely of 'N'. <br> 完全由 'N' 組成的序列。 | **Ambiguous / Likely Contamination/Non-Human** | **Repaired / Error** <br> (May hallucinate or fail) |

## ⚠️ Notes (注意事項)

*   **Model Size**: The first run will download ~5GB of demo model weights. Please ensure a stable internet connection.
    **模型大小**：首次運行會下載約 5GB 的模型權重，請確保網絡穩定。
*   **Accuracy**: Repair predictions are probabilistic based on the model's training data (genomic data from 3000+ species). It does not guarantee biological correctness without experimental verification.
    **準確度**：修復預測是基於示範模型訓練數據（3000+ 物種基因組）的概率推斷，未經實驗驗證不能保證生物學上的絕對正確。

## Demo Model Options (示範模型選項)

| Option | Description (EN) | 說明 (中文) |
| --- | --- | --- |
| **Nucleotide Transformer 2.5B**<br>`InstaDeepAI/nucleotide-transformer-2.5b-1000g` | Highest accuracy; larger memory/VRAM footprint. Best when you have a strong GPU and want the most robust classification/repair. | 準確度最高，但需較多記憶體/VRAM。適合有較強 GPU、追求最佳分類與修復效果的情境。 |
| **Nucleotide Transformer 500M**<br>`InstaDeepAI/nucleotide-transformer-500m-human-ref ` | Smaller and faster to download/load; good for Colab or lower-VRAM setups. Slightly lower accuracy than 2.5B but much lighter. | 體積較小、下載/載入較快，適合 Colab 或較低 VRAM 的環境；準確度略低於 2.5B，但資源需求大幅降低。 |

## Demo Model URL 
[InstaDeepAI/nucleotide-transformer-2.5b-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)

[InstaDeepAI/nucleotide-transformer-2.5b-1000g](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-1000g)

[InstaDeepAI/nucleotide-transformer-500m-human-ref](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref)


## 📜 License

This code is licensed under the MIT License - see the [LICENSE] file for details.
