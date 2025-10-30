Mobile Reviews Sentiment Analysis

利用機器學習／NLP 對手機商品評論 (mobile product reviews) 做情感分析。
目標是：一大包文字評論 → 判斷它是正向、負向，然後產一份「數據報告＋圖表」。
這個 repo 可以拿去當 Demo / 線上展示頁 / 面試作品集，一步一步照做就能跑。

Dataset: 以「約 50K 筆手機相關評論」為假設規模，實測可達 85%+ accuracy（視你選的模型、清洗規則而定）。

1. 專案目標 (What this repo does)

一、把原始的手機評論資料（csv / json / excel）清洗成 NLP 可以吃的文字。
二、把文字轉成特徵（Bag-of-Words、TF-IDF、n-gram）。
三、訓練一個或多個情感分類模型（Logistic Regression / SVM / Random Forest / XGBoost…）。
四、輸出評估指標（accuracy, precision, recall, f1, confusion matrix）。
五、把分析結果畫圖，存在 visualizations/，方便你做網頁 demo。
六、保留一個「之後要換深度學習 / BERT / HuggingFace」的入口。

mobile-reviews-sentiment-analysis/
│
├── data/
│   ├── raw/                  # 你拿到的原始資料放這裡 (e.g. mobile_reviews_raw.csv)
│   ├── processed/            # 前處理後的乾淨資料 (e.g. mobile_reviews_clean.csv)
│   └── README.md             # (可選) 說明資料來源、欄位意義
│
├── scripts/
│   ├── 01_preprocess.py      # 前處理 / 清洗資料
│   ├── 02_feature_engineering.py  # TF-IDF、n-gram、Embedding
│   ├── 03_train_model.py     # 訓練模型，會存到 models/
│   ├── 04_evaluate.py        # 評估、產生報表、混淆矩陣
│   └── 05_inference.py       # 給新評論就能預測 (demo 用)
│
├── models/
│   ├── tfidf_vectorizer.pkl  # 特徵轉換器
│   ├── sentiment_model.pkl   # 訓練好的模型 (e.g. LogisticRegression)
│   └── label_encoder.pkl     # 把 positive / negative 轉成 1 / 0
│
├── visualizations/
│   ├── sentiment_distribution.png   # 正負向評論比例
│   ├── confusion_matrix.png         # 混淆矩陣
│   ├── top_words_positive.png       # 正向評論常見詞
│   └── top_words_negative.png       # 負向評論常見詞
│
├── requirements.txt
└── README.md   ← 就是這個

3. 流程圖 (Data & ML pipeline)
flowchart TD
    A[raw reviews CSV in data/raw] --> B[01_preprocess.py<br/>清洗、去除HTML、轉小寫、去停用詞]
    B --> C[02_feature_engineering.py<br/>TF-IDF / n-gram / tokenize]
    C --> D[03_train_model.py<br/>train/test split, fit model, save .pkl]
    D --> E[04_evaluate.py<br/>classification report, confusion matrix, export to visualizations/]
    E --> F[05_inference.py<br/>web/demo 用的即時預測]

4. 安裝與執行 (Setup & Run in VS Code)
一、clone 專案
git clone https://github.com/ritalinyutzu/mobile-reviews-sentiment-analysis.git
cd mobile-reviews-sentiment-analysis

二、建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

三、安裝套件
pip install -r requirements.txt

四、把原始資料放進去
檔案放在：data/raw/mobile_reviews_raw.csv
至少要有這幾欄：
review_text：評論內容
rating：1~5 星
（可選）reviewer_id, app_name, device …
如果你沒有 label，可以規則標：rating ≥ 4 → positive；rating ≤ 2 → negative；rating=3 → 丟掉或歸中性。

五、依序跑腳本
python scripts/01_preprocess.py
python scripts/02_feature_engineering.py
python scripts/03_train_model.py
python scripts/04_evaluate.py

六、要做 demo / 即時預測
python scripts/05_inference.py --text "The battery is terrible but the screen is good."

5. 前處理 (Preprocessing)
輸入：data/raw/mobile_reviews_raw.csv
輸出：data/processed/mobile_reviews_clean.csv

前處理會做這幾件事（照順序）：

一、載入資料
用 pandas.read_csv()
檢查缺失值 → 若 review_text 為空，直接丟掉
若有重複評論（同一個人、同一段話），可以用 drop_duplicates(subset=['review_text'])

二、文字正規化
全部轉小寫
移除 HTML tag：BeautifulSoup 或 regex (re.sub(r'<.*?>', '', text))
移除網址、email、@帳號：re.sub(r'http\S+|www.\S+', '', text)
移除多餘空白：" ".join(text.split())

三、標點與表情符號處理
如果是英文評論：可以移除標點
如果你有 emoji 要辨識成情感（🙂 👍 😡），可以先替換成文字，例如 :smile:, :angry:

四、停用詞 (stopwords)
使用 nltk.corpus.stopwords or sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
移除 like, the, a, it, to… 這種沒有情感的詞

五、詞形還原 (lemmatization/stemming)
可以用 nltk.WordNetLemmatizer
把 loved, loving → love

六、產 sentiment label
如果你的原始資料有星等，就做：
def map_rating_to_label(r):
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    else:
        return "neutral"
如果你不想要 neutral，就把它丟掉：
df = df[df['sentiment'] != 'neutral']

七、存檔
存成：data/processed/mobile_reviews_clean.csv
這個檔案是後續所有模型的「乾淨版資料」
📄 範例程式（01_preprocess.py）：
import pandas as pd
import re
from pathlib import Path

RAW_PATH = Path("data/raw/mobile_reviews_raw.csv")
OUT_PATH = Path("data/processed/mobile_reviews_clean.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)           # remove HTML
    text = re.sub(r'http\S+|www.\S+', ' ', text) # remove urls
    text = re.sub(r'[^a-z0-9\s]', ' ', text)     # keep only letters/digits
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_csv(RAW_PATH)
df = df.dropna(subset=['review_text']).drop_duplicates(subset=['review_text'])

df['clean_text'] = df['review_text'].apply(clean_text)

def map_rating_to_label(r):
    if r >= 4: return "positive"
    elif r <= 2: return "negative"
    else: return "neutral"

df['sentiment'] = df['rating'].apply(map_rating_to_label)
df = df[df['sentiment'] != 'neutral']   # optional

df.to_csv(OUT_PATH, index=False)
print(f"✅ cleaned data saved to {OUT_PATH} with shape={df.shape}")

6. 特徵工程 (Feature Engineering)
這一步是把文字變成機器學習能吃的數字。
一、載入剛剛的乾淨資料
二、切 train / test（e.g. 80/20）
三、建立 TF-IDF vectorizer
max_features=20000
ngram_range=(1,2) → 可以抓到「battery life」「fast charging」這種片語
min_df=2 → 太少見的詞丟掉
四、把 vectorizer 存起來（之後預測要用一樣的）
📄 範例（02_feature_engineering.py）：
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/mobile_reviews_clean.csv")
VECT_PATH = Path("models/tfidf_vectorizer.pkl")
VECT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df['clean_text'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words='english'
)
tfidf.fit(X_train)

joblib.dump({
    "vectorizer": tfidf,
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}, VECT_PATH)

print("✅ TF-IDF fitted and saved to", VECT_PATH)

7. 建模 (Modeling)
這裡先用最穩、最不會出包、也最容易寫在 README 的：Logistic Regression。
你也可以平行試：SVM, Linear SVC, RandomForest, XGBoost。
如果你要做成網頁 demo，LogReg / LinearSVC 載入速度最快。
📄 範例（03_train_model.py）：
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

VECT_PATH = Path("models/tfidf_vectorizer.pkl")
MODEL_PATH = Path("models/sentiment_model.pkl")

bundle = joblib.load(VECT_PATH)
tfidf = bundle["vectorizer"]
X_train = tfidf.transform(bundle["X_train"])
y_train = bundle["y_train"]

clf = LogisticRegression(
    max_iter=300,
    n_jobs=-1
)
clf.fit(X_train, y_train)

joblib.dump({
    "model": clf,
    "vectorizer": tfidf
}, MODEL_PATH)

print("✅ Model trained and saved to", MODEL_PATH)

8. 評估與視覺化 (Evaluation & Visualizations)

這步就是你說的「挑幾張圖出來，要給網頁用」。

一、載入剛剛的模型＋TF-IDF
二、跑 test set → 產生
accuracy
precision, recall, f1
classification_report
confusion matrix

三、畫圖
visualizations/sentiment_distribution.png → 計算 positive / negative 比例
visualizations/confusion_matrix.png → 用 seaborn / matplotlib 畫
visualizations/top_words_positive.png / visualizations/top_words_negative.png → 從 TF-IDF 裡抓最高權重詞
📄 範例（04_evaluate.py）：
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = Path("models/sentiment_model.pkl")
DATA_PATH = Path("data/processed/mobile_reviews_clean.csv")
VIZ_DIR = Path("visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
vectorizer = bundle["vectorizer"]

df = pd.read_csv(DATA_PATH)
X = df['clean_text'].values
y = df['sentiment'].values

# 這裡也可以只用 test，不過我這裡全跑一次給你看
X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# 1) 混淆矩陣
cm = confusion_matrix(y, y_pred, labels=['negative','positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['neg','pos'],
            yticklabels=['neg','pos'])
plt.title("Confusion Matrix")
plt.savefig(VIZ_DIR / "confusion_matrix.png", bbox_inches='tight')

# 2) 情感分布
sent_counts = df['sentiment'].value_counts()
plt.figure(figsize=(5,4))
sent_counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.ylabel("Count")
plt.savefig(VIZ_DIR / "sentiment_distribution.png", bbox_inches='tight')

print("✅ visualizations saved to", VIZ_DIR)

9. 後處理 (Post-processing)
如果你的評論是多國語言（ZH / EN / ID / VN），可以在這步：
一、加語言偵測 → 非英文先翻譯
二、把 neutral 自動合併到 positive / negative，提高模型穩定度
三、做「品牌層級」或「手機型號層級」的彙總
e.g. df.groupby('model_name')['sentiment'].mean()
可以產一張 visualizations/sentiment_by_model.png
也可以在這步產出一個 report.csv：
review_id, raw_text, clean_text, predicted_sentiment, probability, model_version
...
這樣你做前端頁面就能直接吃。

10. Demo / Web 嵌入建議
你說你要「把他當 demo 做成網頁」，可以照這個做法：
一、後端先跑好上面的 pipeline，把圖片都輸出到 visualizations/
二、前端頁面直接讀固定路徑的圖
/visualizations/sentiment_distribution.png
/visualizations/confusion_matrix.png
三、再做一個簡單的輸入框，呼叫 05_inference.py 替你做即時預測

11. Inference（即時預測）
📄 範例（05_inference.py）：
import argparse
import joblib

bundle = joblib.load("models/sentiment_model.pkl")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

def predict(text: str):
    x = vectorizer.transform([text])
    pred = model.predict(x)[0]
    proba = getattr(model, "predict_proba", None)
    if proba:
        p = model.predict_proba(x).max()
    else:
        p = None
    return pred, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="review text to classify")
    args = parser.parse_args()
    label, prob = predict(args.text)
    print(f"🔮 Sentiment: {label} (prob={prob})")

12. git 指令（自己 push）
這樣下就好：
git add README.md scripts/*.py visualizations/*.png
git commit -m "add detailed README and pipeline scripts"
git push origin main

Author: Rita Lin
Contact with me: msmile09@hotmail.com
Website: http://ritalinyutzu.vercel.app/
