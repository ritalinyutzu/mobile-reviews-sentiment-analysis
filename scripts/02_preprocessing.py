"""
手機評論情感分析 - 文本預處理
Text Preprocessing and Feature Engineering
"""

# %% 導入套件
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 下載必要的 NLTK 資源
print("📥 下載 NLTK 資源...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("✅ NLTK 資源下載完成")

# 設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
Path('visualizations').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("文本預處理與特徵工程")
print("=" * 80)

# %% 載入數據
print("\n📂 載入數據...")
df = pd.read_csv('data/processed/enhanced_data.csv')
print(f"✅ 載入 {len(df):,} 條記錄")

# %% 識別文本欄位
print("\n🔍 識別文本欄位...")
text_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        avg_length = df[col].astype(str).str.len().mean()
        if avg_length > 50:
            text_cols.append(col)
            print(f"  - {col} (平均長度: {avg_length:.0f})")

if not text_cols:
    print("❌ 未找到文本欄位")
    exit()

# 選擇主要文本欄位
main_text_col = text_cols[0]
print(f"\n✅ 主要文本欄位: {main_text_col}")

# %% 文本清洗函數
def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    # 轉小寫
    text = str(text).lower()
    
    # 移除 URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # 移除 email
    text = re.sub(r'\S+@\S+', '', text)
    
    # 移除 HTML 標籤
    text = re.sub(r'<.*?>', '', text)
    
    # 保留字母、數字和空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 移除多餘空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# %% 進階文本處理函數
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """進階文本預處理"""
    # 清洗
    text = clean_text(text)
    
    # 分詞
    tokens = word_tokenize(text)
    
    # 移除停用詞
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # 詞形還原
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

# %% 執行文本清洗
print("\n🧹 執行文本清洗...")
print("這可能需要幾分鐘...")

df['cleaned_text'] = df[main_text_col].apply(clean_text)
df['processed_text'] = df[main_text_col].apply(
    lambda x: preprocess_text(x, remove_stopwords=True, lemmatize=True)
)

# 過濾空文本
df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

print(f"✅ 清洗完成，保留 {len(df):,} 條有效記錄")

# %% 計算情感極性
print("\n😊 計算情感極性...")

def get_sentiment_scores(text):
    """使用 TextBlob 計算情感分數"""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0, 0

sentiment_scores = df['cleaned_text'].apply(get_sentiment_scores)
df['polarity'] = sentiment_scores.apply(lambda x: x[0])
df['subjectivity'] = sentiment_scores.apply(lambda x: x[1])

# 分類情感
def classify_sentiment(polarity):
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['polarity'].apply(classify_sentiment)

print("✅ 情感分析完成")
print("\n情感分佈:")
print(df['sentiment_label'].value_counts())

# %% 視覺化情感分佈
print("\n📊 生成情感分佈圖...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 情感標籤分佈
sentiment_counts = df['sentiment_label'].value_counts()
colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values,
              color=[colors.get(x, 'blue') for x in sentiment_counts.index],
              alpha=0.7, edgecolor='black')
axes[0, 0].set_title('情感標籤分佈', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('數量')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. 極性分數分佈
axes[0, 1].hist(df['polarity'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='中性線')
axes[0, 1].set_title('極性分數分佈', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('極性 (-1: 負面, +1: 正面)')
axes[0, 1].set_ylabel('頻率')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 主觀性分數分佈
axes[1, 0].hist(df['subjectivity'], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('主觀性分數分佈', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('主觀性 (0: 客觀, 1: 主觀)')
axes[1, 0].set_ylabel('頻率')
axes[1, 0].grid(True, alpha=0.3)

# 4. 極性 vs 主觀性
scatter = axes[1, 1].scatter(df['subjectivity'], df['polarity'], 
                           c=df['sentiment_label'].map(colors),
                           alpha=0.5, s=20, edgecolors='none')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].set_title('極性 vs 主觀性', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('主觀性')
axes[1, 1].set_ylabel('極性')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_sentiment_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 情感分析圖已保存")
plt.show()

# %% 詞頻分析
print("\n📊 詞頻分析...")

# 所有詞
all_words = ' '.join(df['processed_text']).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(30)

print("\nTop 30 最常見詞:")
for word, count in top_words[:10]:
    print(f"  {word:20s}: {count:6,}")

# 視覺化 Top 詞頻
plt.figure(figsize=(14, 8))
words, counts = zip(*top_words)
plt.barh(range(len(words)), counts, color='steelblue', alpha=0.7)
plt.yticks(range(len(words)), words)
plt.xlabel('頻率')
plt.title('Top 30 最常見詞', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/07_word_frequency.png', dpi=300, bbox_inches='tight')
print("✅ 詞頻圖已保存")
plt.show()

# %% 按情感分析詞頻
print("\n📊 按情感分析詞頻...")

for sentiment in ['positive', 'negative']:
    sentiment_text = ' '.join(df[df['sentiment_label'] == sentiment]['processed_text'])
    words = sentiment_text.split()
    word_freq = Counter(words)
    
    print(f"\n{sentiment.upper()} 評論 Top 20:")
    for word, count in word_freq.most_common(20)[:10]:
        print(f"  {word:20s}: {count:6,}")

# %% 詞雲
print("\n☁️  生成詞雲...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sentiments = ['positive', 'neutral', 'negative']
colors_wc = ['Greens', 'Greys', 'Reds']

for idx, (sentiment, cmap) in enumerate(zip(sentiments, colors_wc)):
    text = ' '.join(df[df['sentiment_label'] == sentiment]['processed_text'])
    
    if text:
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            colormap=cmap,
                            max_words=100).generate(text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{sentiment.upper()} 評論詞雲', 
                          fontsize=14, fontweight='bold')
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('visualizations/08_wordclouds.png', dpi=300, bbox_inches='tight')
print("✅ 詞雲已保存")
plt.show()

# %% TF-IDF 特徵提取
print("\n🔤 TF-IDF 特徵提取...")

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['processed_text'])

print(f"✅ TF-IDF 矩陣形狀: {tfidf_matrix.shape}")

# 提取 top TF-IDF 詞
feature_names = tfidf.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
tfidf_df = pd.DataFrame(denselist, columns=feature_names)

# Top TF-IDF 特徵
top_tfidf = tfidf_df.mean().sort_values(ascending=False).head(30)

plt.figure(figsize=(14, 8))
top_tfidf.plot(kind='barh', color='darkgreen', alpha=0.7)
plt.xlabel('平均 TF-IDF 分數')
plt.title('Top 30 TF-IDF 特徵', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/09_tfidf_features.png', dpi=300, bbox_inches='tight')
print("✅ TF-IDF 特徵圖已保存")
plt.show()

# %% N-gram 分析
print("\n📝 N-gram 分析...")

# Bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=30)
bigrams = vectorizer.fit_transform(df['processed_text'])
bigram_freq = dict(zip(vectorizer.get_feature_names_out(), 
                       bigrams.sum(axis=0).tolist()[0]))
bigram_freq = dict(sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True))

print("\nTop 20 Bigrams:")
for bigram, count in list(bigram_freq.items())[:20]:
    print(f"  {bigram:30s}: {count:6,}")

# 視覺化
plt.figure(figsize=(14, 8))
bigrams_list = list(bigram_freq.items())[:25]
phrases, counts = zip(*bigrams_list)
plt.barh(range(len(phrases)), counts, color='purple', alpha=0.7)
plt.yticks(range(len(phrases)), phrases)
plt.xlabel('頻率')
plt.title('Top 25 Bigrams (兩詞組合)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/10_bigrams.png', dpi=300, bbox_inches='tight')
print("✅ Bigram 圖已保存")
plt.show()

# %% 保存處理後的數據
print("\n💾 保存處理後的數據...")

# 保存完整數據
processed_path = Path('data/processed/preprocessed_data.csv')
df.to_csv(processed_path, index=False)
print(f"✅ 預處理數據已保存: {processed_path}")

# 保存 TF-IDF 特徵
import pickle
tfidf_path = Path('data/processed/tfidf_vectorizer.pkl')
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"✅ TF-IDF 向量器已保存: {tfidf_path}")

# 保存特徵矩陣
from scipy.sparse import save_npz
features_path = Path('data/processed/tfidf_features.npz')
save_npz(features_path, tfidf_matrix)
print(f"✅ TF-IDF 特徵矩陣已保存: {features_path}")

# %% 生成預處理報告
print("\n📄 生成預處理報告...")

report_path = Path('data/processed/preprocessing_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("文本預處理報告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"處理記錄數: {len(df):,}\n")
    f.write(f"主要文本欄位: {main_text_col}\n\n")
    
    f.write("情感分佈:\n")
    f.write(str(df['sentiment_label'].value_counts()))
    f.write("\n\n")
    
    f.write("極性統計:\n")
    f.write(str(df['polarity'].describe()))
    f.write("\n\n")
    
    f.write("Top 30 詞頻:\n")
    for word, count in top_words:
        f.write(f"  {word:30s}: {count:8,}\n")
    
    f.write("\n\nTF-IDF 特徵數: 1000\n")
    f.write(f"特徵矩陣形狀: {tfidf_matrix.shape}\n")

print(f"✅ 預處理報告已保存: {report_path}")

# %% 總結
print("\n" + "=" * 80)
print("🎉 文本預處理完成")
print("=" * 80)
print(f"\n生成的文件:")
print(f"  📊 視覺化: visualizations/ (6 張圖)")
print(f"  💾 預處理數據: {processed_path}")
print(f"  🔤 TF-IDF 向量器: {tfidf_path}")
print(f"  📊 特徵矩陣: {features_path}")
print(f"  📄 報告: {report_path}")
print("\n下一步: 模型訓練與評估")
print("=" * 80)
