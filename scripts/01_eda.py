"""
手機評論情感分析 - 探索性數據分析 (EDA)
Mobile Reviews Sentiment Analysis - Exploratory Data Analysis
"""

# %% 導入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# 創建輸出目錄
Path('visualizations').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("手機評論情感分析 - EDA")
print("=" * 80)

# %% 載入數據
print("\n📂 載入數據...")

# 嘗試找到數據文件
data_path = Path('data/raw')
csv_files = list(data_path.glob('*.csv'))

if not csv_files:
    print("❌ 未找到數據文件")
    print("請先運行 download_data.py 下載數據")
    exit()

# 載入主要數據文件
df = pd.read_csv(csv_files[0])

print(f"✅ 數據載入成功")
print(f"數據集: {csv_files[0].name}")
print(f"形狀: {df.shape[0]:,} 行 × {df.shape[1]} 列")

# %% 數據基本資訊
print("\n" + "=" * 80)
print("📊 數據基本資訊")
print("=" * 80)

print("\n欄位列表:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"{i:2d}. {col:30s} | 類型: {str(dtype):10s} | 缺失: {null_count:6,} ({null_pct:5.2f}%)")

print(f"\n總記憶體使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %% 查看前幾行
print("\n" + "=" * 80)
print("👀 數據預覽")
print("=" * 80)
print(df.head(10))

# %% 數據類型分佈
print("\n" + "=" * 80)
print("📋 數據類型分佈")
print("=" * 80)

dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

# %% 數值欄位統計
print("\n" + "=" * 80)
print("📈 數值欄位統計")
print("=" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print(df[numeric_cols].describe())
else:
    print("無數值欄位")

# %% 缺失值分析
print("\n" + "=" * 80)
print("🔍 缺失值分析")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失數量': missing,
    '缺失百分比': missing_pct
}).sort_values('缺失數量', ascending=False)

print(missing_df[missing_df['缺失數量'] > 0])

if missing.sum() > 0:
    # 視覺化缺失值
    plt.figure(figsize=(14, 6))
    missing_cols = missing_df[missing_df['缺失數量'] > 0].head(15)
    
    plt.barh(range(len(missing_cols)), missing_cols['缺失百分比'], color='coral', alpha=0.7)
    plt.yticks(range(len(missing_cols)), missing_cols.index)
    plt.xlabel('缺失百分比 (%)')
    plt.title('缺失值分析 - Top 15 欄位', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(missing_cols['缺失百分比']):
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/01_missing_values.png', dpi=300, bbox_inches='tight')
    print("✅ 缺失值圖表已保存")
    plt.show()
else:
    print("✅ 無缺失值")

# %% 識別文本欄位和情感欄位
print("\n" + "=" * 80)
print("🔤 識別關鍵欄位")
print("=" * 80)

# 尋找可能的評論文本欄位
text_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        avg_length = df[col].astype(str).str.len().mean()
        if avg_length > 50:  # 假設評論文本長度 > 50
            text_cols.append(col)
            print(f"文本欄位: {col} (平均長度: {avg_length:.0f})")

# 尋找可能的情感/評分欄位
rating_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['rating', 'score', 'sentiment', 'star']):
        rating_cols.append(col)
        print(f"評分欄位: {col}")

# %% 分類變數分析
print("\n" + "=" * 80)
print("📊 分類變數分析")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# 排除長文本欄位
categorical_cols = [col for col in categorical_cols if col not in text_cols]

for col in categorical_cols[:5]:  # 只顯示前5個
    print(f"\n{col}:")
    value_counts = df[col].value_counts().head(10)
    print(value_counts)
    
    # 視覺化
    if len(df[col].unique()) <= 20:
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='barh', color='steelblue', alpha=0.7)
        plt.title(f'{col} 分佈', fontsize=16, fontweight='bold')
        plt.xlabel('數量')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'visualizations/02_distribution_{col.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# %% 文本長度分析
if text_cols:
    print("\n" + "=" * 80)
    print("📝 文本分析")
    print("=" * 80)
    
    for text_col in text_cols[:2]:  # 分析前2個文本欄位
        print(f"\n分析欄位: {text_col}")
        
        # 計算文本長度
        df[f'{text_col}_length'] = df[text_col].astype(str).str.len()
        df[f'{text_col}_word_count'] = df[text_col].astype(str).str.split().str.len()
        
        print(f"字元數統計:")
        print(df[f'{text_col}_length'].describe())
        print(f"\n詞數統計:")
        print(df[f'{text_col}_word_count'].describe())
        
        # 視覺化文本長度分佈
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 字元數分佈
        axes[0].hist(df[f'{text_col}_length'].dropna(), bins=50, 
                    color='skyblue', alpha=0.7, edgecolor='black')
        axes[0].set_title(f'{text_col} - 字元數分佈', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('字元數')
        axes[0].set_ylabel('頻率')
        axes[0].grid(True, alpha=0.3)
        
        # 詞數分佈
        axes[1].hist(df[f'{text_col}_word_count'].dropna(), bins=50, 
                    color='lightcoral', alpha=0.7, edgecolor='black')
        axes[1].set_title(f'{text_col} - 詞數分佈', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('詞數')
        axes[1].set_ylabel('頻率')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/03_text_length_{text_col.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# %% 評分/情感分析
if rating_cols:
    print("\n" + "=" * 80)
    print("⭐ 評分/情感分析")
    print("=" * 80)
    
    for rating_col in rating_cols[:2]:
        print(f"\n分析欄位: {rating_col}")
        print(df[rating_col].value_counts().sort_index())
        
        # 視覺化
        plt.figure(figsize=(12, 6))
        
        if df[rating_col].dtype in ['int64', 'float64']:
            # 數值評分
            df[rating_col].hist(bins=30, color='gold', alpha=0.7, edgecolor='black')
            plt.title(f'{rating_col} 分佈', fontsize=16, fontweight='bold')
            plt.xlabel(rating_col)
            plt.ylabel('頻率')
        else:
            # 分類評分
            df[rating_col].value_counts().plot(kind='bar', color='gold', alpha=0.7)
            plt.title(f'{rating_col} 分佈', fontsize=16, fontweight='bold')
            plt.xlabel(rating_col)
            plt.ylabel('數量')
            plt.xticks(rotation=45, ha='right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'visualizations/04_rating_{rating_col.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# %% 相關性分析
if len(numeric_cols) > 1:
    print("\n" + "=" * 80)
    print("🔗 相關性分析")
    print("=" * 80)
    
    correlation = df[numeric_cols].corr()
    print(correlation)
    
    # 熱圖
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('特徵相關性熱圖', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 相關性熱圖已保存")
    plt.show()

# %% 保存處理後的數據
print("\n" + "=" * 80)
print("💾 保存數據")
print("=" * 80)

# 保存基本統計
summary_path = Path('data/processed/eda_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EDA 摘要報告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"數據集形狀: {df.shape[0]:,} 行 × {df.shape[1]} 列\n\n")
    
    f.write("欄位資訊:\n")
    f.write("-" * 80 + "\n")
    for col in df.columns:
        f.write(f"{col}: {df[col].dtype}\n")
    
    f.write("\n缺失值:\n")
    f.write("-" * 80 + "\n")
    f.write(str(missing_df[missing_df['缺失數量'] > 0]))
    
    if numeric_cols:
        f.write("\n\n數值欄位統計:\n")
        f.write("-" * 80 + "\n")
        f.write(str(df[numeric_cols].describe()))

print(f"✅ EDA 摘要已保存至: {summary_path}")

# 保存增強後的數據
enhanced_path = Path('data/processed/enhanced_data.csv')
df.to_csv(enhanced_path, index=False)
print(f"✅ 增強數據已保存至: {enhanced_path}")

# %% 總結
print("\n" + "=" * 80)
print("🎉 EDA 完成")
print("=" * 80)
print(f"\n生成的文件:")
print(f"  📊 視覺化圖表: visualizations/ 目錄")
print(f"  📄 EDA 摘要: {summary_path}")
print(f"  💾 增強數據: {enhanced_path}")
print("\n下一步: 數據預處理與特徵工程")
print("=" * 80)
