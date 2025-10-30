"""
下載 Kaggle 數據集
Mobile Reviews Sentiment and Specification
"""

import os
from pathlib import Path
import zipfile
import subprocess

# 創建目錄
data_dir = Path('data/raw')
data_dir.mkdir(parents=True, exist_ok=True)

print("🔽 下載數據集...")
print("數據集: mohankrishnathalla/mobile-reviews-sentiment-and-specification")

# 下載數據集
try:
    subprocess.run([
        'kaggle', 'datasets', 'download', 
        '-d', 'mohankrishnathalla/mobile-reviews-sentiment-and-specification',
        '-p', str(data_dir)
    ], check=True)
    
    print("✅ 下載完成")
    
    # 解壓縮
    zip_file = data_dir / 'mobile-reviews-sentiment-and-specification.zip'
    if zip_file.exists():
        print("📦 解壓縮中...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✅ 解壓完成")
        
        # 刪除 zip 文件
        zip_file.unlink()
        print("🗑️  清理 zip 文件")
        
        # 列出文件
        print("\n📁 數據文件:")
        for file in data_dir.glob('*'):
            size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {file.name} ({size:.2f} MB)")
            
except subprocess.CalledProcessError:
    print("❌ 下載失敗")
    print("\n請確認:")
    print("1. Kaggle API 已配置 (~/.kaggle/kaggle.json)")
    print("2. 已接受數據集的條款")
    print("\n手動下載:")
    print("https://www.kaggle.com/datasets/mohankrishnathalla/mobile-reviews-sentiment-and-specification/data")
