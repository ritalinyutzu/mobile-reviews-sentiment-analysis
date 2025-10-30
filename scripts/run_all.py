"""
手機評論情感分析 - 主控腳本
運行完整的分析流程
"""

import subprocess
import sys
from pathlib import Path
import time

def print_header(text):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """運行 Python 腳本"""
    print_header(f"Step: {description}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {description} 完成 (耗時: {elapsed:.1f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失敗")
        print(f"錯誤: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ 找不到腳本: {script_name}")
        return False

def main():
    print_header("手機評論情感分析 - 完整流程")
    
    # 創建目錄結構
    print("📁 創建目錄結構...")
    dirs = [
        'data/raw',
        'data/processed',
        'visualizations',
        'models',
        'scripts'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ 目錄結構已創建\n")
    
    # 定義流程
    steps = [
        ('download_data.py', '下載數據集'),
        ('01_eda.py', '探索性數據分析'),
        ('02_preprocessing.py', '文本預處理'),
        ('03_modeling.py', '模型訓練與評估')
    ]
    
    # 詢問是否跳過下載
    print("⚠️  如果數據已下載，可以跳過下載步驟")
    skip_download = input("是否跳過下載數據？(y/n): ").lower() == 'y'
    
    if skip_download:
        steps = steps[1:]  # 跳過第一步
    
    # 執行流程
    total_start = time.time()
    success_count = 0
    
    for i, (script, desc) in enumerate(steps, 1):
        print(f"\n進度: {i}/{len(steps)}")
        
        if run_script(script, desc):
            success_count += 1
        else:
            print(f"\n⚠️  流程在 '{desc}' 步驟中斷")
            choice = input("是否繼續下一步？(y/n): ").lower()
            if choice != 'y':
                break
    
    # 總結
    total_time = time.time() - total_start
    
    print_header("完成總結")
    print(f"成功完成: {success_count}/{len(steps)} 個步驟")
    print(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
    
    if success_count == len(steps):
        print("\n🎉 所有步驟成功完成！")
        print("\n生成的文件:")
        print("  📊 視覺化圖表: visualizations/")
        print("  💾 處理後數據: data/processed/")
        print("  🤖 訓練模型: models/")
        print("  📄 分析報告: data/processed/ 和 models/")
    else:
        print("\n⚠️  部分步驟未完成，請檢查錯誤訊息")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
