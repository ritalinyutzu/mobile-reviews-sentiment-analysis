"""
手機評論情感分析 - 模型訓練與評估
Model Training and Evaluation
"""

# %% 導入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.sparse import load_npz

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# 設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
Path('visualizations').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

print("=" * 80)
print("模型訓練與評估")
print("=" * 80)

# %% 載入數據
print("\n📂 載入數據...")

df = pd.read_csv('data/processed/preprocessed_data.csv')
X = load_npz('data/processed/tfidf_features.npz')
y = df['sentiment_label']

print(f"✅ 特徵矩陣: {X.shape}")
print(f"✅ 標籤分佈:\n{y.value_counts()}")

# %% 資料分割
print("\n✂️  分割訓練集與測試集...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ 訓練集: {X_train.shape[0]:,} 樣本")
print(f"✅ 測試集: {X_test.shape[0]:,} 樣本")

# %% 定義模型
print("\n🤖 定義模型...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print(f"✅ 定義了 {len(models)} 個模型")

# %% 訓練與評估所有模型
print("\n🏋️  訓練模型...")

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"訓練: {name}")
    print('='*60)
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # 交叉驗證（選擇部分模型以節省時間）
    if name in ['Logistic Regression', 'Naive Bayes']:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean, cv_std = None, None
    
    # 保存結果
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'y_pred': y_pred
    }
    
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if cv_mean is not None:
        print(f"交叉驗證: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # 詳細分類報告
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, zero_division=0))

# %% 模型比較
print("\n" + "=" * 80)
print("📊 模型比較")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
print(comparison_df.to_string(index=False))

# 視覺化模型比較
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                      color='steelblue', alpha=0.7, legend=False)
    ax.set_title(f'{metric} 比較', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # 顯示數值
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/11_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ 模型比較圖已保存")
plt.show()

# %% 找出最佳模型
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_y_pred = results[best_model_name]['y_pred']

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"準確率: {results[best_model_name]['accuracy']:.4f}")

# %% 混淆矩陣
print("\n📊 生成混淆矩陣...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    if idx >= 6:
        break
    
    cm = confusion_matrix(y_test, result['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               xticklabels=sorted(y.unique()),
               yticklabels=sorted(y.unique()))
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('真實標籤')
    axes[idx].set_xlabel('預測標籤')

plt.tight_layout()
plt.savefig('visualizations/12_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ 混淆矩陣已保存")
plt.show()

# %% 最佳模型詳細分析
print(f"\n🔍 {best_model_name} 詳細分析...")

# 混淆矩陣
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
           xticklabels=sorted(y.unique()),
           yticklabels=sorted(y.unique()),
           cbar_kws={'label': '數量'})
plt.title(f'{best_model_name} - 混淆矩陣', fontsize=16, fontweight='bold')
plt.ylabel('真實標籤', fontsize=12)
plt.xlabel('預測標籤', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/13_best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ 最佳模型混淆矩陣已保存")
plt.show()

# %% 特徵重要性分析（如果模型支援）
if hasattr(best_model, 'coef_'):
    print("\n📊 特徵重要性分析...")
    
    # 載入 TF-IDF 向量器
    with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    feature_names = tfidf.get_feature_names_out()
    
    # 獲取係數（多類別）
    if len(best_model.coef_.shape) > 1:
        # 取平均或選擇特定類別
        coef = np.abs(best_model.coef_).mean(axis=0)
    else:
        coef = best_model.coef_
    
    # Top 特徵
    top_n = 30
    top_indices = np.argsort(coef)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = [coef[i] for i in top_indices]
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_values, color='darkgreen', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('重要性')
    plt.title(f'{best_model_name} - Top {top_n} 重要特徵', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('visualizations/14_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ 特徵重要性圖已保存")
    plt.show()

# %% 錯誤分析
print("\n🔍 錯誤分析...")

# 找出預測錯誤的樣本
errors = y_test != best_y_pred
error_indices = np.where(errors)[0]

print(f"錯誤預測數: {errors.sum():,} ({errors.sum()/len(y_test)*100:.2f}%)")

# 錯誤類型分析
error_analysis = pd.DataFrame({
    'True': y_test.values[errors],
    'Predicted': best_y_pred[errors]
})
error_counts = error_analysis.groupby(['True', 'Predicted']).size().reset_index(name='Count')
error_counts = error_counts.sort_values('Count', ascending=False)

print("\n錯誤類型分佈:")
print(error_counts.to_string(index=False))

# %% ROC 曲線（多類別）
if len(np.unique(y)) <= 3:  # 只對 3 類或更少繪製 ROC
    print("\n📈 生成 ROC 曲線...")
    
    # 二值化標籤
    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
    n_classes = y_test_bin.shape[1]
    
    # 獲取預測概率
    if hasattr(best_model, 'predict_proba'):
        y_score = best_model.predict_proba(X_test)
    elif hasattr(best_model, 'decision_function'):
        y_score = best_model.decision_function(X_test)
        # 標準化到 [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        y_score = scaler.fit_transform(y_score)
    else:
        print("⚠️  該模型不支援概率預測，跳過 ROC 曲線")
        y_score = None
    
    if y_score is not None:
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'green', 'blue']
        for i, (color, class_name) in enumerate(zip(colors, sorted(y.unique()))):
            if i < n_classes:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{best_model_name} - ROC 曲線', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/15_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✅ ROC 曲線已保存")
        plt.show()

# %% 保存最佳模型
print("\n💾 保存最佳模型...")

model_path = Path(f'models/best_model_{best_model_name.replace(" ", "_")}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ 最佳模型已保存: {model_path}")

# 保存所有模型結果
results_path = Path('models/all_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"✅ 所有結果已保存: {results_path}")

# %% 生成模型報告
print("\n📄 生成模型報告...")

report_path = Path('models/model_evaluation_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("模型評估報告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"數據集大小:\n")
    f.write(f"  訓練集: {X_train.shape[0]:,} 樣本\n")
    f.write(f"  測試集: {X_test.shape[0]:,} 樣本\n")
    f.write(f"  特徵數: {X.shape[1]:,}\n\n")
    
    f.write("模型比較:\n")
    f.write("-" * 80 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")
    
    f.write(f"最佳模型: {best_model_name}\n")
    f.write("-" * 80 + "\n")
    f.write(f"準確率: {results[best_model_name]['accuracy']:.4f}\n")
    f.write(f"精確率: {results[best_model_name]['precision']:.4f}\n")
    f.write(f"召回率: {results[best_model_name]['recall']:.4f}\n")
    f.write(f"F1-Score: {results[best_model_name]['f1']:.4f}\n\n")
    
    f.write("詳細分類報告:\n")
    f.write("-" * 80 + "\n")
    f.write(classification_report(y_test, best_y_pred, zero_division=0))
    f.write("\n\n")
    
    f.write("錯誤分析:\n")
    f.write("-" * 80 + "\n")
    f.write(f"錯誤預測數: {errors.sum():,} ({errors.sum()/len(y_test)*100:.2f}%)\n\n")
    f.write("錯誤類型:\n")
    f.write(error_counts.to_string(index=False))

print(f"✅ 模型報告已保存: {report_path}")

# %% 總結
print("\n" + "=" * 80)
print("🎉 模型訓練與評估完成")
print("=" * 80)
print(f"\n最佳模型: {best_model_name}")
print(f"準確率: {results[best_model_name]['accuracy']:.4f}")
print(f"\n生成的文件:")
print(f"  📊 視覺化: visualizations/ (5 張圖)")
print(f"  🤖 最佳模型: {model_path}")
print(f"  💾 所有結果: {results_path}")
print(f"  📄 評估報告: {report_path}")
print("\n完成！可以使用最佳模型進行預測")
print("=" * 80)
