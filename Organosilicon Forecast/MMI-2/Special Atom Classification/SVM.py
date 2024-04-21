from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pickle



# 加载数据(需要对数据进行归一化)
data = pd.read_excel('data_muti2.xlsx')

# 定义输入和输出
X = data.drop(['output'], axis=1)  # 输入变量
y = data['output']  # 输出变量，更改为对应的分类列

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 由于SVM对输入的规模敏感，我们使用管道将SVM和StandardScaler结合起来
pipe = make_pipeline(StandardScaler(), SVC(probability=True))

# 定义要搜索的参数网格
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# 使用最佳参数重新训练模型
best_svm = grid_search.best_estimator_
y_pred_train = best_svm.predict(X_train)
precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred_train, average='weighted')
accuracy = accuracy_score(y_train, y_pred_train)

print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", fscore)
print("Support: ", support)  # 只在没有使用平均值时显示
print("Accuracy: ", accuracy)
try:
    # 计算并打印 ROC AUC 分数; 适用于二分类问题
    roc_auc = roc_auc_score(y_test, best_svm.predict_proba(X_test)[:, 1])
    print("ROC AUC Score: ", roc_auc)
except ValueError:
    # 无法计算 ROC AUC 时，可能是因为它不适用于当前的分类问题
    print("ROC AUC couldn't be calculated for this case.")

# 预测和评估
# 预测和评估
y_pred = best_svm.predict(X_test)

# 计算各种评价指标
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", fscore)
print("Support: ", support)  # 只在没有使用平均值时显示
print("Accuracy: ", accuracy)

# 如果你的类别是二分类，还可以计算 ROC AUC 分数
# 对于多分类任务，确保你的标签已经被二值化，或者使用 `average` 参数
try:
    # 计算并打印 ROC AUC 分数; 适用于二分类问题
    roc_auc = roc_auc_score(y_test, best_svm.predict_proba(X_test)[:, 1])
    print("ROC AUC Score: ", roc_auc)
except ValueError:
    # 无法计算 ROC AUC 时，可能是因为它不适用于当前的分类问题
    print("ROC AUC couldn't be calculated for this case.")


s=pickle.dumps(best_svm)
f=open('svm.model', "wb+")
f.write(s)
f.close()
