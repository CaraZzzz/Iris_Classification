'''
本代码使用鸢尾花（Iris）数据集
数据集较为简单，决策树和朴素贝叶斯的测试集准确率都可以达到100%
'''

## 以下导入库的代码之前
## 复制  pip install scikit-learn matplotlib seaborn  到terminal（终端）
## VScode中打开终端：
## 方法1：ctrl + `
## 方法2：点击左上角 View -> Terminal

## 导入库
from sklearn.datasets import load_iris  # sklearn包自带的鸢尾花数据集
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 决策树算法函数
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯算法函数
from sklearn.model_selection import train_test_split    # 分割测试集与训练集函数
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 模型评估函数
import matplotlib.pyplot as plt # 画图的
import seaborn as sns # 画好看的热力图的


#############################  导入数据+划分训练集和测试集  #############################
## 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = list(iris.target_names) 

## 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#############################  决策树训练  #############################
print("Decision Tree \n")

## 创建决策树模型（使用基尼系数作为分裂标准）
'''
    对于一个包含K类别的数据集D，Gini index (基尼系数)如下：
    Gini(D) = 1 - \sum_{k=1}^K p_k^2
    p_k是数据集中第k个类别样本的占比
    Gini(D)值越大表示类别混合越混乱
'''
DecisionTree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)  # 加载模型
'''
    criterion='gini':
        用于指定“使用基尼系数作为分裂标准”
    max_depth=3:  
        决定树的深度。由于这个数据集比较简单，限制最大树深为3。
    random_state=42:  
        随机种子，保证程序每次运行出来结果一样。改变随机种子的数值可能会导致不一样的决策树。
'''
DecisionTree.fit(X_train, y_train) # 用训练集数据训练模型



#############################  朴素贝叶斯训练  #############################
print('Naive Bayes \n')

# 创建朴素贝叶斯分类器
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)


#############################  模型评估  #############################

# X_test, y_test = X, y 
## 如果要对整个数据集进行分类则加上这一行
## 如果只对测试集数据进行分类则注释掉这一行（用“#”）

print("Classification Result of Decision Tree \n")
## 预测
y_pred = DecisionTree.predict(X_test) # 用测试集X预测对应的y

## 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree: {accuracy*100:.2f}%")
print("\nClassification Report of Dicision Tree:")
print(classification_report(y_test, y_pred, target_names=class_names))

## 绘制结果
## 1. 用 matplotlib 绘制决策树
##### 此图重要，要放进report里
'''
每个节点包含以下关键信息（以根节点为例）：
   petal length (cm) <= 2.45
       分裂条件：如果花瓣长度 ≤ 2.45cm，则向左分支，否则向右分支。
   gini = 0.667
        当前节点的基尼不纯度（值越大表示类别混合越混乱）。
   samples = 120
        到达该节点的训练样本总数。
   value = [40, 41, 39]
        样本在3个类别的分布：[setosa, versicolor, virginica]。
   class = versicolor
        该节点的预测类别（多数类）。
颜色深浅反映节点的​​纯度​​：颜色越深，该节点中多数类的占比越高。
'''
plt.figure(figsize=(15, 10))
plot_tree(
    DecisionTree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree (Iris)")
plt.show()

## 2. 绘制混淆矩阵
##### 此图重要，要放进report里
'''
    混淆矩阵是什么、用来干什么建议搜索学习一下
    从混淆矩阵中可以计算classification report中的precision, recall f1-score
'''
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")  # 预测的标签
plt.ylabel("Actual Label")  # 真实的标签
plt.title("Confusion Matrix of Decision Tree")
plt.show()



print("Classification Result of Naive Bayes \n")
## 预测
y_pred_nb = NaiveBayes.predict(X_test)

## 评估模型
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nAccuracy of Naive Bayes: {accuracy*100:.2f}%")
print("\nClassification Report of Naive Bayes:")
print(classification_report(y_test, y_pred_nb, target_names=class_names))

## 绘制朴素贝叶斯分类的混淆矩阵
##### 此图重要，要放进report里
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, y_pred_nb),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix of Naive Bayes")
plt.show()


## 比较两种分类器的准确率
##### 看报告是否需要，可以不放进去
plt.figure(figsize=(6, 4))
plt.bar(['Decision Tree', 'Naive Bayesian'], [accuracy, accuracy_nb], color=['blue', 'green'])
plt.ylim(0.9, 1.05)
plt.ylabel("Accuracy")
plt.title("Comparison of Decision Tree and Naive Bayesian")
plt.show()