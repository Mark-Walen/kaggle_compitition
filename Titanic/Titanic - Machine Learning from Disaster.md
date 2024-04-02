# Titanic - Machine Learning from Disaster

## The Challenge

在历史上，泰坦尼克号的沉没是最臭名昭著的海难之一。

1912年4月15日，在她的处女航中，广受称赞的“永不沉没”的 RMS 泰坦尼克号与冰山相撞后沉没。不幸的是，救生艇数量不足以容纳所有乘客和机组人员，导致2224名乘客和机组人员中的1502人死亡。

尽管生存中有一定的运气成分，但似乎某些人群比其他人更有可能生存。

在这个挑战中，**我们要求你建立一个预测模型来回答这个问题：“哪些人更有可能生存？”** 使用乘客数据（例如姓名、年龄、性别、社会经济阶层等）。

## Data

Start here! Predict survival on the Titanic and get familiar with ML basics

![data_dict](.\image\data_dict.png)

### Variable Notes

**pclass**: 社会经济地位
1st = Upper
2nd = Middle
3rd = Lower

**age**: 如果年龄是分数，表示小于 1. 如果年龄是估计值的形式: xx.5

**sibsp**: 兄弟姐妹和夫妻关系
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: 父母和孩子关系
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
一些孩子和祖父母一起出行, 对应的 parch 为 0.

## 评估

### 目标

你的任务是预测乘客是否在泰坦尼克号沉没中幸存。
对于测试集中的每个乘客，你变量预测值必须为 0 或 1。

### 度量标准

你的得分是你正确预测的乘客的百分比。这被称为准确率。

### 提交文件的格式

你应该提交一个带有确切 418 条记录加上一个标题行的CSV文件。如果你有额外的列（超出 PassengerId 和Survived），或者行数不对，你的提交将会出错。
文件应该有确切2列：

PassengerId（以任何顺序排序）
Survived（包含你的二元预测：1表示幸存，0表示死亡）

```text
PassengerId,Survived
892,0
893,1
894,0
Etc.
```

