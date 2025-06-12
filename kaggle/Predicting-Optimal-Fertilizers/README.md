# 简介

为了更方便培训新同事如何针对已有数据训练并预测，所以记录于此，方便后续使用。
本文会根据低维度特征数据进行分类预测，包含数据分析、特征工程、数据预处理、数据集划分、模型训练、模型评估、模型预测、模型保存、模型加载、模型预测等步骤。
如果有不合适或者错误，欢迎提出。

农作物肥料预测 数据源来自kaggle比赛，链接如下：https://www.kaggle.com/competitions/playground-series-s5e6/data

## 分类预测流程

### 1. 找到目标

如果是比赛，那么直接看目标就行了，如果不是比赛，那么你需要根据你实际的场景去确定一个目标。因为通常来讲我们都是想要实现某个目标，然后才会采集相应的数据。
（当然，反过来已经有了一大推数据，我们想要根据这些数据去找到人类感知不到的一些联系，这又是另一码事了。）
根据比赛的要求，我们的目标为：根据不同的天气、土壤条件和作物选择最好的肥料。

### 2. 数据分析

我们通过对数据集的查看可以得到数据集如下：
温度、湿度、水分、氮、磷、钾、农作物类型、土壤类型、肥料名称（目标）
其中除了Soil Type、Crop Type、 Fertilizer Name这3个字段是object类型，其他字段的数据类型都是数字类型。 
分类目标是Fertilizer Name，我们需要对其进行预测。 
训练数据集的字段如下：

| 字段名称         | `id`            | `Temparature` | `Humidity` | `Moisture` | `Soil Type` | `Crop Type` | `Nitrogen` | `Phosphorous` | `Potassium` | `Fertilizer Name` |
|------------------|-----------------|---------------|------------|------------|-------------|-------------|------------|----------------|-------------|-------------------|
| **字段描述**     | 样本编号        | 温度          | 湿度       | 水分       | 土壤类型    | 作物类型    | 氮         | 磷             | 钾          | 肥料名称          |

数据分析里面的内容有很多，比如查看数据的缺失值、查看数据的分布、查看数据的相关性、查看数据的异常值、查看数据的分布等等。
这里只做一些简单的分析，提供下示例代码。

数据集基础分析：查看前几行数据，行数，统计结果简述
```python
def basic_info():
    '''
    数据集基础分析
    :return:
    '''
    # 查看前几行数据
    print("--查看前几行数据 --")
    print(df_train.head())

    # 查看数据基本信息（缺失值、数据类型等）
    print("--查看数据基本信息 --")
    print(df_train.info())

    # 统计描述（针对数值型字段）
    print("--统计描述 --")
    # 显示完整的描述统计结果，不进行截断
    pd.set_option('display.max_columns', None)
    print(df_train.describe())

    # 恢复列显示限制（可选）
    pd.reset_option('display.max_columns')
```
绘制数据类型直方图
```python
def plot_numeric_histograms():
    try:
        # 获取数值型列名列表
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

        # 设置图形布局参数
        ncols = 2
        nrows = (len(numeric_cols) + 1) // 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
        print("开始绘制直方图")

        for i, col in enumerate(numeric_cols):
            try:
                row, col_idx = divmod(i, ncols)
                sns.histplot(df_train[col], ax=axes[row, col_idx], kde=True)
            except Exception as e:
                print(f"绘制列 {col} 出错: {e}")

        # 隐藏多余的子图
        for j in range(i + 1, nrows * ncols):
            row, col_idx = divmod(j, ncols)
            fig.delaxes(axes[row, col_idx])

        plt.tight_layout()
        plt.show()  # 显示图像
        plt.close()  # 关闭图像资源，避免内存占用过高
    except Exception as e:
        print(f"绘制直方图出错: {e}")
```
绘制对象类型箱线图
```python
def plot_categorical_histograms():
    # 提取类别型列
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    soil_types = df_train['Soil Type'].unique()
    crop_types = df_train['Crop Type'].unique()
    print("soil_types:", soil_types)
    print("crop_types:", crop_types)

    # 设置图形布局参数
    ncols = 2
    nrows = (len(categorical_cols) + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    print("开始绘制类别型列分布")

    # 遍历所有类别型列并绘图
    for i, col in enumerate(categorical_cols):
        try:
            row, col_idx = divmod(i, ncols)  # 修复：使用 ncols 而不是 col
            sns.countplot(data=df_train, x=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Count Plot of {col}')
            axes[row, col_idx].tick_params(axis='x', rotation=45)  # 防止标签重叠
            print("{col}",col)
        except Exception as e:
            print(f"绘制列 {col} 出错: {e}")

    # 隐藏多余的子图
    for j in range(i + 1, nrows * ncols):
        row, col_idx = divmod(j, ncols)
        fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    plt.show()
    plt.close()  # 关闭图像资源，避免内存占用过高
```
数值类型数据target占比
```python
def feature_distribution_by_target():
    '''
    数值类型数据target占比
    :return:
    '''
    # 筛选数值型列（排除 object 类型）
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # 明确排除 'id' 和目标列（如果有）
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    # 排除可能包含的目标列本身（如果有）
    if 'Fertilizer Name' in numeric_cols:
        numeric_cols.remove('Fertilizer Name')
    # 按目标变量分组并计算均值
    grouped = df_train.groupby('Fertilizer Name')[numeric_cols].mean()

    # 打印表格形式的统计结果
    print("\n不同 Fertilizer Name 下各数值特征的平均值：")
    print(grouped)

    # 绘制热力图（Heatmap），展示不同类别下特征均值的差异
    plt.figure(figsize=(14, 10))
    sns.heatmap(grouped.T, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('平均值 Heatmap')
    plt.tight_layout()
    plt.show()
    plt.close()

    # 可选：绘制每个特征的柱状图，显示不同类别的均值
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Fertilizer Name', y=col, data=df_train, estimator=np.mean, ci=None)
        plt.xticks(rotation=45)
        plt.title(f'Mean {col} by Fertilizer Name')
        plt.tight_layout()
        plt.show()
        plt.close()
    # 查看object类型的数据
```
查看数值类型的数据分布
```python
def categorical_distribution_by_target():
    '''
    查看数据类型的数据分布
    :return:
    '''
    # 提取类别型列（排除 'id' 和目标列）
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'Fertilizer Name' in categorical_cols:
        categorical_cols.remove('Fertilizer Name')

    target_col = 'Fertilizer Name'

    for col in categorical_cols:
        # 创建交叉表
        cross_tab = pd.crosstab(df_train[target_col], df_train[col], normalize='index')

        # 绘制堆叠柱状图
        cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title(f'Distribution of {col} by Fertilizer Name (Normalized)')
        plt.xlabel('Fertilizer Name')
        plt.ylabel(f'Relative Frequency of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title=col)
        plt.show()
        plt.close()

```
查看目标值种类及其数量
```python
def target_column_analysis():
    # 查看肥料种类及其数量
    print(df_train['Fertilizer Name'].value_counts())

    # 可视化目标变量分布
    sns.countplot(y='Fertilizer Name', data=df_train, order=df_train['Fertilizer Name'].value_counts().index)
    plt.title('Distribution of Fertilizer Names')
    plt.tight_layout()
    plt.show()
```
### 3. 特征工程

我们使用人工智能肯定是为了解决现实中的问题，其实很多独立的数据是有着必然的关联的，这就需要一些先验的知识。
本次是土壤施肥的问题，那我们就按照这个举例子来进行特征工程。
1. 氮(N)、磷(P)和钾(K)是植物生长所必需的主要营养元素，它们在土壤中的总含量可以反映土壤肥力的整体水平。
2. 温度(Temparature)和湿度(Humidity)是影响植物生长的重要因素，它们的组合可以反映土壤的温度和湿度特性。
3. 水分(Moisture)是植物生长过程中不可或缺的一个因素，它可以反映土壤的水分含量。
4. 土壤类型(Soil Type)和作物类型(Crop Type)是影响植物生长的重要因素，它们的组合可以反映土壤和作物的特性。
5. 等等
 这里是举例说明的一种特征工程，实际中可能存在更多的特征，而且构建特征以后还需要shap分析等操作，
然后再回头改进特征工程，这是一个循环迭代的过程，往往不可能一次就能够满足需求。所以数据分析基本上是贯穿全局的，
不仅仅是“调参侠”还需要扎实的基础才能让你整合后的数据符合期望效果。
```python
import numpy as np
# -----------------------------
# 特征构造
# -----------------------------
def add_agricultural_features(df):
    """
    构造农业领域相关特征
    """
    df['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Env_Index'] = df['Temparature'] * df['Humidity'] * df['Moisture']
    df['Fertility_Score'] = (
            df['Nitrogen'] * 0.3 +
            df['Phosphorous'] * 0.3 +
            df['Potassium'] * 0.4
    )
    crop_n_preference = {
        'Wheat': 0.8,
        'Maize': 0.7,
        'Oil seeds': 0.3,
        'Paddy': 0.5,
        'Cotton': 0.6,
        'Barley': 0.7,
        'Millets': 0.5,
        'Sugarcane': 0.4,
        'Ground Nuts': 0.4,
        'Tobacco': 0.5,
        'Pulses': 0.4
    }
    df['Crop_Nitrogen_Preference'] = df['Crop Type'].map(crop_n_preference).fillna(0.5)
    df['Weighted_N'] = df['Nitrogen'] * df['Crop_Nitrogen_Preference']
    df['N_sqrt'] = np.sqrt(df['Nitrogen'])
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    return df

```

查看热力图及shap分析哪个元素影响大
```python
def plot_feature_vs_target(df, target_col):
    """
    绘制目标列与所有数值型特征之间的关系图

    参数:
        df (pd.DataFrame): 数据框
        target_col (str): 目标列名称（如 'Fertilizer Name'）
    """
    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于 DataFrame 中")

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 类别分布图
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_col, data=df, palette="Set2")
    plt.title(f'{target_col} 类别分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 数值型特征与目标的关系图
    for col in df.columns:
        if col != target_col and df[col].dtype != 'object':
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_col, y=col, data=df, palette="Set3")
            plt.title(f'{col} vs {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
def generate_data_report(df, target_col='Fertilizer Name'):
    """
    生成完整的数据探索报告：
        - 类别分布图
        - 每个数值特征在不同类别下的分布（boxplot）
        - 使用XGBoost + SHAP进行特征重要性分析
    """
    print("📊 开始生成数据探索报告...")

    # 1. 绘制目标列与各数值特征的关系图
    # plot_feature_vs_target(df, target_col)

    # 2. 构建训练集并训练轻量模型用于特征重要性分析
    print("🧠 训练XGBoost模型以评估特征重要性...")
    y = df['Fertilizer Name'].values
    # 编码标签（虽然你已处理过，但确保是整数形式）
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=[target_col]),
        y,
        test_size=0.2,
        random_state=42
    )

    # 数据预处理
    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # 训练轻量模型
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    model.fit(X_train_processed, y_train)

    # 3. 特征重要性可视化（SHAP）
    print("🔍 生成特征重要性图表...")
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X_train_processed)

        # 如果是多分类任务，取第一个类别的 SHAP 值
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 确保 X_train 是稠密数组
        if hasattr(X_train_processed, "toarray"):
            X_train_dense = X_train_processed.toarray()
        else:
            X_train_dense = X_train_processed

        # 获取正确的特征名称
        def get_transformed_column_names(preprocessor, numeric_features, categorical_features):
            transformers = []
            for name, trans, cols in preprocessor.transformers_:
                if trans == 'drop':
                    continue
                if name == 'cat':
                    new_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
                    transformers.extend(new_cols)
                else:
                    transformers.extend(cols)
            return transformers

        feature_names = get_transformed_column_names(preprocessor, numeric_features, categorical_features)

        # 绘制 summary plot
        shap.summary_plot(shap_values, X_train_dense, feature_names=feature_names, plot_type="bar")

    except Exception as e:
        print(f"⚠️ SHAP 图无法生成：{e}")

    print("✅ 数据探索报告已完成。")
```

### 4. 数据预处理
数据预处理这块：包括但不限于数据加载，生成特征，object类型编码，缺失值处理，目标值编码，切分训练集和测试集，数据标准化...
这里我直接吧全部代码贴出来，不再区分。
```python
# -----------------------------
# 特征构造
# -----------------------------
def add_agricultural_features(df):
    """
    构造农业领域相关特征
    """
    df['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Env_Index'] = df['Temparature'] * df['Humidity'] * df['Moisture']
    df['Fertility_Score'] = (
            df['Nitrogen'] * 0.3 +
            df['Phosphorous'] * 0.3 +
            df['Potassium'] * 0.4
    )
    crop_n_preference = {
        'Wheat': 0.8,
        'Maize': 0.7,
        'Oil seeds': 0.3,
        'Paddy': 0.5,
        'Cotton': 0.6,
        'Barley': 0.7,
        'Millets': 0.5,
        'Sugarcane': 0.4,
        'Ground Nuts': 0.4,
        'Tobacco': 0.5,
        'Pulses': 0.4
    }
    df['Crop_Nitrogen_Preference'] = df['Crop Type'].map(crop_n_preference).fillna(0.5)
    df['Weighted_N'] = df['Nitrogen'] * df['Crop_Nitrogen_Preference']
    df['N_sqrt'] = np.sqrt(df['Nitrogen'])
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    return df

def encode_and_transform(X, y=None, fit=False):
    """
    使用 ColumnTransformer 编码并标准化特征
    """
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    agri_feature_cols = [col for col in X.columns if col not in categorical_cols + numerical_cols]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import TargetEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', TargetEncoder(), categorical_cols),
            ('num', StandardScaler(), numerical_cols + agri_feature_cols)
        ],
        remainder='passthrough'
    )

    if fit:
        X_processed = preprocessor.fit_transform(X, y)
        joblib.dump(preprocessor, SCALER_PATH)
    else:
        preprocessor = joblib.load(SCALER_PATH)
        X_processed = preprocessor.transform(X)

    return X_processed

def save_processed_data(df, filename='processed_train.csv'):
    """
    保存处理后的DataFrame到CSV文件
    :param df: 处理后的DataFrame
    :param filename: 输出文件名
    """
    output_path = os.path.join(data_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ 已保存处理后的数据至 {output_path}")

# -----------------------------
# 数据预处理与编码
# -----------------------------
def preprocess_data(df, is_train=True):
    """
    对数据进行预处理（仅用于训练）
    :param df: 原始 DataFrame
    :param is_train: 是否是训练数据
    :return: 处理后的特征矩阵 X 和标签 y
    """
    df = add_agricultural_features(df)

    if is_train:
        X = df.drop(columns=['Fertilizer Name', 'id'])
        y = df['Fertilizer Name'].values
    else:
        X = df.drop(columns=['id'])
        y = None

    return X, y

def load_and_preprocess(data='train.csv'):
    file_path = os.path.join(DATA_DIR, data)
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df, is_train=(data == 'train.csv'))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, LABEL_ENCODER_PATH)

    X_processed = encode_and_transform(X, y_encoded, fit=True)

    feature_names = pd.DataFrame(X_processed).columns.tolist()
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_original = pd.read_csv(file_path)
    df_processed['Fertilizer Name'] = df_original['Fertilizer Name'].values
    save_processed_data(df_processed, 'processed_train.csv')

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42,
                                                      stratify=y_encoded)
    return X_train, X_val, y_train, y_val, le.classes_

```

### 5. 模型选择
到了这一步就是选择模型了，根据数据分析步骤中，我们大概就已经了解了我们本次的分类任务，那么根据什么去选择模型呢？
我个人是根据以下几个层级选择模型的：
1. 回归任务还是分类任务
2. 有监督还是无监督
3. 数据量、数据质量、模型复杂度、训练时间等等

说是这么说，但是实际选择的时候诸多论文大牛都已经发布了很多模型，如果根据实际需求去选择合适的模型，而不是一个个试，这也是考验工程师的技能。
本次模型数据量一共有75000，不大也不小，由于是分类任务，所以我拟定使用了XGBoost、lightgbm、catboost、随机森林、mlp等模型。
最后同样的数据下选择了效果最高的XGBoost，然后针对这一模型进行模型调参，最后将结果保存为csv文件。
这块直接调用库就行了。
```python
print("XGBoost 模型训练中...")
clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='hist'
)
eval_set = [(X_val, y_val)]
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_set=eval_set, verbose=False)
print("XGBoost训练结束")
# # LightGBM 示例
# clf = LGBMClassifier(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )
y_pred = clf.predict(X_val)
print("Val Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
```

### 6. 模型结果分析

这里模型结果分析主要是模型的性能指标（准确率、精准率、召回率、F1分数等）、类别分类报告、模型预测结果的混淆矩阵热力图等。
通过这些数据知道自己训练出来的模型哪些类别特别强，哪些类别特别弱等等。
然后根据这些数据再掉过头来数据分析、特征工程、数据加工再来一遍。

### 查看训练情况
```shell
tensorboard --logdir={#path_to_your_logs}
```