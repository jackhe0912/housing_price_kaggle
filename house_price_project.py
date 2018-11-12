import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

from sklearn.base import BaseEstimator,TransformerMixin, RegressorMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler  #数据包含许多异常值，使用均值和方差缩放可能并不是一个很好的选择，可以使用RobustScaler或robust_scaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline,make_pipeline
from scipy.stats import skew

from sklearn.model_selection import cross_val_predict,GridSearchCV,KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV,Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.linear_model import SGDRegressor,BayesianRidge
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgbm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def loadset():
    """
    加载数据,
    :return:
    """
    train=pd.read_csv('D:/machine learning/kaggle/housing pricekaggle/all/train.csv')
    test=pd.read_csv('D:/machine learning/kaggle/housing pricekaggle/all/test.csv')

    return train,test

def dropOutliers(dataset):
    """
    删除训练集中GrLivArea异常点
    :param dataset:
    :return:
    """
    dataset.drop(dataset[(dataset['GrLivArea']>4000)&(dataset['SalePrice']<300000)].index,inplace=True)


    return dataset

def concatSet(train,test):
    """
    j将训练集与测试集合并
    :param train:
    :param test:
    :return: dataset
    """
    dataset=pd.concat([train,test],ignore_index=True)

    dataset.drop('SalePrice',axis=1,inplace=True)

    return dataset

def firstdropfeature(dataset):
    """
    丢弃缺失值超过15%的特征以及 明显与目标值无关的特征  Utilities的值基本只有一个类别，所以应该与目标值无关
    :param dataset:
    :return:
    """
    dataset.drop(['Id','PoolQC','MiscFeature','Alley','Utilities'],axis=1,inplace=True)

    return dataset

def fillmissing(dataset):
    """
    填充缺失值
    :param dataset:
    :return:
    """
    ##根据Neighborhood整理LotFrontage
    dataset['LotAreaCut'] = pd.qcut(dataset.LotArea, 10)  # LotArea为连续值特征，将其qcut为10份
    dataset['LotFrontage'] = dataset.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform \
        (lambda x: x.fillna(x.median()))
    dataset['LotFrontage'] = dataset.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    #填充其他特征的缺失值
    cols = ['MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageArea']
    for i in cols:
        dataset[i].fillna(0, inplace=True)
    cols1 = ['Fence', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageYrBlt', 'GarageType',
             'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
    for i in cols1:
        dataset[i].fillna("None", inplace=True)
    # fill with mode  填充众数
    cols2 = ['MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Electrical', 'KitchenQual',
             'SaleType', 'Exterior1st', 'Exterior2nd']
    for col in cols2:
        dataset[col].fillna(dataset[col].mode().iloc[0], inplace=True)
    dataset.drop('LotAreaCut', axis=1, inplace=True)

    return dataset
def addmorefeat(dataset):
    """
    将地下室和第一层，第二层加在一起构成总的室内面积
    将总的室内面积加上车库面积构成总的居住面积
    :param dataset:
    :return:
    """
    dataset['TotalHouse'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    dataset['TotalArea'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'] + dataset['GarageArea']

    return dataset

def tranfer2str(dataset):
    """
    把一些伪数值特征转换成字符型特征,方便后续进行Labelencoder和getdummies
    :param dataset:
    :return:
    """
    Num2Str = ['MSSubClass', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'MoSold',
              'YrSold', 'YearBuilt', 'YearRemodAdd', 'LowQualFinSF', 'GarageYrBlt']

    for i in Num2Str:
        dataset[i] = dataset[i].astype(str)

    return dataset

def skew_dummiestransform(dataset):
    """
    检查数值型特征数据的偏度（skewness），对偏度过大的特征进行log1p转换，object类型的数据无法计算skewness,因此计算时要过滤掉
    偏度转换后，对数据进行getdummies
    """
    dataset_num=dataset.select_dtypes(exclude=['object'])

    skewness=dataset_num.apply(lambda x:skew(x))

    skewness_features=skewness[abs(skewness)>=0.8].index

    dataset[skewness_features]=np.log1p(dataset[skewness_features])

    return dataset
def replace_process(dataset):
    """
    将部分类别特征按照对应SalePrice的median和mean值，转换成序列值
    :param dataset:
    :return:
    """

    dataset['MSSubClass'] = dataset.MSSubClass.replace({'180': 1, '30': 2, '45': 2, '190': 3, '50': 3, '90': 3,
                                                        '85': 4, '40': 4, '160': 4, '70': 5, '20': 5, '75': 5, '80': 5,
                                                        '150': 5,
                                                        '120': 6, '60': 6})
    dataset['MSZoning'] = dataset.MSZoning.replace({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    dataset['Neighborhood'] = dataset.Neighborhood.replace({'MeadowV': 1,
                                                            'IDOTRR': 2, 'BrDale': 2,
                                                            'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                            'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                            'NPkVill': 5, 'Mitchel': 5,
                                                            'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                            'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                            'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                            'StoneBr': 9,
                                                            'NoRidge': 10, 'NridgHt': 10})

    dataset['Condition1'] = dataset.Condition1.replace({'Artery': 1, 'Feedr': 2, 'RRAe': 2, 'Norm': 3, 'RRAn': 3,
                                                        'PosN': 4, 'RRNe': 4,
                                                        'PosA': 5, 'RRNn': 5})

    dataset['BldgType'] = dataset.BldgType.replace({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    dataset['HouseStyle'] = dataset.HouseStyle.replace({'1.5Unf': 1,
                                                        '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                                        '1Story': 3, 'SLvl': 3,
                                                        '2Story': 4, '2.5Fin': 4})

    dataset['Exterior1st'] = dataset.Exterior1st.replace({'BrkComm': 1,
                                                          'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                          'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3,
                                                          'HdBoard': 3,
                                                          'BrkFace': 4, 'Plywood': 4,
                                                          'VinylSd': 5,
                                                          'CemntBd': 6,
                                                          'Stone': 7, 'ImStucc': 7})

    dataset['MasVnrType'] = dataset.MasVnrType.replace({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    dataset['ExterQual'] = dataset.ExterQual.replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    dataset['Foundation'] = dataset.Foundation.replace({'Slab': 1,
                                                        'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                                        'Wood': 3, 'PConc': 4})

    dataset['BsmtQual'] = dataset.BsmtQual.replace({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    dataset['BsmtExposure'] = dataset.BsmtExposure.replace({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    dataset['Heating'] = dataset.Heating.replace({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    dataset['HeatingQC'] = dataset.HeatingQC.replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    dataset['KitchenQual'] = dataset.KitchenQual.replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    dataset['Functional'] = dataset.Functional.replace(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    dataset['FireplaceQu'] = dataset.FireplaceQu.replace({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    dataset['GarageType'] = dataset.GarageType.replace({'CarPort': 1, 'None': 1,
                                                        'Detchd': 2,
                                                        '2Types': 3, 'Basment': 3,
                                                        'Attchd': 4, 'BuiltIn': 5})
    dataset['GarageFinish'] = dataset.GarageFinish.replace({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    dataset['PavedDrive'] = dataset.PavedDrive.replace({'N': 1, 'P': 2, 'Y': 3})

    dataset['SaleType'] = dataset.SaleType.replace({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                                    'CWD': 2, 'Con': 3, 'New': 3})

    dataset['SaleCondition'] = dataset.SaleCondition.replace({'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2,
                                                              'Family': 2, 'Normal': 3, 'Partial': 4})

    return dataset

def transfer_encoder(dataset):
    """
    将部分类别特征转换成序列值
    :param dataset:
    :return:
    """
    lab=LabelEncoder()

    str_list=['BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'MoSold',
    'YrSold', 'YearBuilt', 'YearRemodAdd', 'LowQualFinSF', 'GarageYrBlt']
    for col in str_list:
        dataset[col]=lab.fit_transform(dataset[col])
    return dataset

def get_dummies(dataset):
    """
    将一些类别特征转换成稀疏数值型特征
    :param dataset:
    :return:
    """
    dataset=pd.get_dummies(dataset)

    return dataset

def scale_feature(dataset):
    """
    将特征值标准化
    :param dataset:
    :return:
    """
    scaler = RobustScaler()
    train,test=loadset()
    train=dropOutliers(train)
    n_train=train.shape[0]

    X_train=dataset[:n_train]
    X_test=dataset[n_train:]
    Y_train=train['SalePrice']

    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.fit_transform(X_test)
    y_train_log=np.log(Y_train)

    return X_train_scaled,X_test_scaled,y_train_log

def lasso_select(dataset):
    """
    use lasso 来做特征选择
    :param dataset:
    :return:
    """
    X_train_scaled,X_test_scaled,y_train_log=scale_feature(dataset)

    lasso=LassoCV(alphas=[1, 0.1, 0.001, 0.0005])
    lasso.fit(X_train_scaled,y_train_log)
    L1_lasso = pd.DataFrame({'Importance': lasso.coef_}, index=dataset.columns)
    dataset_feature=L1_lasso[L1_lasso['Importance'] != 0].index

    X_train=pd.DataFrame(X_train_scaled,columns=dataset.columns)[dataset_feature]
    X_test=pd.DataFrame(X_test_scaled,columns=dataset.columns)[dataset_feature]
    Y_train=y_train_log

    return X_train, X_test,Y_train

class stacking(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_model, meta_model):
        self.base_model =base_model  # 初级学习器的集合，初级学习器作用于初始训练集和初始测试集
        self.meta_model = meta_model  # 次级学习器，用于对初级学习器的输出进行预测
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)  # K折交叉验证，目的初始训练器不经产生的训练集会过拟合

    def fit(self, X, y):
        """
        采用多个初级训练器，对训练集进行交叉验证，得到次级训练器的训练集
        :param X:
        :param y:
        :return:
        """
        self.base_model_ = [[] for i in self.base_model]  # 创建长度为model个数的空list，用于保存每次交叉验证的模型
        self.meta_model_ = clone(self.meta_model)
        blend_train = np.zeros((X.shape[0], len(self.base_model)))  # 创建为0的数组，用于存储K折交叉验证后预测的结果

        for i, model in enumerate(self.base_model):
            for train_index, val_index in self.kf.split(X, y):
                clone_model = clone(model)  # 由于要进行K次训练，每次训练的训练集都不一样，所以克隆一份初始的model
                clone_model.fit(X[train_index], y[train_index])
                self.base_model_[i].append(clone_model)  # 对每次交叉验证时的模型进行保存，用于后续对测试集进行训练
                blend_train[val_index, i] = clone_model.predict(X[val_index])

        self.meta_model_.fit(blend_train, y)
        return self

    def predict(self, X_test):
        """
         用K折交叉验证中每一折的模型对测试集进行预测，并对同类模型的预测结果进行平均，
        将不同模型平均后的结果构成新的测试集用于预测
         :param X_test:
         :return:
        """
        blend_test = np.column_stack([np.column_stack([model.predict(X_test) for model in each_model]).mean(axis=1)
                                      for each_model in self.base_model_])

        return self.meta_model_.predict(blend_test)

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

def main():

    train, test = loadset()  # 加载数据集
    train=dropOutliers(train)   #删除训练集中的异常点

    dataset=concatSet(train, test)    #合并训练集和测试集
    dataset=firstdropfeature(dataset)    #初次特征探索后，删除的特征
    dataset=fillmissing(dataset)       #填充缺失值
    dataset=addmorefeat(dataset)       #添加挖掘出的特征
    dataset=tranfer2str(dataset)       #将部分数值型特征转换成类别型
    dataset=skew_dummiestransform(dataset)  #对数据的偏度进行处理
    dataset=replace_process(dataset)
    dataset=transfer_encoder(dataset)
    dataset=get_dummies(dataset)


    X_train, X_test, Y_train=lasso_select(dataset)  #lasso特征选择

    lasso = Lasso(alpha=0.0001, max_iter=10000)
    ridge = Ridge(alpha=25)
    kernelridge = KernelRidge(alpha=20, kernel='polynomial', degree=3, coef0=7.0)
    elesticnet = ElasticNet(alpha=0.01, l1_ratio=0.001, max_iter=10000)
    svr = SVR(C=19, kernel='rbf', gamma=0.0004, epsilon=0.009)

    stack_model = stacking([lasso, ridge, kernelridge, elesticnet, svr], kernelridge)
    stack_model.fit(X_train.values, Y_train.values)
    rmse_cv(stack_model, X_train.values, Y_train.values).mean()

    Predictions = stack_model.predict(X_test.values)

    submission = pd.DataFrame({'Id_test': test['Id'], 'Predicions': Predictions})
    submission.to_csv('housing_price.csv', index=False)

main()