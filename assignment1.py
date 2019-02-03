from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y_train = train.pop('SalePrice')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), axis=0)
miss_data = all_data.isnull().sum().sort_values(ascending=False)

remove_list = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']
for i in remove_list:
    all_data.pop(i)

col_mean = all_data.mean()
all_data = all_data.fillna(col_mean)
all_dummy = pd.get_dummies(all_data)
x_train = all_dummy.iloc[:1460]
x_test = all_dummy.iloc[1460:]


rig = Ridge(alpha = 10)
rig.fit(x_train,y_train)
y_predict = rig.predict(x_test)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_predict})
submission.to_csv('submission.csv',index=False)