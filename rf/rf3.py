
# 随机僧林
from sklearn.externals import joblib
import pandas as pd
import numpy as np

test_x = pd.read_csv('../../yuChuli/tuesdaySamples_test_x.csv',index_col=0)

test_x == test_x.replace('NaN',0)
del test_x['store_id']

rf = joblib.load('../../models/RF/model/RF.pkl')
print(rf)

pre = rf.predict(test_x)
print (pre)