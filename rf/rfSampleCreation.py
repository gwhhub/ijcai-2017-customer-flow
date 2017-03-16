import os
import sys
sys.path.append("..")
from tools import *
from rfYuchuli import *

rcParams['figure.figsize'] = 20, 6
from datetime import datetime, timedelta

# 提取7，8，9，10月的样本
def sampleCreation():

    # 排除商家
    index10 = paichuShop()  

    ds = pd.read_csv('../../yuChuli/parsedFlowTranspose.csv',index_col=0,parse_dates=True)
    ds.columns = pd.to_datetime(ds.columns)
    ds = ds[pd.date_range('2016-07-01','2016-10-31')]
    print (ds)

    # 选中所有的礼拜二
    # 样本构建
    temp = np.zeros([26851,22])
    count=0
    tuesdayDates = []
    for tuesday in ds.columns:
        for index in index10:
            # 选中礼拜二
            if tuesday.isoweekday() == 2:
                tmp = ds.loc[index,tuesday:tuesday + timedelta(20)]
                panduan = ( (pd.isnull(tmp).sum().sum()) < 2 ) and (len(tmp[tmp.values == 0]) < 2 ) and ('2016-11-01' not in tmp.index ) and ('2015-12-11' not in tmp.index) and (len(tmp)==21)

                if  panduan: # 某行dataframe中的空值个数
                    temp[count, 0] = index
                    temp[count, 1:15] = tmp[tuesday:tuesday + timedelta(13)]
                    tuesdayDates.append(tuesday + timedelta(14))
                    temp[count, 15:] = tmp[tuesday + timedelta(14) : tuesday + timedelta(21)]
                    count = count +1
    sampl =  pd.DataFrame(temp)

    # print (sampl)
    # print (count)

    # sampl_x, sampl_y 分开
    sampl_x = sampl.loc[:,1:14]
    sampl_dates = pd.DataFrame({'14': tuesdayDates})
    sampl_y = sampl.loc[:,15:]

    # 预处理 
    # 排除异常：把NaN用当周的中位数 Imputer
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    imp.fit(sampl_x.loc[:,:])
    nullRemoval = imp.transform(sampl_x.loc[:,:])
    sample_x = pd.DataFrame(nullRemoval)
    sample_x.index = sampl[0]
    # sample_x.to_csv('../../yuChuli/tmp.csv')

    	
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    imp.fit(sampl_y.loc[:,:])
    nullRemoval = imp.transform(sampl_y.loc[:,:])
    sample_y = pd.DataFrame(nullRemoval)

    # 均值，中位数特征
    # feature creation
    sample_x_mean = sample_x.mean(axis=1)
    sample_x_median = sample_x.median(axis=1)

    # 交叉特征
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2,interaction_only=True,include_bias=False)
    jiaocha = poly.fit_transform(sample_x)
    jiaochaFatures = pd.DataFrame(jiaocha)
    jiaochaFatures['store_id'] = sample_x.index
    jiaochaFatures['mean'] = sample_x_mean.values
    jiaochaFatures['median'] = sample_x_median.values
    jiaochaFatures['dates'] = sampl_dates
    
    # print (jiaochaFatures)
    flowSamples = pd.concat([jiaochaFatures,sample_y],axis=1)
    print (flowSamples)

    # 商家资讯特征
    pay = pd.read_csv('../../yuChuli/parsedWithTest.csv',index_col=0)
    inf = pay[['location_id','category_1', 'store_id', 'shop_level','average_pay','comment_level','comment_num']]
    inf = inf.drop_duplicates(['store_id'])
    
    info = pd.get_dummies(inf, columns=['location_id', 'category_1',  'shop_level', 'comment_level'])

    samples = pd.merge(left=info, right= flowSamples, right_on = 'store_id', left_on='store_id')


    # samples = pd.concat([samples_x ,sample_y],axis=1)
    samples.to_csv('../../yuChuli/TuesdaySamples.csv')

# 生成线上的test_x
def sampleTest_xCreate():

     # 排除商家
    index10 = paichuShop()

    pay = pd.read_csv('../../yuChuli/parsedWithTest.csv',index_col=0)
    pay.index = pd.to_datetime(pay['date'])
    dsl = []
    for index in index10:
        tmp = (pay[pay['store_id']==index]).copy()
        tmp_t = tmp.loc['2016-10-18':'2016-10-31',:]
        tmp_t = tmp_t[['flow']].T
        tmp_t.index = [index]
        dsl.append(tmp_t)

    ds = pd.concat(dsl)
    ds.columns = pd.to_datetime(ds.columns)

    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    imp.fit(ds.loc[:,:])
    nullRemoval = imp.transform(ds)
    test_x = pd.DataFrame(nullRemoval)
    test_x.index = ds.index

    # 均值，中位数特征
    # feature creation
    test_x_mean = test_x.mean(axis=1)
    test_x_median = test_x.median(axis=1)

    # 交叉特征
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2,interaction_only=True,include_bias=False)
    jiaocha = poly.fit_transform(test_x)
    jiaochaFatures = pd.DataFrame(jiaocha)
    jiaochaFatures['store_id'] = test_x.index
    jiaochaFatures['mean'] = test_x_mean.values
    jiaochaFatures['median'] = test_x_median.values
    jiaochaFatures.index = test_x.index
    print (jiaochaFatures)
    # jiaochaFatures['dates'] = sampl_dates

    # 商家资讯特征
    inf = pay[['location_id','category_1', 'store_id', 'shop_level','average_pay','comment_level','comment_num']]
    inf = inf.drop_duplicates(['store_id'])
    info = pd.get_dummies(inf, columns=['location_id', 'category_1',  'shop_level', 'comment_level'])

    test_x = pd.merge(left=info, right= jiaochaFatures, right_on='store_id', left_on='store_id')


    test_x.to_csv('../../yuChuli/tuesdaySamples_test_x1.csv')


if __name__ == '__main__':


    sampleTest_xCreate()





