import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 6

'''将多维数组转换为DataFrame'''
'''github 上大神开源 '''
def transfrom_Arr_DF(arr,col_name = 'col_'):
    if(len(arr.shape)==1):
        df = DataFrame(arr,columns=['col_0'])
    else:
        df = DataFrame(arr,columns=[col_name+str(i) for i in range(arr.shape[1])])
    return df
    
# 评测
# DataRange should be 2000 * 14 
def evaluSeries(realFlow, predictFlow, storeCount):


    real_DataRange = len(realFlow)
    predict_DataRange = len(predictFlow)
    if real_DataRange == predict_DataRange:
        print ('评测开始: ' + str(predict_DataRange))

        # 确认没有空值
      
        store_num = storeCount
        timeRange = 14
        diff = realFlow - predictFlow
        summ = realFlow + predictFlow
        
        bigPool_value = np.sum(np.abs((diff/summ)))/(store_num * timeRange)
        abs_ = np.abs(diff)
        mae_ = abs_.mean() # 平均绝对误差
        rmse_ = ((abs_**2).mean())**0.5 # 均方根误差      
        mape_ = (abs_/realFlow).mean() # 平均绝对百分误差

        return bigPool_value, mae_, rmse_, mape_ 
    
    else:
        print ('数据长度不同,' + str(real_DataRange) + ', '+ str(predict_DataRange))
        return None


def concatAns(fromPath, toPath):

    files = [f for f in os.listdir(fromPath)]
    files.remove('.DS_Store')
    indexes = []
    for ele in files:
        indexes.append(ele)
    
    dslfixd = []

    for index in indexes:
        files = [f for f in os.listdir(fromPath+index+'/')]

        fetched = False
        for f in files:
        	
        	if f.split('.')[1] == 'csv' and not fetched:
        		tmp = pd.read_csv(fromPath + index +'/' + f, parse_dates=True)
        		tmp.index = tmp['date']
        		tmp = tmp['2016-11-01':'2016-11-14']
        		tmp['store_id'] = [index] * len(tmp)
        		tempResult1 = tmp[['store_id','flow']]
        		tempResult1.columns = ['store_id','flow']
        		dslfixd.append(tempResult1)

        		# if len(tempResult1) > 14 :
        		# 	print (index)

        		fetched = True
	        elif f.split('.')[1] == 'csv':
	        	print (index)

    dsfixd = pd.concat(dslfixd) 
    dsfixd.to_csv(toPath)

    return dsfixd



def modifyIndex(data, stores, start, end):
	
    dl=[]
    for store in stores:
        tmp = data[data['store_id']==str(store)]
        tmp.index = pd.date_range(start,end)
        dl.append(tmp)
    ds = pd.concat(dl)
    return ds


def tijiao(data, fileName):

    # 格式确认
    if len(data) != 28000:
        print ('太可怕了，数据长度不对！', end=',')
        print (len(data))
        return False

    if data['flow'].min() < 0:
        print ('预测值存在负数')
        return False

    # try:
    #     dsfixd = modifyIndex(data, np.arange(1,2001,1), '2016-11-01', '2016-11-14')
    # except:
    #     print ('Modifying index wrong！')
    #     return False


    data_listTijiao = []
    data['flow']= round(data['flow']).apply(int)
    for index in np.arange(1,2001,1):
        
        temp = data[data['store_id']==index]

        temp = temp[['flow']].T
        temp.index = [index]
        data_listTijiao.append(temp)
        
    # print (data_listTijiao)
    tijiao = pd.concat(data_listTijiao)
    tijiao.to_csv(fileName, index_label=None, header=None)
    return True


if __name__ == '__main__':

	fromFileFloder = '../predictData/lr/'
	toFilePath = '../predictData/'
	toFileName = 'lr_Loss_0227.csv'
	tfn = toFilePath + toFileName

	tijiaoPath = '../eval/提交/'
	tijiaoFileName = 'upload18.csv'
	tj = tijiaoPath + tijiaoFileName
	

	# if tijiao(concatAns(fromFileFloder, tfn), tj):

	# 	print (" " + tj + "generated!")
	data = pd.read_csv('../predictData/lr_Loss_0227.csv', parse_dates=True)
	data.index = pd.to_datetime(data['date'])
	tijiao(data, tj)


