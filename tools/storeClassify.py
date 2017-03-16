import pandas as pd
import numpy as np

# handNeedless
# handNeed
# 输出某一时间段商家流量的空值个数
def checkNullNumbers(data, startTime, endTime):

	count = 0
	handNeedIndex = []
	for index in np.arange(1,2001,1):
	    tmp = data[data['商家id']==index].copy()
	    tmp.index = pd.to_datetime(tmp['日期'])
	    tmp1 = tmp[startTime:endTime]
	    
	 
	    if len(tmp1) < len(pd.date_range(startTime,endTime)) - 2:
	#         print (tmp)
	        count = count + 1
	        handNeedIndex.append(index)
	        # print (index, len(tmp1),len(tmp2))
	        
	return handNeedIndex

# 反向选择
def availbleStoresId(handNeedIndex):
	handNeedless = [] 
	for ele in np.arange(1,2001,1):
	    if ele not in handNeedIndex:
	        handNeedless.append(ele)

	return handNeedless


if __name__ == '__main__':

	data = pd.read_csv('../yuChuli/withWeather1.csv')
	handNeedIndex = checkNullNumbers(data, '2016-10-08','2016-10-31')
	print (handNeedIndex)
	data.index = pd.to_datetime(data['日期'])
	for index in handNeedIndex:
		tmp = data[data['商家id']==index].copy()
		tmp = tmp['2016-09-01':'2016-09-30']

		print (''+ str(index) + ',' + str(len(tmp[tmp['商家id']==index])), end = ' ')
	
