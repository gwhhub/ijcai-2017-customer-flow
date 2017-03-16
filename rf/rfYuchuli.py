
import sys
sys.path.append("..")
from tools import *

###
# 排除放在模型中不适合的商家，或者说用随机森林模型不能很好的预测
###
def paichuShop(pay):
	index10 = [] 

	index10_bad = []

	# pay = pd.read_csv('../../yuChuli/parsedWithTest.csv',index_col=0)
	pay.index = pd.to_datetime(pay['date'])

	for index in np.arange(1,2001,1):
		tmp = pay[pay['store_id']==index]
		tmp10 = (tmp['2016-10-09':'2016-10-31'])
		tmp8910 = tmp['2016-08':'2016-10']

		##
		# 判断的条件为：10月的空值数，10月相对于8，9，10月的客流量中位数差异
		## (len(tmp['2016-10'])<21) or
		panduan =  len(tmp10) <= 20 \
			# ( tmp['flow'].mean(axis=0) > (tmp8910['flow'].mean(axis=0) +10) ) or \
			# ( tmp10['flow'].mean(axis=0) < (tmp8910['flow'].mean(axis=0) -10) ) or \
			# ( tmp10['flow'].mean(axis=0) > (tmp8910['flow'].mean(axis=0) +10) ) 


		if panduan:
			index10_bad.append(index)
			continue
		index10.append(index)

	print (index10_bad)
	# 回传适合随机森林模型的商店
	return index10, index10_bad




	