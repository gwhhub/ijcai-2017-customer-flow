import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import datetime
import matplotlib
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 6


def getWeekendsData(dataSet, startTime, endTime):

    dates = list(pd.date_range(startTime, endTime))
    weekends = []
    data = []
    for day in dates:
        # 判断日期是星期几
        weekday = day.strftime("%w")
        if weekday == '6' or weekday == '0':
            try:
                data.append((dataSet.loc[day.strftime('%Y-%m-%d')])['flow'])
                weekends.append(day)

            except:
                # print ('' + day.strftime('%Y-%m-%d') + ' has no value!')
                # data.append(0)
                continue
            
    weekendsData = pd.DataFrame({
            'date':weekends,
            'flow':data
        })
        
    return weekendsData



def getFestivalsData(dataSet, startTime, endTime):


    dates = []
    data = []
    # 元旦
    newYearDays = pd.date_range('2016-01-01','2016-01-03')
    chineseNewYearDays = pd.date_range('2016-02-07','2016-02-13')
    chingMingFestival = pd.date_range('2016-04-03','2016-04-05')
    internationalLaborDay = pd.date_range('2016-05-01','2016-05-03')
    dragonBoatFestival = pd.date_range('2016-06-09','2016-06-11')
    midAutumnFestival = pd.date_range('2016-09-15','2016-09-17')
    nationalDay = pd.date_range('2016-10-01','2016-10-07')
    christmas = pd.date_range('2016-12-24','2016-12-25')

    lastChristmas = pd.date_range('2015-12-24','2015-12-25')
    lastNationalDay = pd.date_range('2015-10-01','2015-10-07')


    festivals = [lastChristmas, lastNationalDay, newYearDays,chineseNewYearDays,chingMingFestival,internationalLaborDay,dragonBoatFestival,
    midAutumnFestival,nationalDay,christmas]

    for festival in festivals:
        for day in festival:
            try:
                data.append((dataSet.loc[day.strftime('%Y-%m-%d')])['flow'])
                dates.append(day)

            except:
                # print ('' + day.strftime('%Y-%m-%d') + ' has no value!')
                # data.append(0)
                continue

    festivalsData = pd.DataFrame({
            'date':dates,
            'flow':data
        })
        
    return festivalsData

def getRainyData(dataSet, startTime, endTime):

    dates = list(pd.date_range(startTime, endTime))
    rainyDate = []
    data = []
    for day in dates:
        try:
            date = day.strftime('%Y-%m-%d')
            weather = ((dataSet.loc[date])['weather_1'])

            if '中雨' in weather:
                data.append( ((dataSet.loc[date])['flow']) )
                rainyDate.append(day)

        except:
            # print ('' + day.strftime('%Y-%m-%d') + ' has no value!')
            # data.append(0)
            continue
            
    rainyData = pd.DataFrame({
            'date':rainyDate,
            'flow':data
        })
        
    return rainyData

# '/Users/wiwi/Desktop/IJCAI-17 口碑商家客流量预测/dataPreProcessed/user_pay_countbyday.csv' 
# with weekends lables
def visualData(dataSet, storesId, startTime, endTime, folderName, withTest = False):

    
    #定义自定义字体，文件名从1.b查看系统中文字体中来  
    myfont = matplotlib.font_manager.FontProperties(fname=
                            '/Users/wiwi/anaconda/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf')  
    #解决负号'-'显示为方块的问题  
    rcParams['axes.unicode_minus']=False  
    rcParams['figure.figsize'] = 25, 10 


    # dataSet = pd.read_csv('../dataPreProcessed/user_pay_countbyday.csv',
    #                   parse_dates=True, index_col=1)
    dataSet = dataSet[startTime:endTime]
    for storeId in storesId:
        tempDataSet = dataSet[dataSet['store_id']==storeId]
        
        weekendsData = getWeekendsData(tempDataSet, startTime, endTime)
        festivalsData = getFestivalsData(tempDataSet, startTime, endTime)
        rainyData = getRainyData(tempDataSet, startTime, endTime)
        # print (rainyData)
        
        # 离群点排除
        q75, q25 = np.percentile(tempDataSet['flow'], [75 ,25])
        iqr = q75-q25

        plt.plot(tempDataSet.index, tempDataSet['flow'],
                 color='green', linestyle='dashed', marker='o',markerfacecolor='black', markersize=10, 
                 label='' + startTime + ': ' + endTime + ' flow')
        plt.plot(weekendsData['date'], weekendsData['flow'], 'r^', label='weekends')
        plt.plot(festivalsData['date'], festivalsData['flow'] + 3 , 'y^', label='festivals')
        plt.plot(rainyData['date'], rainyData['flow'] + 7 , 'bo', markersize=8 , label='rainys')
        if withTest:
            pre = pd.read_csv('../eval/提交/upoad18_0301_1142.csv',index_col=0,header=None)
            pre = pre[pre.index == storeId]
            # print (pre)
            plt.plot(pd.date_range('2016-11-01','2016-11-14'), pre.T.values, 'o--', color='darkgreen', label='predict')
            plt.xlim(datetime.datetime.strptime(startTime,'%Y-%m-%d')-datetime.timedelta(days=3)
                 , datetime.datetime.strptime('2016-11-16','%Y-%m-%d'))
        else:
            plt.xlim(datetime.datetime.strptime(startTime,'%Y-%m-%d')-datetime.timedelta(days=3)
                 , datetime.datetime.strptime(endTime,'%Y-%m-%d')+datetime.timedelta(days=3))
        
        # plt.ylim((q75-1.5*iqr)*0.8, 
        #          (q25+1.5*iqr)*1.1)
        plt.legend(loc='best') # 显示右上角的图例
        plt.title('商家：' + str(tempDataSet['store_id'][0]) + ' , ' + tempDataSet['city'][0] + ', ' + str(tempDataSet['category_1'][0]) + 
                  ' ' + str(tempDataSet['category_2'][0]) + ' ' + str(tempDataSet['category_3'][0]) + ', 人均消费：' + 
                  str(tempDataSet['average_pay'][0]) + ', ' + '评分：' + str(tempDataSet['comment_level'][0]) + ', ' + 
                  '评论数：' + str(tempDataSet['comment_num'][0]) + ', ' + '门店等级：' + str(tempDataSet['shop_level'][0])
                  ,fontproperties=myfont)
        plt.grid(True) # 显示网格  datetime.timedelta

        # mkdir foldername
        directory = folderName 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # "20151001_20160930_pic/"
        # "20160301_20160930_pic/"

        
        plt.savefig(directory + str(storeId) +".png", format='png')
        # plt.show()
        plt.clf()

