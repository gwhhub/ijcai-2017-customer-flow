from utility import *
rcParams['figure.figsize'] = 20, 6
import sys

# supervised 用于先下做测试
# data要求索引是
# data.index = data['日期']
def lmm(data, stores, trainStart, trainEnd, testStart, testEnd, order=1, evalu=True, folderName='tmp',supervised=True):
    
    weather = pd.read_csv('../dataSet/weatherFixed_lessCategory.csv',index_col=False)
    
    
    tag = '' + trainStart + '_' + trainEnd + '_' + testStart + '_' + testEnd

    for index in stores:
        
        directory = folderName +str(index) + '/'
        
        # 填空值后的时间序列
        # temp = fillNAperiod([index],data,trainStart,testEnd)
        temp = data
        temp.loc[:,'日期'] = pd.to_datetime(temp.loc[:,'日期'])
        temp.index = temp.loc[:,'日期']
        f = temp[trainStart:trainEnd]
        tt = f.loc[:,'流量']
        
        # 原来的时间序列
        tmpOriginal = data[data.loc[:,'商家id']==index]

        try:
            tmpOriginal1 = tmpOriginal[trainStart:trainEnd]
            tt1 = tmpOriginal1.loc[:,'流量']
            count = len(pd.date_range(trainStart,trainEnd)) - len(tt1) 
            if count > 0:
                print ('商家' + str(index) + '在' + trainStart + '到' + trainEnd + '空值的个数：'  + str(count) + '，已填空')
        except:
            print('起始时间10-08商家没有营业。')
        
        # 历史天气数据
        # histWeather = tmpOriginal[tmpOriginal.index.values[0]:tmpOriginal.index.values[len(tmpOriginal.index)-1]]
        histWeek = tmpOriginal[tmpOriginal.index.values[0]:tmpOriginal.index.values[len(tmpOriginal.index)-1]]
    
        # 十一月天氣數據
        weather_area = weather[weather['area']==tmpOriginal['市名'][0]]
        weather_area.index = pd.to_datetime(weather_area['date'])
        weather_11 = weather_area['2016-11-01':'2016-11-14']
        
        
        # 确认没有空值
        if len(tt[np.isnan(tt)]) > 0:
            print ('怎么还是有空值。。？ ' + len(tt1[np.isnan(tt1)]))
            continue
        else:
            # 创建目录
            if not os.path.exists(directory):
                os.makedirs(directory)

            ################
            ### trainSet ###
            ################  
            rol_mean_7 = pd.ewma(histWeek.loc[:,'流量'], span=7)
            f.loc[:,'weekday'] = f.loc[:,'日期'].apply(lambda x: x.isoweekday())
            histWeek['weekday'] = histWeek.loc[:,'日期'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').isoweekday())
            df = f[['weekday','商家id','市名','流量','天气']]
            

            # 星期指数
            histWeek.loc[:,'rol_mean_7'] = rol_mean_7.values
            histWeek.loc[:,'星期指数_随机'] = histWeek['流量']/histWeek['rol_mean_7']
            tmp_week = histWeek.iloc[(7-1):,:]
            para_week = dict(tmp_week['星期指数_随机'].groupby(tmp_week['weekday']).median())
            
            # 星期指数填入
            Set_week = []
            for day in range(1,8):
                
                train = df[df['weekday']==day]
                train.loc[:,'星期指数'] = [para_week[(day)]] * len(train)
                Set_week.append(train)                
            trainS = pd.concat(Set_week)
            trainS = trainS.sort_index()

            # 天气指数是在消除星期因素下计算
            trainS.loc[:,'消除星期_因素流量值'] = trainS['流量'] / (trainS['星期指数'])
            rol_mean_3 = pd.ewma(trainS.loc[:,'消除星期_因素流量值'], span=3)
            trainS.loc[:,'rol_mean_3'] = rol_mean_3.values
            trainS['天氣指数_随机'] = trainS['消除星期_因素流量值'] / (trainS['rol_mean_3'])
            tmp_weather = trainS.iloc[(2-1):,:]
            para_weather = dict(tmp_weather.loc[:,'天氣指数_随机'].groupby(tmp_weather.loc[:,'天气']).median())
            
            Set_weather = []
            for ty in para_weather.keys():
                trainSe = trainS[trainS['天气']==ty]
                trainSe['天氣指数'] = [para_weather[ty]] * len(trainSe)
                Set_weather.append(trainSe)
                
            trainSet = pd.concat(Set_weather)
            trainSet['消除星期_天氣因素流量值'] = trainSet['流量'] / (trainSet['天氣指数']*trainSet['星期指数'])
             
            ### 生成预测值， 拟合函数， 规则 ### 
            x = np.array(np.arange(0,len(trainSet),1))
            y = np.array(trainSet['消除星期_天氣因素流量值'])
            fit = np.polyfit(x, y, order)
            fit_fn = np.poly1d(fit)
            trainSet['拟合预测流量值'] = fit_fn(x)
            trainSet['星期_天氣因素_预测流量值'] = trainSet['拟合预测流量值']*trainSet['星期指数']*trainSet['天氣指数']
            trainSet = trainSet.sort_index()
            

            ###############
            ### testSet ###
            ###############
            p = temp[testStart:testEnd]
            p.loc[:,'weekday'] = p.loc[:,'日期'].apply(lambda x: x.isoweekday())
            p['天气'] = weather_11['天气fixed']
            p = p[['weekday','商家id','市名','流量','天气']]
            # print (weather_11['天气fixed'])

            Set_week = []
            for day in range(1,8):
                test = p[p['weekday']==day]
                test.loc[:,'星期指数'] = [para_week[(day)]] * len(test)
                Set_week.append(test)                
            testS = pd.concat(Set_week)
            testS.loc[:,'天氣指数']=None
            for date in pd.date_range('2016-11-01','2016-11-14'):
                
                if testS.loc[date,'天气'] in para_weather.keys():
                    testS.loc[date,'天氣指数'] = para_weather[testS.loc[date,'天气']]
                else:
                    testS.loc[date,'天氣指数'] = 1
           
            testS = testS.sort_index()
            

            # 合并
            d_p = pd.concat([trainSet,testS])
            d_p.loc['2016-11-01':'2016-11-14','流量'] = None
            d_p.loc[:,'市名'] = d_p.loc[:,'市名'].fillna(method='ffill')

            ############
            # 加入预测值 #
            ############
            # predeict = fit_fn(np.arange(92,122,1))
            predeict = fit_fn(np.arange(len(trainSet),len(trainSet)+7,1))  # 注意
            start = pd.to_datetime(testStart,format='%Y-%m-%d')
            end =  pd.to_datetime(testEnd,format='%Y-%m-%d')
            d_p.index = pd.to_datetime(d_p.index)
            # 预测7天
            d_p.ix[start:end,'拟合预测流量值'] = list(predeict) + list(predeict) 
            d_p.ix[start:end,'星期_天氣因素_预测流量值'] = d_p.loc[:,'拟合预测流量值'] * d_p.loc[:,'星期指数'] * d_p.loc[:,'天氣指数']

            # 输出/作图
            d_p.to_csv(directory + '/' + tag + '_store' + str(index) + '_order' + str(order) + '.csv')
            plt.plot(d_p.index, d_p.loc[:,'星期_天氣因素_预测流量值'], 'ro--', label='predictFlow')
            
            if supervised:
                plt.plot(d_p.index, d_p.loc[:,'流量'], 'bo--', label='realFlow')
            plt.grid(True) 
            plt.title('' + str(index) + '_' + trainStart + '_' + trainEnd + '_' + testStart + '_' + testEnd) 
            plt.legend(loc='best')
            plt.savefig(directory + '/' + tag + '_store' + str(index) + '_order' + str(order) +".png", format='png')
            plt.clf()
            
            # 评测
            if evalu:
                d_pEvalu = d_p[testStart:testEnd]
                bigPool_value, mae_, rmse_, mape_ = evalu1(d_pEvalu.loc[:,'流量'], d_pEvalu.loc[:,'星期因素_预测流量值'],1)
                wucha  = open(directory + tag +'_error_order' + str(order) + '.txt', 'w') 

                wucha.write('' + str(bigPool_value) + ',' + str(mae_) + ',' +  str(rmse_) + ',' + str(mape_))
                wucha.close()


# 将资料资料架的各商家答案合成一个dataframe
def concatAns(folderName,fileName):
    ## import os
    files = [f for f in os.listdir(folderName)]
    files.remove('.DS_Store')
    indexes = []
    for ele in files:
        indexes.append(ele)
    
    dslfixd = []
    for index in indexes:
        files = [f for f in os.listdir(folderName + index+'/')]
        for f in files:
    #         print (f)
            if f.split('.')[1] == 'csv':
                tmp = pd.read_csv(folderName + index +'/' + f, parse_dates=True,index_col=0)
                tmp1 = tmp['2016-11-01':'2016-11-14'].copy()
                if tmp1['星期_天氣因素_预测流量值'].min() < 0:
                    print ('流量有负数' + str(tmp1['商家id'][2]))
                    tmp2= pd.concat([tmp['2016-10-25':'2016-10-31'].copy(),tmp['2016-10-25':'2016-10-31'].copy()])
                    if tmp2['星期_天氣因素_预测流量值'].min()<0:
                        print ('流量有负数' + str(tmp2['商家id'][2]))
                    tempResult2 = tmp2[['商家id','流量']]
                    # 拟合出来有负数值，填写最近一礼拜的均值
                    tempResult2['流量'] = [tempResult2['流量'].mean()] * len(tempResult2)
                    tempResult2.columns = ['商家id','预测值']
                    dslfixd.append(tempResult2)
                else:
                    tempResult1 = tmp1[['商家id','星期_天氣因素_预测流量值']]
                    tempResult1.columns = ['商家id','预测值']
                    dslfixd.append(tempResult1)
                    
    dsfixd = pd.concat(dslfixd)  
    dsfixd.to_csv('../lm/' + fileName)
    return True


if __name__ == '__main__':


    folderName = '../lm/tmp/'
    toLm = True
    toConcat = False

    if toLm:
        # 指定要丢到规则的商家
        specific = []
        for i in range(1, len(sys.argv)):
            specific.append(int(sys.argv[i].split(',')[0]))

        # withWeather3.csv
        data = pd.read_csv('../yuChuli/历史数据筛选/1/history.csv',index_col=0,parse_dates=True)
        # data.index = data['日期']
        # from storeClassify import  checkNullNumbers, availbleStoresId
        # handNeedIndex = checkNullNumbers(data, '2016-10-08','2016-10-31')
        # handNeedless = availbleStoresId(handNeedIndex)

        trainStart = '2016-10-08'
        trainEnd = '2016-10-31'
        testStart = '2016-11-01'
        testEnd = '2016-11-14'


        trainStart1 = '2016-09-01'
        trainEnd1 = '2016-10-31'
        testStart1 = '2016-11-01'
        testEnd1 = '2016-11-14'

        trainStart2 = data.index.values[0]
        trainEnd2 = data.index.values[len(data)-1]
        testStart2 = '2016-11-01'
        testEnd2 = '2016-11-14'

        # 填写要处理的序列 handNeedless or specific
        # storesId = np.arange(1,2001,1)

        lmm(data, specific, str(trainStart2),str(trainEnd2),testStart1,testEnd1,order=1,evalu=False,folderName=folderName)

    # 结果合成
    if toConcat:
        filename = '1008_1031_handNeedless.csv'
        filename = 'tmp.csv'
        if concatAns(folderName, fileame):
            print ('Result concat succeed!')











