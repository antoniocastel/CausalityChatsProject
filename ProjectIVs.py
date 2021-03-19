
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from scipy.special import expit
#matplotlib inline
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from sklearn.metrics.pairwise import manhattan_distances

import statsmodels.api as sm
from scipy import stats
from linearmodels.iv import IV2SLS

import os, glob
import pandas as pd

#merge sessions
path = "/Users/antoniomoshe/OneDrive - Technion/TECHNION PHD/Winter 2020/Casuality/Devouir_4/Sessions"
all_files = glob.glob(os.path.join(path, "sessions*.csv"))
sessions_month= pd.concat([pd.read_csv(f) for f in all_files])
#---
len(sessions_month) #502,033

#calculate average service time per hour
sessions_month.columns
sessions_AcceptedInvo = sessions_month[sessions_month[" outcome"] < 5]
(sessions_AcceptedInvo[" queue_sec"] == 0).sum() #all are known abandonments
sessions_AcceptedInvo = sessions_AcceptedInvo[sessions_AcceptedInvo[" queue_sec"] > 0]

len(sessions_AcceptedInvo) #23383

#get StartDay and StartHour for invitation acceptance time ------
#convert strings into date

sessions_AcceptedInvo[' invitation_submit_date'] =\
    pd.to_datetime(sessions_AcceptedInvo[' invitation_submit_date'],\
                                      format=' %d/%m/%Y %H:%M:%S')

sessions_AcceptedInvo[' Invitation_Acep_Day_of_week']=\
    sessions_AcceptedInvo[' invitation_submit_date'].dt.dayofweek
sessions_AcceptedInvo['Invitation_Acep_Hour']=\
    sessions_AcceptedInvo[' invitation_submit_date'].dt.hour

#Treatment Wait variable ---ֱ
#short wait is the treatment i would expect them to buy
sessions_AcceptedInvo['WaitTreatment'] = np.where(sessions_AcceptedInvo[' queue_sec']< 60, 1, 0)
#----

sessions_AcceptedInvo['Y'] =\
    np.where(sessions_AcceptedInvo[' conversion_time']> 0, 1, 0)


#calculate total service duration

timezero=sessions_AcceptedInvo[" invitation_submit_time"].min()

sessions_AcceptedInvo['dummy'] =\
sessions_AcceptedInvo[" end_time"]- \
    sessions_AcceptedInvo[" chat_start_time"]

#known abandonments have no service time
sessions_AcceptedInvo['Total_Service_Duration'] =\
    np.where(sessions_AcceptedInvo['dummy']>= timezero, 0, sessions_AcceptedInvo['dummy'])
sessions_AcceptedInvo.drop('dummy', inplace=True, axis=1)

#write csv
sessions_AcceptedInvo.to_csv("sessions_AcceptedInvo.csv")


#using closure time
timemax=sessions_AcceptedInvo[" invitation_submit_time"].max()
#time in seconds
numberofbins=round((timemax-timezero)/3600)

sessions_AcceptedInvo['dummy'] =\
sessions_AcceptedInvo[" end_time"]- \
    sessions_AcceptedInvo[" chat_start_time"]

#known abandonments have no service time
sessions_AcceptedInvo['Total_Service_Duration'] =\
    np.where(sessions_AcceptedInvo['dummy']>= timezero, 0, sessions_AcceptedInvo['dummy'])
sessions_AcceptedInvo.drop('dummy', inplace=True, axis=1)

sessions_AcceptedInvo['Total_Service_Duration'].mean()

( (sessions_AcceptedInvo[' outcome']==1).sum()+(sessions_AcceptedInvo[' outcome']==2).sum()) \
    /  (sessions_AcceptedInvo[' queue_sec'].sum())



#check with linear regression what influences outcome Y -----
#and also is data for other treatments
dataForRegression = pd.read_csv('DataForRegression.csv', index_col=0)

regresion_check = IV2SLS(dataForRegression.Y,\
                         dataForRegression[['queue_sec','invite_type', 'engagement_skill','target_skill','region','city','country','continent','user_os',\
                                             'browser','score','other_time','other_lines','other_number_words',\
                                            'inner_wait', 'visitor_duration',\
                                            'agent_duration', 'visitor_number_words', 'agent_number_words',\
                                        	'visitor_lines', 'agent_lines',	\
                                            'total_canned_lines', 'average_sent', 'min_sent', 'max_sent', 'n_sent_pos', 'n_sent_neg',	'first_sent',\
                                            'last_sent', 'id_rep_code', \
                                            'Invitation_Acep_Day_of_week', 'Invitation_Acep_Hour', \
                                            'NumberofAssigned', 'NumberofAssignedwhenAssigned', \
                                            'Rho_atarrival',\
                                             ]], None, None).fit(cov_type='unadjusted')

print(regresion_check)

regresion_check2 = IV2SLS(dataForRegression.Y,\
                         dataForRegression[['queue_sec','invite_type', 'engagement_skill','target_skill','score','other_time',\
                                            'agent_number_words',\
                                        	'visitor_lines', 'agent_lines',	\
                                             ]], None, None).fit(cov_type='unadjusted')
print(regresion_check2)


#-----


res_first = IV2SLS(dataFirstT.WaitTreatment,dataFirstT.iloc[:, 1:14], None, None).fit(cov_type='unadjusted')



res_first = IV2SLS(dataFirstT.WaitTreatment,dataFirstT.iloc[:, 1:14], None, None).fit(cov_type='unadjusted')

print(res_first)

res_second = IV2SLS(dataFirstT.Y,dataFirstT[['region','city','country','continent','user_os',\
                                             'browser','score','Invitation_Acep_Day_of_week','Invitation_Acep_Hour'\
                                             ]],\
                    dataFirstT.queue_sec*60, dataFirstT[['Rho_atarrival','invite_type',\
                                             'engagement_skill','target_skill'\
                                             ]]).fit(cov_type='unadjusted')

def covariance(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    denominator = len(x)-1
    cov = numerator/denominator
    return cov

covariance(dataFirstT.queue_sec,dataFirstT.Rho_atarrival)
#-0.12

corralation=dataFirstT[['queue_sec','Rho_atarrival','invite_type',\
        'engagement_skill','target_skill']].corr()


#FIRST TREATMENT

#data that we are using after fix in Matlab
dataFirstTi = pd.read_csv('DataForFirstTreatment.csv', index_col=0)
len(dataFirstTi) #23383
#get only service customers
dataFirstT=dataFirstTi

#len(dataFirstT) #13345

dataFirstT.WaitTreatment\
    = np.where(dataFirstT['queue_sec']< 30, 1, 0)

dataFirstT.WaitTreatment\
    = np.where(dataFirstT['queue_sec']< 60, 1, 0)

#dataFirstT['RhoBinary']\
#    = np.where(dataFirstT['Rho_atarrival']< 1, 0, 1)

dFT=dataFirstT[[\
'invite_type','region','city','engagement_skill','target_skill','country','continent','user_os','browser','score','Invitation_Acep_Day_of_week','Invitation_Acep_Hour','Rho_atarrival','WaitTreatment','Y'\
]]


modelP = LogisticRegression(max_iter=10000)
FirstX = dataFirstT[[\
                     'Invitation_Acep_Hour', \
                    'Rho_atarrival', 'invite_type', \
                    'engagement_skill','target_skill',\
                    'Invitation_Acep_Day_of_week',\
                    'score','region','city','country','continent',\
                    'user_os','browser'\
                     ]]
#Wait time
FirstY = dataFirstT.WaitTreatment
modelP.fit(FirstX, FirstY)

#reg2 = LinearRegression().fit(FirstX, FirstY)

First_Propensityscore = np.asarray(modelP.predict_proba(FirstX))
treatment_plt=plt.hist(First_Propensityscore[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt=plt.hist(First_Propensityscore[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_wait_60000.pdf',format='pdf')

#hist, bin_edges = np.histogram(First_Propensityscore[:, 1])

FirstIndex=\
np.where((First_Propensityscore[:, 1] < 0.59)& (First_Propensityscore[:, 1] > 0.41))


First_PropensityscoreB=\
First_Propensityscore[FirstIndex[0]]

treatment_plt1B=plt.hist(First_PropensityscoreB[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt1B=plt.hist(First_PropensityscoreB[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_wait_6000000_Last.pdf',format='pdf')

dataFirstT2=dFT.iloc[FirstIndex[0]]
len(dataFirstT2) #30 sec 10041 #60 sec 6,316
len(dataFirstT) #23383

ATE_IPW=(((dataFirstT2['WaitTreatment']*dataFirstT2['Y'])/(First_PropensityscoreB[:, 1])).sum())*(1/len(dataFirstT2))\
-( ( ( (1-dataFirstT2['WaitTreatment'])*dataFirstT2['Y'] )/(First_PropensityscoreB[:, 0]) ).sum())*(1/len(dataFirstT2))


dataFirstT.groupby("Y")["queue_sec"].mean()
#Out[132]:
#Y
#0    70.728748
#1    48.358257

# ----- T-learner --------
#dataFirstTB = dataFirstT2[[\
#                     'Invitation_Acep_Hour', \
#                    'Rho_atarrival', 'invite_type', \
#                     'engagement_skill', 'target_skill',\
#                     'WaitTreatment', 'Y']]
dataFirstTB=dataFirstT2
dataFirstTB0 = dataFirstTB[dataFirstTB["WaitTreatment"] == 0]
dataFirstTB0 = dataFirstTB0.drop(["WaitTreatment"], axis=1)
dataFirstTB0x = dataFirstTB0.iloc[:, :-1]
dataFirstTB0Y = dataFirstTB0['Y']

dataFirstTBx = dataFirstTB.iloc[:, :-1]
dataFirstTBx = dataFirstTBx.drop(["WaitTreatment"], axis=1)

dataFirstTB1 = dataFirstTB[dataFirstTB["WaitTreatment"] == 1]
dataFirstTB1 = dataFirstTB1.drop(["WaitTreatment"], axis=1)
dataFirstTB1x = dataFirstTB1.iloc[:, :-1]
dataFirstTB1Y = dataFirstTB1['Y']

model_0 = LassoCV()
model_1 = LassoCV()

model_0.fit(dataFirstTB0x, dataFirstTB0Y)
model_1.fit(dataFirstTB1x, dataFirstTB1Y)

prediction0 = model_0.predict(dataFirstTBx)
prediction1 = model_1.predict(dataFirstTBx)

ATE_TLearner = float((prediction1-prediction0).sum()/len(dataFirstTBx))
# ----- T-learner --------

# ------- S-Learner ---------
dataFirstTBX = dataFirstTB.iloc[:, :-1]
dataFirstTBY = dataFirstTB['Y']

modelS = Ridge()
modelS.fit(dataFirstTBX, dataFirstTBY)

#T=1
dummT1 = dataFirstTB[dataFirstTB["WaitTreatment"] == 1]
dummT1x = dataFirstTB.iloc[:, :-1]
predictionUn = modelS.predict(dummT1x)

dummT1["WaitTreatment"] = 0
dummT1x = dummT1.iloc[:, :-1]
predictionZe = modelS.predict(dummT1x)

ATE_SLearner = (predictionUn.sum() / len(dummT1)) - (predictionZe.sum() / len(dummT1))
# ------- S-Learner ---------

# ---------matching---------
#T=0
dummT0 = dataFirstTB[dataFirstTB["WaitTreatment"] == 0]
dummT0 = dummT0.drop(["WaitTreatment"], axis=1)
dummT0x = dummT0.iloc[:, :-1]
dummT0Y = dummT0['Y']

#T=1
dummT1 = dataFirstTB[dataFirstTB["WaitTreatment"] == 1]
dummT1 = dummT1.drop(["WaitTreatment"], axis=1)
dummT1x = dummT1.iloc[:, :-1]
dummT1Y = dummT1['Y']

dummT0V = dummT0.values
dummT1V = dummT1.values

dummT1XV=dummT1x.values
dummT0XV=dummT0x.values


ITE = []
for row in dummT1V:
    dif=abs(row[:-1]-dummT0V[:,:-1])
    #index
    #dif.argmin()
    #value
    distance=row[-1]-dummT0V[dif.argmin(),-1]
    ITE.append(distance)
#Sum of the Treated/number of Treated
ATE_Matchingb = float(sum(ITE))/float(len(dataFirstTB))



# ---------matching---------


#-----------IV's-------------
corralation=dataFirstTB.corr()

res_second = IV2SLS(dataFirstTB.Y,dataFirstTB[['invite_type','engagement_skill','region','city','country','continent','user_os','browser','score','Invitation_Acep_Hour','Invitation_Acep_Day_of_week']],\
                    dataFirstTB.WaitTreatment, dataFirstTB.Rho_atarrival).fit(cov_type='unadjusted')

print(res_second)

covariance(dataFirstT.queue_sec,dataFirstT.Rho_atarrival)
#-0.12

corralation=dataFirstT[['queue_sec','Rho_atarrival','invite_type',\
        'engagement_skill','target_skill']].corr()

#-----------IV's-------------


# treatment two inner Wait ------
dataForRegression['IW_Treatment'] = np.where(dataForRegression['other_time']< 30, 1, 0)
#120 #130

modelP2 = LogisticRegression(max_iter=10000)
SecondX = dataForRegression[[\
                     'invite_type', 'engagement_skill','target_skill','region','city','country','continent','user_os',\
                                             'browser','score','other_number_words',\
                                            'visitor_duration',\
                                            'agent_duration', 'visitor_number_words', 'agent_number_words',\
                                            'total_canned_lines', 'average_sent', 'min_sent', 'max_sent', 'n_sent_pos', 'n_sent_neg',	'first_sent',\
                                            'last_sent', 'id_rep_code', \
                                            'Invitation_Acep_Day_of_week', 'Invitation_Acep_Hour', \
                                            'NumberofAssigned', 'NumberofAssignedwhenAssigned', \
                     ]]
SecondY = dataForRegression.IW_Treatment
modelP2.fit(SecondX, SecondY)

Second_Propensityscore = np.asarray(modelP2.predict_proba(SecondX))
treatment_plt2=plt.hist(Second_Propensityscore[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt2=plt.hist(Second_Propensityscore[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_IW2_b.pdf',format='pdf')


ATE_IPW2=(((dataForRegression['IW_Treatment']*dataForRegression['Y'])/(Second_Propensityscore[:, 1])).sum())*(1/len(dataForRegression))\
-( ( ( (1-dataForRegression['IW_Treatment'])*dataForRegression['Y'] )/(Second_Propensityscore[:, 0]) ).sum())*(1/len(dataForRegression))

dataForRegression.groupby("Y")["other_time"].mean()
#Out[132]:
#Y
#0    139.905420
#1    237.277638


dataFirstTB = dataForRegression[[\
                     'invite_type', 'engagement_skill','target_skill','region','city','country','continent','user_os',\
                                             'browser','score','other_number_words',\
                                            'visitor_duration',\
                                            'agent_duration', 'visitor_number_words', 'agent_number_words',\
                                            'total_canned_lines', 'average_sent', 'min_sent', 'max_sent', 'n_sent_pos', 'n_sent_neg',	'first_sent',\
                                            'last_sent', 'id_rep_code', \
                                            'Invitation_Acep_Day_of_week', 'Invitation_Acep_Hour', \
                                            'NumberofAssigned', 'NumberofAssignedwhenAssigned','IW_Treatment','Y',\
                     ]]

dataFirstTB0 = dataFirstTB[dataFirstTB["IW_Treatment"] == 0]
dataFirstTB0 = dataFirstTB0.drop(["IW_Treatment"], axis=1)
dataFirstTB0x = dataFirstTB0.iloc[:, :-1]
dataFirstTB0Y = dataFirstTB0['Y']

dataFirstTBx = dataFirstTB.iloc[:, :-1]
dataFirstTBx = dataFirstTBx.drop(["IW_Treatment"], axis=1)

dataFirstTB1 = dataFirstTB[dataFirstTB["IW_Treatment"] == 1]
dataFirstTB1 = dataFirstTB1.drop(["IW_Treatment"], axis=1)
dataFirstTB1x = dataFirstTB1.iloc[:, :-1]
dataFirstTB1Y = dataFirstTB1['Y']

model_0 = LassoCV()
model_1 = LassoCV()

model_0.fit(dataFirstTB0x, dataFirstTB0Y)
model_1.fit(dataFirstTB1x, dataFirstTB1Y)

prediction0 = model_0.predict(dataFirstTBx)
prediction1 = model_1.predict(dataFirstTBx)

ATE_TLearner2 = float((prediction1-prediction0).sum()/len(dataFirstTBx))

#S-Learner
dataFirstTBX = dataFirstTB.iloc[:, :-1]
dataFirstTBY = dataFirstTB['Y']

modelS = Ridge()
modelS.fit(dataFirstTBX, dataFirstTBY)

#T=1
dummT1 = dataFirstTB[dataFirstTB["IW_Treatment"] == 1]
dummT1x = dataFirstTB.iloc[:, :-1]
predictionUn = modelS.predict(dummT1x)

dummT1["IW_Treatment"] = 0
dummT1x = dummT1.iloc[:, :-1]
predictionZe = modelS.predict(dummT1x)

ATE_SLearner2 = (predictionUn.sum() / len(dummT1)) - (predictionZe.sum() / len(dummT1))

# ---------matching---------
#T=0
dummT0 = dataFirstTB[dataFirstTB["IW_Treatment"] == 0]
dummT0 = dummT0.drop(["IW_Treatment"], axis=1)
dummT0x = dummT0.iloc[:, :-1]
dummT0Y = dummT0['Y']

#T=1
dummT1 = dataFirstTB[dataFirstTB["IW_Treatment"] == 1]
dummT1 = dummT1.drop(["IW_Treatment"], axis=1)
dummT1x = dummT1.iloc[:, :-1]
dummT1Y = dummT1['Y']

dummT0V = dummT0.values
dummT1V = dummT1.values

dummT1XV=dummT1x.values
dummT0XV=dummT0x.values


ITE = []
for row in dummT1V:
    dif=abs(row[:-1]-dummT0V[:,:-1])
    #index
    #dif.argmin()
    #value
    distance=row[-1]-dummT0V[dif.argmin(),-1]
    ITE.append(distance)
#Sum of the Treated/number of
ATE_Matchingb2 = float(sum(ITE))/float(len(dataFirstTB))

#--------

corralation=dataFirstTB.corr()

res_second = IV2SLS(dataFirstTB.Y,dataFirstTB[[\
                     'invite_type', 'engagement_skill','target_skill','region','city','country','continent','user_os',\
                                             'browser','score','other_number_words',\
                                            'visitor_duration',\
                                            'agent_duration', 'visitor_number_words', 'agent_number_words',\
                                            'total_canned_lines', 'average_sent', 'min_sent', 'max_sent', 'n_sent_pos', 'n_sent_neg',	'first_sent',\
                                            'last_sent', 'id_rep_code', \
                                            'Invitation_Acep_Day_of_week', 'Invitation_Acep_Hour', \
                                            \
                     ]],\
                    dataFirstTB.IW_Treatment,dataFirstTB.NumberofAssigned).fit(cov_type='unadjusted')

print(res_second)












covariance(dataFirstT.queue_sec,dataFirstT.Rho_atarrival)
#-0.12

corralation=dataFirstT[['queue_sec','Rho_atarrival','invite_type',\
        'engagement_skill','target_skill']].corr()

#treatment 3 Invitation type
DataForThirdTreatment = pd.read_csv('DataForThirdTreatment.csv', index_col=0)

modelP3 = LogisticRegression(max_iter=10000)
ThirdX = DataForThirdTreatment[[\
                     'region','city','country','continent','user_os', 'browser', 'score',\
                    'Arrival_Day_of_week','Arrival_Hour',\
                    'LOS_in_website_before_invorRequest',\
                     ]]
ThirdY = DataForThirdTreatment.InvT
modelP3.fit(ThirdX, ThirdY)

Third_Propensityscore = np.asarray(modelP3.predict_proba(ThirdX))
treatment_plt3=plt.hist(Third_Propensityscore[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt3=plt.hist(Third_Propensityscore[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_Inv.pdf',format='pdf')

ThirdIndex=\
np.where((Third_Propensityscore[:,1] < 0.8)& (Third_Propensityscore[:,1]  > 0.25))



Third_PropensityscoreB=\
Third_Propensityscore[ThirdIndex[0]]

treatment_plt3B=plt.hist(Third_PropensityscoreB[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt3B=plt.hist(Third_PropensityscoreB[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_Inv2.pdf',format='pdf')

datathird=DataForThirdTreatment[[\
                     'region','city','country','continent','user_os', 'browser', 'score',\
                    'Arrival_Day_of_week','Arrival_Hour',\
                    'LOS_in_website_before_invorRequest',\
                     'InvT','Y']]

dataFirstT2=datathird.iloc[ThirdIndex[0]]
len(dataFirstT2) #6387
len(dataFirstT) #23383

ATE_IPW=(((dataFirstT2['InvT']*dataFirstT2['Y'])/(Third_PropensityscoreB[:, 1])).sum())*(1/len(dataFirstT2))\
-( ( ( (1-dataFirstT2['InvT'])*dataFirstT2['Y'] )/(Third_PropensityscoreB[:, 0]) ).sum())*(1/len(dataFirstT2))

dataFirstT.groupby("Y")["queue_sec"].mean()
#Out[132]:
#Y
#0    70.728748
#1    48.358257



dataFirstTB0 = datathird[datathird["InvT"] == 0]
dataFirstTB0 = dataFirstTB0.drop(["InvT"], axis=1)
dataFirstTB0x = dataFirstTB0.iloc[:, :-1]
dataFirstTB0Y = dataFirstTB0['Y']

dataFirstTBx = datathird.iloc[:, :-1]
dataFirstTBx = dataFirstTBx.drop(["InvT"], axis=1)

dataFirstTB1 = datathird[datathird["InvT"] == 1]
dataFirstTB1 = dataFirstTB1.drop(["InvT"], axis=1)
dataFirstTB1x = dataFirstTB1.iloc[:, :-1]
dataFirstTB1Y = dataFirstTB1['Y']

model_0 = LinearRegression()
model_1 = LinearRegression()

model_0.fit(dataFirstTB0x, dataFirstTB0Y)
model_1.fit(dataFirstTB1x, dataFirstTB1Y)

prediction0 = model_0.predict(dataFirstTBx)
prediction1 = model_1.predict(dataFirstTBx)

ATE_TLearner3 = float((prediction1-prediction0).sum()/len(dataFirstTBx))

#S-Learner
dataFirstTBX = datathird.iloc[:, :-1]
dataFirstTBY = datathird['Y']

modelS = Ridge()
modelS.fit(dataFirstTBX, dataFirstTBY)

#T=1
dummT1 = datathird[datathird["InvT"] == 1]
dummT1x = datathird.iloc[:, :-1]
predictionUn = modelS.predict(dummT1x)

dummT1["InvT"] = 0
dummT1x = dummT1.iloc[:, :-1]
predictionZe = modelS.predict(dummT1x)

ATE_SLearner3 = (predictionUn.sum() / len(dummT1)) - (predictionZe.sum() / len(dummT1))