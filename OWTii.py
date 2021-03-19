
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



import statsmodels.api as sm
from scipy import stats
from linearmodels.iv import IV2SLS

import os, glob
import pandas as pd

#FIRST TREATMENT ii

dataFirstT = pd.read_csv('DataForFirstTreatmentii.csv', index_col=0)
len(dataFirstT)#12,343

dataFirstT.WaitTreatment\
    = np.where(dataFirstT['queue_sec']< 30, 1, 0)

dataFirstT.WaitTreatment\
    = np.where(dataFirstT['queue_sec']< 60, 1, 0)


dFT=dataFirstT[[\
'invite_type','engagement_skill','target_skill','region','city','country','continent','user_os','browser','score','Invitation_Acep_Day_of_week','Invitation_Acep_Hour','Rho_atarrival','WaitTreatment','Y'\
]]

modelP = LogisticRegression(max_iter=10000)
FirstX = dataFirstT[[\
                     'Invitation_Acep_Hour', \
                    'Rho_atarrival', 'invite_type', \
                     'engagement_skill', 'target_skill',\
                    'Invitation_Acep_Day_of_week',\
                    'score','region','city','country','continent',\
                    'user_os','browser'\
                     ]]
#Wait time
FirstY = dataFirstT.iloc[:, -2]
modelP.fit(FirstX, FirstY)

reg2 = LinearRegression().fit(FirstX, FirstY)

First_Propensityscore = np.asarray(modelP.predict_proba(FirstX))
treatment_plt=plt.hist(First_Propensityscore[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt=plt.hist(First_Propensityscore[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_wait60ii.pdf',format='pdf')

#hist, bin_edges = np.histogram(First_Propensityscore[:, 1])

FirstIndex=\
np.where((First_Propensityscore[:, 1] < 0.55)& (First_Propensityscore[:, 1] > 0.46))


First_PropensityscoreB=\
First_Propensityscore[FirstIndex[0]]

treatment_plt1B=plt.hist(First_PropensityscoreB[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt1B=plt.hist(First_PropensityscoreB[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_wait602ii.pdf',format='pdf')

dataFirstT2=dFT.iloc[FirstIndex[0]]
len(dataFirstT2) #30 sec #1295 #60 sec #4,675
len(dataFirstT) #23383

ATE_IPWii=(((dataFirstT2['WaitTreatment']*dataFirstT2['Y'])/(First_PropensityscoreB[:, 1])).sum())*(1/len(dataFirstT2))\
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

ATE_TLearnerii = float((prediction1-prediction0).sum()/len(dataFirstTBx))
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

ATE_SLearnerii = (predictionUn.sum() / len(dummT1)) - (predictionZe.sum() / len(dummT1))
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
    dif=abs(row[:-1]-dummT0V[:,:-1]).sum(axis=1)
    #index
    #dif.argmin()
    #value
    distance=row[-1]-dummT0V[dif.argmin(),-1]
    ITE.append(distance)
#Sum of the Treated/number of Treated
ATE_Matchingbii = float(sum(ITE))/float(len(dataFirstTB))

# ---------matching---------


#-----------IV's-------------
corralation=dataFirstTB.corr()

res_second = IV2SLS(dataFirstTB.Y,dataFirstTB[['invite_type','engagement_skill','Rho_atarrival','region','city','country','continent','user_os','browser','score','Invitation_Acep_Hour']],\
                    dataFirstTB.WaitTreatment, dataFirstTB.Invitation_Acep_Day_of_week).fit(cov_type='unadjusted')

print(res_second)

covariance(dataFirstT.queue_sec,dataFirstT.Rho_atarrival)
#-0.12

corralation=dataFirstT[['queue_sec','Rho_atarrival','invite_type',\
        'engagement_skill','target_skill']].corr()

#-----------IV's-------------


#FIRST TREATMENT ii -30 sec

dataFirstT = pd.read_csv('DataForFirstTreatmentii.csv', index_col=0)
dataFirstT.WaitTreatment\
    = np.where(dataFirstT['queue_sec']< 30, 1, 0)

len(dataFirstT)#12,343

dFT=dataFirstT[[\
'invite_type','engagement_skill','target_skill','region','city','country','continent','user_os','browser','score','Invitation_Acep_Day_of_week','Invitation_Acep_Hour','Rho_atarrival','WaitTreatment','Y'\
]]

modelP = LogisticRegression(max_iter=10000)
FirstX = dataFirstT[[\
                     'Invitation_Acep_Hour', \
                    'Rho_atarrival', 'invite_type', \
                     'engagement_skill', 'target_skill',\
                    'Invitation_Acep_Day_of_week',\
                    'score','region','city','country','continent',\
                    'user_os','browser'\
                     ]]
#Wait time
FirstY = dataFirstT.iloc[:, -2]
modelP.fit(FirstX, FirstY)

reg2 = LinearRegression().fit(FirstX, FirstY)

First_Propensityscore = np.asarray(modelP.predict_proba(FirstX))
treatment_plt=plt.hist(First_Propensityscore[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt=plt.hist(First_Propensityscore[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_waiti2.pdf',format='pdf')

#hist, bin_edges = np.histogram(First_Propensityscore[:, 1])

FirstIndex=\
np.where((First_Propensityscore[:, 1] < 0.60)& (First_Propensityscore[:, 1] > 0.45))


First_PropensityscoreB=\
First_Propensityscore[FirstIndex[0]]

treatment_plt1B=plt.hist(First_PropensityscoreB[:, 1],fc=(0,0,1,0.5),bins=10,label='Treated')
cont_plt1B=plt.hist(First_PropensityscoreB[:, 0],fc=(1,0,0,0.5),bins=10,label='Control')
plt.legend();
plt.xlabel('propensity score');
plt.ylabel('number of units');
plt.savefig('prop_score_wait2ii.pdf',format='pdf')

dataFirstT2=dFT.iloc[FirstIndex[0]]
len(dataFirstT2) #4,667
len(dataFirstT) #23383

ATE_IPWii=(((dataFirstT2['WaitTreatment']*dataFirstT2['Y'])/(First_PropensityscoreB[:, 1])).sum())*(1/len(dataFirstT2))\
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

ATE_TLearnerii = float((prediction1-prediction0).sum()/len(dataFirstTBx))
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

ATE_SLearnerii = (predictionUn.sum() / len(dummT1)) - (predictionZe.sum() / len(dummT1))
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
    dif=abs(row[:-1]-dummT0V[:,:-1]).sum(axis=1)
    #index
    #dif.argmin()
    #value
    distance=row[-1]-dummT0V[dif.argmin(),-1]
    ITE.append(distance)
#Sum of the Treated/number of Treated
ATT_Matchingbii = float(sum(ITE))/float(len(dummT1V))

# ---------matching---------


#-----------IV's-------------
corralation=dataFirstTB.corr()

res_second = IV2SLS(dataFirstTB.Y,dataFirstTB[['engagement_skill','target_skill','region','city','country','continent','user_os','browser','score','Invitation_Acep_Day_of_week','Invitation_Acep_Hour']],\
                    dataFirstTB.WaitTreatment, dataFirstTB.Rho_atarrival).fit(cov_type='unadjusted')

print(res_second)

covariance(dataFirstT.queue_sec,dataFirstT.Rho_atarrival)
#-0.12

corralation=dataFirstT[['queue_sec','Rho_atarrival','invite_type',\
        'engagement_skill','target_skill']].corr()

#-----------IV's-------------
