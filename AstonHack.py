import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from ipykernel import kernelapp as app
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

df=pd.read_csv("Agriculture.csv",encoding = "ISO-8859-1")
df.dtypes

df.head

df['Production']=pd.to_numeric(df['Production'],errors='coerce')

data=df.groupby(['Crop_Year'])['Area','Production'].mean()
data=data.reset_index(level=0, inplace=False)
data

#crop productivity index
data['CPI']=data['Production']/data['Area']
data.head()

#data.describe()


#BOXPLOT
#import seaborn as sns
#sns.boxplot(x=data['CPI'])
#
data = data[np.isfinite(data['CPI'])]
data=data[data.CPI >43]
data=data[data.CPI <51]
data.set_index('Crop_Year')
data

#HISTOGRAM
data.hist()

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#Plotting Production and Area on a Area vs Crop Year graph
x_axis=data.Crop_Year
y_axis=data.Area

y1_axis=data.Production

plt.plot(x_axis,y_axis)
plt.plot(x_axis,y1_axis,color='r')

plt.title("Production and area ")
plt.legend(["Production ","Area"])
plt.show()

#plotting of production
x_axis=data.Crop_Year
y1_axis=data.Production



plt.plot(x_axis,y1_axis)

plt.title("Year vs Production ")
plt.legend(["Year ","Production"])
plt.show()


#Applying Random Forest Regressor
#Checking Y_pred on X itself, instead of dividing onto x_train
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

x=data.iloc[:,0:1].values
y=data.iloc[:,3].values
regressor=RandomForestRegressor(n_estimators=12,random_state=0,n_jobs=1,verbose=13)

regressor.fit(x,y)


y_pred=regressor.predict(x)
y_pred


x_grid=np.arange(min(x),max(x),0.001)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='r')
plt.plot(x_grid,regressor.predict(x_grid),color='b')
a=plt.show()
a

#actual and predicted values
dm = pd.DataFrame({'Actual': y, 'Predicted': y_pred}).reset_index()
x_axis=dm.index
y_axis=dm.Actual
y1_axis=dm.Predicted
plt.plot(x_axis,y_axis)
plt.plot(x_axis,y1_axis)
plt.title("Actual vs Predicted")
plt.legend(["actual ","predicted"])
b=plt.show()
b

#Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
#print('MSE is')
MSE= mean_squared_error(y, y_pred)
#print(mean_squared_error(y, y_pred))
##print(accuracy_score(y, y_pred))
#print('')
#print('R squared value is')
r=r2_score(y, y_pred)
#print(r2_score(y, y_pred))

print ('Mean squared error is', MSE, 'R squared value is', r)