

# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns


# In[ ]:


#set working directory
os.chdir("F:\Others\Project Bike Rental")


# In[ ]:


#Load data
data_bike = pd.read_csv("day.csv")


# In[ ]:


data_bike.dtypes


# In[ ]:


data_bike[dteday] = int(data_bike[dteday])


# In[ ]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(data_bike.isnull().sum())


# In[ ]:


#Reset Index
missing_val = missing_val.reset_index()


# In[ ]:


#Rename variables
missing_val = missing_val.rename(columns={'index':'Variables', 0 : 'Missing_Percentage'})


# In[ ]:


#calculate percentage
missing_val["Missing_Percentage"] = (missing_val['Missing_Percentage']/len(data_bike))*100


# In[ ]:


#desecnding order 
missing_val = missing_val.sort_values('Missing_Percentage', ascending = False).reset_index(drop = True)


# In[ ]:


missing_val


# In[ ]:


#plot boxplot to visualize outliers
get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(data_bike['casual'])


# In[ ]:


data_bike.head(10)


# In[ ]:


#Save numeric variables
cnames = ["instant","temp", "atemp","hum","windspeed","casual","registered","cnt"]


# In[ ]:


#Detect and delete outliers from the data
for i in cnames:
    print (i)
    q75, q25 = np.percentile(data_bike.loc[:,i], [75 ,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5) #Innerfence
    max = q75 + (iqr*1.5) #Upperfence
    print(min)
    print(max)
    data_bike = data_bike.drop(data_bike[data_bike.loc[:,i] < min].index)
    data_bike = data_bike.drop(data_bike[data_bike.loc[:,i] > max].index)


# In[ ]:


##Correlation plot
#Corelation plot
df_corr = data_bike.loc[:,cnames]


# In[ ]:


#Set the width and height of plot
f , ax = plt.subplots(figsize =(7,5))

#Set correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask = np.zeros_like(corr ,dtype = np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[ ]:


##Chi square test
#Save categorical variable
cat_names = ["dteday","yr","mnyh","holiday ","weekday","workingday","weathersit"]


# In[ ]:


cat_names


# In[ ]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data_bike['season'],cat_names[i]))
    print(p)


# In[ ]:


## Drop the variable 
#Drop the variables from data
data_bike = data_bike.drop(['dteday', 'weathersit', 'yr', 'holiday', 'weekday', 'workingday', 'temp', 'registered'], axis=1)


# In[ ]:


#Normality check
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data_bike['cnt'], bins = 'auto')

#Normalisation
for i in cnames:
    print(i)
    data_bike[i] = (data_bike[i] - min(data_bike[i]))/(max(data_bike[i]) -data_bike[i]))


# In[ ]:


data_bike.head(10)


# In[ ]:


#Import libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


# In[ ]:


##############decision tree for regression############

#Divide data into train and test
train , test = train_test_split(data_bike , test_size = 0.2)


# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


#decision tree for regresion
fit_DT = DecisionTreeRegressor(max_depth = 2).fit(train.iloc[:,0:7], train.iloc[:,7])


# In[ ]:


fit_DT


# In[ ]:


#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:7])


# In[ ]:


#Calculate MAPE
def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return mape


# In[ ]:


MAPE(test.iloc[:,7], predictions_DT)


###################### Linear Regression ##################

# In[ ]:


#Import libraries for LR
import statsmodels.api as sm

#Train the model using the training sets
model = sm.OLS(train.iloc[:,7],
              train.iloc[:,0:7]).fit()


# In[ ]:


#Print out the summary
model.summary()


# In[ ]:


#make the predictions by the model
predictions_LR = model.predict(test.iloc[:,0:7])


# In[ ]:


#Calculate MAPE
MAPE(test.iloc[:,7], predictions_LR)

