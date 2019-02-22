import pandas as pd 
import numpy as np 




#Reading the Train file
train = pd.read_csv("train.csv")


#Preprocessing the data:

df1 = pd.DataFrame(train)
df1 = df1.drop(["MiscFeature","PoolQC","Alley","Fence","Id","FireplaceQu"],axis=1)


#print(df1.info())

X = df1.drop("SalePrice" ,axis= 1)
X["BoughtOffPlan"] = X.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
X = X.drop("SaleCondition", axis = 1)
#print(X.head(50))
#print(X.info())
Y = df1['SalePrice'].values
Y = Y.reshape(-1, 1)
#print(Y.head())
#X_Dum = df1[['MSZoning','Street','SaleType']]
#
#print(X_dummy.head(50))
categorical_features = X.select_dtypes(include = ["object"]).columns
numerical_features = X.select_dtypes(exclude = ["object"]).columns
print(str(len(categorical_features)))
print(str(len(numerical_features)))
X_num =X[numerical_features]
X_num = X_num.fillna(X_num.mean())
#print(X_num.head())
X_cat = X[categorical_features]
X_dummy = pd.get_dummies(X_cat)
print(X_dummy.info())

#print(X_dummy.head())

X_Concat = pd.concat([X_num,X_dummy], axis=1 )
X_Concat = X_Concat.values
#print(X_Concat.info())

# Building the Machine Learning Algorithm 
# Multiple Linear Regression 
import statsmodels.api as sm
X_Scaled1 = sm.add_constant(X_Scaled)
model = sm.OLS(Y, X_Scaled1)
fitt = model.fit()
fitt.summary()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()
X_Scaled = sc.fit_transform(X_Concat)

from sklearn.linear_model import LinearRegression
clf_gini = LinearRegression()
X_tr, X_val, Y_tr, Y_val = train_test_split(X_Scaled, Y,test_size=0.2, random_state = 3)
clf_gini.fit(X_tr,Y_tr) 
#Y_scaled= sc.fit_transform(Y)
#Y = Y.reshape(-1, 1)

predictions= clf_gini.predict(X_val)

for i in range(10):
    print(Y_val[i], predictions[i])

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_val, predictions)))
print(metrics.r2_score(Y_val, predictions))