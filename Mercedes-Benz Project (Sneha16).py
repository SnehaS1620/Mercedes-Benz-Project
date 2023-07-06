#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


dataset_test= pd.read_csv('test.csv')
dataset_train= pd.read_csv('train.csv')


# In[79]:


dataset_test.head(8)


# In[80]:


dataset_train.head(8)


# In[81]:


dataset_train.shape


# In[82]:


dataset_test.shape


# In[83]:


for i in dataset_train.columns:
    data_type = dataset_train[i].dtype
    if data_type == 'object':
        print(i) 


# ### 1.If for any column(s), the variance is equal to zero, then you need to remove those variable(s).

# In[84]:


variance=pow(dataset_train.drop(columns={'ID', 'y'}).std(),2).to_dict() 
null_count = 0 
for key, value in variance.items():
    if(value==0):
        print('Name=',key)
        null_count=null_count+1 
print('Number of columns which has zero variance=',null_count)


# In[85]:


dataset_train=dataset_train.drop(columns={'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X28', 'X290', 'X293' ,'X297','X330','X347'})


# In[86]:


dataset_train.head()


# In[87]:


dataset_train.shape


# ### 2.Check for null and unique values for test and train sets.

# In[88]:


dataset_train.isnull().sum().any()


# In[90]:


dataset_train.nunique()


# In[91]:


dataset_test.isnull().sum().any()


# In[92]:


dataset_test.nunique()


# In[93]:


#Filterout the column having object datatype
# For train dataset


# In[94]:


object_datatype= dataset_train.select_dtypes(include=[object])


# In[95]:


object_datatype


# In[96]:


object_datatype_columns=object_datatype.columns
object_datatype_columns


# ### 3.Apply label encoder

# In[98]:


from sklearn import preprocessing


# In[99]:


label_encoder=preprocessing.LabelEncoder()


# In[100]:


dataset_train['X0'].unique()


# In[101]:


dataset_train['X0'] = label_encoder.fit_transform (dataset_train['X0'])
dataset_train['X0'].unique()


# In[102]:


dataset_train['X1'] = label_encoder.fit_transform (dataset_train['X1'])
dataset_train['X2'] = label_encoder.fit_transform (dataset_train['X2'])
dataset_train['X3'] = label_encoder.fit_transform (dataset_train['X3'])
dataset_train['X4'] = label_encoder.fit_transform (dataset_train['X4'])
dataset_train['X5'] = label_encoder.fit_transform (dataset_train['X5'])
dataset_train['X6'] = label_encoder.fit_transform (dataset_train['X6'])
dataset_train['X8'] = label_encoder.fit_transform (dataset_train['X8'])


# In[103]:


dataset_train


# #### For test dataset

# In[104]:


dataset_test.nunique()


# In[105]:


object_datatype_test=dataset_test.select_dtypes(include=[object])
object_datatype_test


# In[106]:


object_datatype_test_columns=object_datatype_test.columns
object_datatype_test_columns


# #### Applying label encoder

# In[107]:


dataset_test['X0']=label_encoder.fit_transform(dataset_test['X0'])
dataset_test['X0'].unique()


# In[108]:


dataset_test['X1']=label_encoder.fit_transform(dataset_test['X1'])
dataset_test['X2']=label_encoder.fit_transform(dataset_test['X2'])
dataset_test['X3']=label_encoder.fit_transform(dataset_test['X3'])
dataset_test['X4']=label_encoder.fit_transform(dataset_test['X4'])
dataset_test['X5']=label_encoder.fit_transform(dataset_test['X5'])
dataset_test['X6']=label_encoder.fit_transform(dataset_test['X6'])
dataset_test['X8']=label_encoder.fit_transform(dataset_test['X8'])


# In[109]:


dataset_test.head()


# ### 3.Perform dimensionality reduction.

# In[ ]:


#For train dataset


# In[110]:


from sklearn.decomposition import PCA


# In[111]:


sklearn_PCA= PCA(n_components=0.95)


# In[112]:


sklearn_PCA.fit(dataset_train)


# In[113]:


x_train_transformed=sklearn_PCA.transform(dataset_train)


# In[114]:


x_train_transformed.shape


# In[115]:


sklearn_pca_98=PCA(n_components=0.98)


# In[116]:


sklearn_pca_98.fit(dataset_train)


# In[117]:


x_train_transformed_98=sklearn_pca_98.transform(dataset_train)
x_train_transformed_98.shape


# In[118]:


dataset_train.y


# In[ ]:


# For test dataset


# In[119]:


sklearn_PCA_test_98= PCA(n_components=0.98)


# In[120]:


sklearn_PCA_test_98.fit(dataset_test)


# In[121]:


x_test_transformed_98=sklearn_PCA_test_98.transform(dataset_test)
x_test_transformed_98.shape


# In[122]:


x_test_transformed_98=sklearn_PCA_test_98.transform(dataset_test)
x_test_transformed_98.shape


# In[126]:


#Train and test split on dataset


# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


X=dataset_train.drop('y' , axis=1)
Y=dataset_train.y
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=42)


# In[125]:


print(xtrain)
print(xtrain.shape)


# In[127]:


print(ytrain)
print(ytrain.shape)


# In[128]:


print(xtest)
print(xtest.shape)


# In[129]:


pca_xtrain=PCA(n_components=0.95)
pca_xtrain.fit(xtrain)


# In[130]:


pca_xtrain_transformed=pca_xtrain.transform(xtrain)
pca_xtrain_transformed.shape


# In[131]:


pca_xtest=PCA(n_components=0.95)
pca_xtest.fit(xtest)


# In[132]:


pca_xtest_transformed=pca_xtest.transform(xtest)
pca_xtest_transformed.shape


# In[133]:


print(pca_xtest.explained_variance_)
print(pca_xtest.explained_variance_ratio_)


# In[134]:


print(pca_xtrain.explained_variance_)
print(pca_xtrain.explained_variance_ratio_)


# ### 4.Predict your test_df values using XGBoost

# In[135]:


import warnings
warnings.filterwarnings('ignore')


# In[136]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[137]:


xgb_r = xgb.XGBRegressor(objective ='reg:linear', n_estimators = 10)


# In[138]:


xgb_r.fit(xtrain, ytrain)


# In[139]:


y_pred = xgb_r.predict(xtrain)


# In[140]:


print("The RMSE value of the model on training dataset is : ")
print(np.sqrt(mean_squared_error(ytrain , y_pred)))


# In[141]:


print("The r2 score of the model on training dataset is : ")
print(r2_score(ytrain , y_pred))


# In[142]:


ytest_pred_xgb = xgb_r.predict(xtest)
xtest_output = pd.DataFrame(ytest_pred_xgb, columns=['xtest_output'])
xtest_output


# In[ ]:




