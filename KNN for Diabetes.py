#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Loading the dataset
df = pd.read_csv('diabetes.csv')


# In[5]:


#Print the first 5 rows of the dataframe.
df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.hist(figsize = (20,20))


# In[15]:


sns.countplot(data=df, x=df.Outcome)
plt.xlabel("Outcome")
plt.ylabel("count of each data type")
plt.show()


# In[16]:


sns.pairplot(df, hue = 'Outcome')


# In[17]:


plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn') 


# In[19]:


x= df.iloc[:, :-1]
y= df.iloc[:,-1]
print(x)
print(y)


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[60]:


from sklearn.preprocessing import StandardScaler


# In[61]:


std_scaler = StandardScaler()


# In[62]:


X_train1 = std_scaler.fit_transform(X_train)


# In[63]:


X_test1 = std_scaler.transform(X_test) #data leakage


# In[64]:


X_train1, X_test1


# In[65]:


from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[66]:


# score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[67]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[68]:


plt.figure(figsize=(12,5))
sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[69]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[70]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[71]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[52]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[53]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[54]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# In[55]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# In[57]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))


# In[ ]:




