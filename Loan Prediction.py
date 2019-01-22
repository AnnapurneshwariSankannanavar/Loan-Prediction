
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

df = pd.read_csv("E:/Anu/MyLearning/MachineLearning/MachineLearning/LoanPrediction/train.csv")

#Reading the first few rows
df.head()


# In[5]:


# Checking the summary of the data set. 
# From below summary it is seen that some of the fields have missing values and the ouliers
df.describe()


# In[9]:


df['Credit_History'].value_counts()


# In[102]:


df['Property_Area'].value_counts()


# In[11]:


#The applicant income has some outliers
df.hist('ApplicantIncome',bins=50)


# In[12]:


df.boxplot(column='ApplicantIncome')


# In[14]:


df.boxplot(column='ApplicantIncome', by='Education')


# In[17]:


df['LoanAmount'].hist(bins=50)


# In[18]:


df.boxplot(column='LoanAmount')


# In[103]:


temp1 = df['Credit_History'].value_counts()


# In[46]:


#For people having credit history, the probability of getting loan is high
temp2 =df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=
               lambda x:x.map({'Y':1,'N':0}).mean())
temp2


# In[50]:



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of applicatnts')
ax1.set_title('Applicants by credit history')
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('Credit history')
ax2.set_ylabel('Loan status')
ax2.set_title('Loan status by credit history')





# In[53]:


temp3 = pd.crosstab(df['Credit_History'],df['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[62]:


df.apply(lambda x:sum(x.isnull()), axis =0)


# In[58]:


#filling the missing value of LoanAmount with mean value
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)



# In[60]:


df['Self_Employed'].value_counts()


# In[61]:


#filling missing value with No(mode value), for Self_Employed column
df['Self_Employed'].fillna('No', inplace=True)


# In[64]:


#To take into account of ouliers getting the value
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=50)


# In[66]:


# We can combne the applicant income and the co-applicant income.
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

df['TotalIncome_log'].hist(bins=50)


# In[67]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace =True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[71]:


df.apply(lambda x:sum(x.isnull()),axis=0)



# In[91]:


from sklearn.linear_model  import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes



# In[97]:


def classification_model(model, data, predictors,outcome):
    model.fit(data[predictors],data[outcome])
    predicted_values = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predicted_values,data[outcome])
    print "Accuracy: %s", accuracy
    
    


# In[100]:


#Using logistic regression to predict the chance of getting loan
predictors= ['Credit_History', 'Education', 'Married','Self_Employed']

outcome = ['Loan_Status']

model = LogisticRegression()

classification_model(model, df,predictors,outcome)


# In[125]:


from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x,y = df.loc[:,['Credit_History', 'Education', 'Married','Self_Employed']],df.loc[:,'Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=4)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

predicted_values = logreg.predict(y_train)
accuracy = metrics.accuracy_score(predicted_values,data[outcome])
print(accuracy)

