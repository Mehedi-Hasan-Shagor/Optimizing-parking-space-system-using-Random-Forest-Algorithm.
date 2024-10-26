#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# In[9]:


file_dir=r"C:\Users\Vechile.csv"


# In[10]:


df=pd.read_csv(file_dir)
df.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder
labellencoder=LabelEncoder()
for i in df.columns:
    df[i]=labellencoder.fit_transform(df[i])
df.head()


# In[27]:


df.describe()


# In[11]:


x=df.drop('Vehicle',axis=1)
y=df['Vehicle']
print(x)


# In[38]:


from sklearn.prepocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform()


# In[1]:


pip install scikit-learn


# In[2]:


from sklearn.prepocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform()


# In[3]:


pip install scikit-learn


# In[4]:


pip install --upgrade scikit-learn


# In[13]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.75,random_state=400)
print(x_train)


# In[17]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y_train,data=df,palette="mako_r")
plt.ylabel('class')
plt.xlabel('Total')
plt.show()


# In[21]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[24]:


y_train_pre=rf.predict(x_train)
y_test_pre=rf.predict(x_test)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[29]:


print(confusion_matrix(y_train,y_train_pre))
print(confusion_matrix(y_test,y_test_pre))


# In[30]:


print(accuracy_score(y_train,y_train_pre))
print(accuracy_score(y_test,y_test_pre))


# In[31]:


print(classification_report(y_train,y_train_pre))
print(classification_report(y_test,y_test_pre))


# In[ ]:







# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[15]:


file_dir=r"C:\Users\Vechile.csv"
df=pd.read_csv(file_dir)
df.head(21)


# In[25]:


from sklearn.preprocessing import LabelEncoder
labellencoder=LabelEncoder()
for i in df.columns:
    df["ID"]=labellencoder.fit_transform(df["ID"])
df.head(20)
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#x=sc.fit_transform(x)
#x


# In[27]:


df.describe()


# In[28]:


x=df.drop('Vehicle',axis=1)
# y means target vechile
y=df['Vehicle']
print(x)


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.50,random_state=400)
print(x_train)


# In[37]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y_train,data=df,palette="mako_r")
plt.ylabel('class')
plt.xlabel('Total')
plt.show()


# In[38]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[39]:


y_train_pre=rf.predict(x_train)
y_test_pre=rf.predict(x_test)
print(y_train_pre)
print(y_test_pre)


# In[43]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_train,y_train_pre))
print(confusion_matrix(y_test,y_test_pre))
print(accuracy_score(y_train,y_train_pre))
print(accuracy_score(y_test,y_test_pre))
print(classification_report(y_train,y_train_pre))
print(classification_report(y_test,y_test_pre))


# In[61]:


x_test_outside=[["rtr4",10.3,2.4,3.1,12300]]
print(x_test_outside[0][0])
x_test_outside[0]=labellencoder.fit_transform(x_test_outside[0])
print(x_test_outside[0][0])
y_pre=rf.predict(x_test_outside)
print(y_pre)


# In[ ]:




