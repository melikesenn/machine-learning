#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


veri = pd.read_csv("tvmarketing.csv") #tvmarketing verisini okunur.


# In[5]:


veri.head()  #ilk 5 satırını göster. Default 5 yazılıdır.


# In[6]:


veri.tail(3) #son 3 satırını göster


# In[7]:


veri.info() # veri hakkında bize bilgi verir.


# In[9]:


veri.shape #satır ve kolon sayısını gösterir.


# In[11]:


veri.describe() #verinin istatisliklerini bize verir. %delik kartil değerleride buna dahildir.


# In[19]:


import seaborn as sns # seaborn ile verimizi görselleştirelim bununiçin seaborn kütüphanesini içeri aktardık.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


sns.pairplot(veri, x_vars = 'TV', y_vars = "Sales", height =5 ,aspect = 0.6, kind = "scatter")


# In[27]:


# Veri setimizin eksenlerini değişkenlere atalım. X ve Y olarak ayırdık.
X = veri['TV']
X.head(3)


# In[29]:


Y = veri['Sales']
Y.head(4)


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.7,random_state=100) #veriyi train ve test set olarak ayırdık. Eğitim seti oranı %70 olarak seçildi


# In[34]:


x_train = x_train[:,np.newaxis] #x_train normalde(140,)olarak ayarlıdır fakat biz newaxismetodu ile(140,1) yaptık
x_test = x_test[:,np.newaxis]


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


lineer_regresyon = LinearRegression() #LinearRegression objesini lineer_regresyona değişkenine attık


# In[37]:


lineer_regresyon.fit(x_train,y_train) #fit ile modeli öğreniriz.


# In[38]:


#lineer regresyon formüllerinden katsayıları öğrenebiliriz.
print(lineer_regresyon.intercept_)
print(lineer_regresyon.coef_)


# In[39]:


y_pred = lineer_regresyon.predict(x_test)#modelin gücünü ölçmek için xtest kısmını modele input olarak verdik.


# In[57]:


t = np.linspace(0,140,60)


# In[51]:


fig = plt.figure()


# In[59]:


plt.plot(t,y_test,linestyle ="-",color="pink")  #gerçek değerler


# In[62]:


plt.plot(t,y_pred,color ="red")  #tahminlenmiş değerler


# In[67]:


#hatamızı görmek için
fig = plt.figure()
print("test oranı {0}".format(y_test-y_pred))
plt.title("hata oranı")
plt.xlabel("index")
plt.ylabel("y_test-y_pred")
plt.plot(t,y_test-y_pred,color="blue")


# In[72]:


#model başarısına bakarsak
from sklearn.metrics import mean_squared_error, r2_score #R SQUARED İÇİN kütüphanemizi çağırdık
ort_hata = mean_squared_error(y_test,y_pred)
r_squared = r2_score(y_test,y_pred)
print("ortama hata {0} \nr squared {1} ".format(ort_hata,r_squared))


# In[76]:


plt.scatter(y_test,y_pred)
plt.xlabel("y test")
plt.ylabel("y prediction")
plt.title("tahminlenmiş ve gerçek değer arasındaki korelasyon ilişkisi")


# In[ ]:




