
# coding: utf-8

# ## Realizando as importações

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split


# ## Importando os dados de tumores para usar como base de treinamento 

# In[14]:


previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')


# In[15]:


previsores.head()


# In[16]:


classe.head()


# ## Criando  a base de dados para treinamento e teste
# ### Será utilizado 25% da base de dados para testes utilizando o metodo train_test_split do sklearn

# In[17]:


previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)


# # Criando os neuronios 
# ## Usando Keras, criamos 30 neuronios na camada de entrada, uma pra cada particularidade dos dados de cancer
# ## Usando 16 neuronios na camada oculta, formula (30 entrada + 1 saida / 2)
# ## Usando 1 neuronio na camada de saida para resultados falsos ou verdadeiros

# In[18]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[19]:


classificador = Sequential()


# In[20]:


classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))


# In[21]:


classificador.add(Dense(units=1, activation='sigmoid'))


# In[22]:


classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])


# In[23]:


classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)


# In[ ]:




