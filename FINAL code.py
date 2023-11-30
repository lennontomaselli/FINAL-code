#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import pandas as pd

current_directory = os.getcwd()
print(current_directory)


# In[32]:


# Change working directory
new_directory_path = r'/Users/lennontomaselli'
os.chdir(new_directory_path)


# In[33]:


updated_dir = os.getcwd()
print(updated_dir)


# In[87]:


file_path = "Hospital1.txt"

try:
    with open(file_path, "r") as file:
        content = file.read()
        print(content)
except filenotFoundError:
        print(f"file '{file_path}' not found.")
except IOError:
        print("An error occurred while reading this file.")


# In[88]:


# Number of patients readmitted
df = pd.read_csv(file_path)
Readmission = (df[' Readmission'] == 1).sum()

print(f"The number of readmission patients is {Readmission}.")


# In[69]:


# Staff satisfaction
df = pd.read_csv(file_path)
StaffSatisfaction = (df[' StaffSatisfaction']).mean()

print(f" The Staff satisfaction is {StaffSatisfaction}")


# In[70]:


# Overall satisfaction
df['OverallSatisfaction'] = df[[' StaffSatisfaction', ' CleanlinessSatisfaction',
                          ' FoodSatisfaction', ' ComfortSatisfaction',
                          ' CommunicationSatisfaction']].mean(axis = 1)


# In[79]:


# Calculating statistics
import numpy as np
num_readmitted_1 = np.sum(df[' Readmission'])
satisfaction_staff = np.mean(df[' StaffSatisfaction'])
satisfaction_cleanliness = np.mean(df[' CleanlinessSatisfaction'])
satisfaction_food = np.mean(df[' FoodSatisfaction'])
satisfaction_comfort = np.mean(df[' ComfortSatisfaction'])
satisfaction_communication = np.mean(df[' CommunicationSatisfaction'])


# In[77]:


# Logistic Regression

import sklearn.linear_model

X = df['OverallSatisfaction'].values.reshape(-1,1)
Y = df[' Readmission']

log_reg = sklearn.linear_model.LogisticRegression().fit(X, Y)


# In[74]:


# Print descriptive stats

print(f"Number of patients readmitted: {num_readmitted_1}.")
print(f"Average staff satisfaction: {satisfaction_staff}.")
print(f"Average cleanliness satisfaction: {satisfaction_cleanliness}.")
print(f"Average food satisfaction: {satisfaction_food}.")
print(f"Average comfort satisfaction: {satisfaction_comfort}.")
print(f"Average communication satisfaction: {satisfaction_communication}.")


# In[68]:


# Correlation coefficient 

correlation_coefficient = log_reg.coef_[0][0]


# In[42]:


#Plot

import matplotlib.pyplot as plt

plt.plot(X, log_reg.predict(X), label = "Regression Line", color= "pink")
plt.scatter(df['OverallSatisfaction'], df[' Readmission'], color= "green")
plt.show()


# In[85]:


file_path2 = "Hospital2.txt"

try:
    with open(file_path, "r") as file:
        content = file.read()
        print(content)
except filenotFoundError:
        print(f"file '{file_path2}' not found.")
except IOError:
        print("An error occurred while reading this file.")


# In[94]:


# Number of patients readmitted
file_path2 = "Hospital2.txt"

df = pd.read_csv(file_path2)

Readmission2 = (df[' Readmission'] == 1).sum()

print(f"The number of readmission patients is {Readmission2}.")


# In[95]:


# Staff satisfaction
df = pd.read_csv(file_path2)
StaffSatisfaction = (df[' StaffSatisfaction']).mean()

print(f" The Staff satisfaction is {StaffSatisfaction}")


# In[96]:


# Overall satisfaction
df['OverallSatisfaction'] = df[[' StaffSatisfaction', ' CleanlinessSatisfaction',
                          ' FoodSatisfaction', ' ComfortSatisfaction',
                          ' CommunicationSatisfaction']].mean(axis = 1)


# In[97]:


# Calculating statistics
import numpy as np
num_readmitted_2 = np.sum(df[' Readmission'])
satisfaction_staff = np.mean(df[' StaffSatisfaction'])
satisfaction_cleanliness = np.mean(df[' CleanlinessSatisfaction'])
satisfaction_food = np.mean(df[' FoodSatisfaction'])
satisfaction_comfort = np.mean(df[' ComfortSatisfaction'])
satisfaction_communication = np.mean(df[' CommunicationSatisfaction'])


# In[98]:


# Print descriptive stats

print(f"Number of patients readmitted: {num_readmitted_2}.")
print(f"Average staff satisfaction: {satisfaction_staff}.")
print(f"Average cleanliness satisfaction: {satisfaction_cleanliness}.")
print(f"Average food satisfaction: {satisfaction_food}.")
print(f"Average comfort satisfaction: {satisfaction_comfort}.")
print(f"Average communication satisfaction: {satisfaction_communication}.")


# In[101]:


# Logistic Regression

import sklearn.linear_model

X = df['OverallSatisfaction'].values.reshape(-1,1)
Y = df[' Readmission']

log_reg = sklearn.linear_model.LogisticRegression().fit(X, Y)


# In[93]:


# Correlation coefficient 

correlation_coefficient = log_reg.coef_[0][0]


# In[51]:


#Plot

import matplotlib.pyplot as plt

plt.plot(X, log_reg.predict(X), label = "Regression Line", color= "blue")
plt.scatter(df['OverallSatisfaction'], df[' Readmission'], color= "red")
plt.show()


# In[53]:


#Comparison
num_readmitted_1 = np.sum(df[' Readmission'])

if num_readmitted_1 > num_readmitted_2:
    print("The number of patients readmitted to hospital 1 is greater than that of hospital 2.")
elif num_readmitted_1 < num_readmitted_2:
    print("The number of patients readmitted to hospital 1 is lower than that of hospital 2.")
else:
    print("The satisfaction rate for both hospitals is the samne.")


# In[89]:


print(f"The number of readmission patients is {Readmission}.")


# In[90]:


print(f"Number of patients readmitted: {num_readmitted_2}.")


# In[ ]:




