#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('heart.csv')

#data analysis
dataset.info()
null_values = dataset.isnull().sum()
statistics = dataset.describe().transpose()

#Plotting relatioship

#The men are more likely to have heart disease
sns.countplot('sex',hue = 'target',data = dataset)

#Distribution of men and women age with heart disease

#Men start to have it more likely between the age of 40 and 60
men_age_positive = dataset[(dataset['sex'] == 1) & (dataset['target'] == 1)]['age']
sns.distplot(men_age_positive,kde = False)

#And for women we can say almost the same just that with woman goes as far as 70
women_age_positive = dataset[(dataset['sex'] == 0) & (dataset['target'] == 1)]['age']
sns.distplot(women_age_positive,kde = False)

#Distribution of the ones with no heart desease
#near the 60's is the age when men's suffer it the most
men_age_negative = dataset[(dataset['sex'] == 1) & (dataset['target'] == 0)]['age']
sns.distplot(men_age_negative,kde = False)

#And for women as well, we can say
women_age_negative = dataset[(dataset['sex'] == 1) & (dataset['target'] == 0)]['age']
sns.distplot(women_age_negative,kde = False,bins = 20)