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
#We have almost the same kind of age distribution for both sex like the positive ones
men_age_negative = dataset[(dataset['sex'] == 1) & (dataset['target'] == 0)]['age']
sns.distplot(men_age_negative,kde = False)

women_age_negative = dataset[(dataset['sex'] == 1) & (dataset['target'] == 0)]['age']
sns.distplot(women_age_negative,kde = False,bins = 20)

#Plotting the cholesterol and the heart rate, it seems that high cholesterol and heart rate
#Is almost the same for both people with heart desease and not.
sns.scatterplot('chol','thalach',hue = 'target',data = dataset)

#Plotting the resting blood pressure and the heart rate
sns.scatterplot('trestbps','thalach',hue = 'target',data = dataset)

#Trying different plot kinds to try to make more sense of the data
#It looks the that the heart rate is the most correlated feature to the target
sns.catplot('target','chol',kind = 'swarm',data = dataset)
sns.catplot('target','thalach',kind = 'swarm',data = dataset)
sns.catplot('target','trestbps',kind = 'swarm',data = dataset)

#Plotting the other features with respect to the target
sns.countplot('cp',hue = 'target',data = dataset)
sns.countplot('fbs',hue = 'target',data = dataset)
sns.countplot('restecg',hue = 'target',data = dataset)
sns.countplot('exang',hue = 'target',data = dataset)
sns.countplot('slope',hue = 'target',data = dataset)
sns.countplot('ca',hue = 'target',data = dataset)
sns.countplot('thal',hue = 'target',data = dataset)

#Checking the correlation of the features, a few features are corralated but not too high
sns.heatmap(dataset.corr(),annot = True)