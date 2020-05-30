#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

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

#OneHotEncoding the categorical features
columns_ = ['cp','fbs','restecg','exang','slope','ca','thal']
X = pd.get_dummies(dataset.drop(['target'],axis = 1),columns = columns_,drop_first = True)

#Gettiing my targets
y = dataset.iloc[:,-1]

#Splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

#Scaling the features
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
transformer = ColumnTransformer(transformers = [('scaler',StandardScaler(),columns_to_scale)],
                                remainder = 'passthrough')
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

#Picking the best estimator

#Naive Bayes - means score - 76% - report score 85%
from sklearn.naive_bayes import GaussianNB
naives = GaussianNB()
naives_score = cross_val_score(naives,X_train, y_train,cv = 10)
print(np.mean(naives_score))

naives.fit(X_train, y_train)
y_pred = naives.predict(X_test)
print(classification_report(y_test,y_pred))