#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
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
transformer = ColumnTransformer(transformers = [('scaler',MinMaxScaler(feature_range = (0,1)),columns_to_scale)],
                                remainder = 'passthrough')
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

#Picking the best estimator

#Naive Bayes - mean score = 78% - report score = 89%
from sklearn.naive_bayes import GaussianNB
naives = GaussianNB()
naives_score = cross_val_score(naives,X_train, y_train,cv = 10)
print(np.mean(naives_score))

naives.fit(X_train, y_train)
y_pred = naives.predict(X_test)
print(classification_report(y_test,y_pred))

#KNeighbors - mean score = 82% - report score = 80%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
params_grid = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(knn,params_grid,cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

knn = KNeighborsClassifier(n_neighbors = 9)
knn_score = cross_val_score(knn,X_train, y_train,cv = 10)
print(np.mean(knn_score))

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

#SVC - mean score = 84% - report score = 90%
from sklearn.svm import SVC
svc = SVC()
params_grid = {'C':[0.1,1,10,100],
               'kernel':['linear','rbf','sigmoid'],
               'gamma':[0.1,1,10,100]}
grid_search =GridSearchCV(svc,params_grid, cv = 10)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

svc = SVC(C = 10,kernel = 'linear',gamma = 0.1)
svc_score = cross_val_score(svc,X_train, y_train,cv = 10)
print(np.mean(svc_score))

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test,y_pred))

#Decision Tree mean score = 78% - report score = 70%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
params_grid = {'criterion':['gini','entropy'],
               'max_depth':[1,2,3,4,5,6,7,8,9,10],
               'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],
               'min_samples_split':[2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(tree,params_grid,cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 1,min_samples_leaf = 1,
                              min_samples_split = 2)
tree_score = cross_val_score(tree,X_train,y_train, cv = 10)
print(np.mean(tree_score))

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test,y_pred))

#Implement PCA to reduce the features numbers and see if we can get better results

#Scaling the features
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
transformer = ColumnTransformer(transformers = [('scaler',MinMaxScaler(feature_range = (0,1)),columns_to_scale)],
                                remainder = 'passthrough')
X = transformer.fit_transform(X)

#Looking for the best number of components
pca = PCA()
pca.fit(X)
plt.bar(range(0,pca.n_components_),pca.explained_variance_ratio_)
plt.show()

pca = PCA(n_components = 7)
pca.fit(X)
X = pca.transform(X)

#Data split with the pca transformation
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

#Trying the estimators again with the dimension reduced

#Naive Bayes - mean score = 79% - report score = 79%
from sklearn.naive_bayes import GaussianNB
naives = GaussianNB()
naives_score = cross_val_score(naives,X_train, y_train,cv = 10)
print(np.mean(naives_score))

naives.fit(X_train, y_train)
y_pred = naives.predict(X_test)
print(classification_report(y_test,y_pred))

#KNeighbors - mean score = 80% - report score = 84%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
params_grid = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(knn,params_grid,cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

knn = KNeighborsClassifier(n_neighbors = 7)
knn_score = cross_val_score(knn,X_train, y_train,cv = 10)
print(np.mean(knn_score))

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

#SVC - mean score = 83% - report score = 77%
from sklearn.svm import SVC
svc = SVC()
params_grid = {'C':[0.1,1,10,100],
               'kernel':['linear','rbf','sigmoid'],
               'gamma':[0.1,1,10,100]}
grid_search =GridSearchCV(svc,params_grid, cv = 10)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

svc = SVC(C = 100,kernel = 'sigmoid',gamma = 0.1)
svc_score = cross_val_score(svc,X_train, y_train,cv = 10)
print(np.mean(svc_score))

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test,y_pred))

#Decision Tree mean score = 82% - report score = 84%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
params_grid = {'criterion':['gini','entropy'],
               'max_depth':[1,2,3,4,5,6,7,8,9,10],
               'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],
               'min_samples_split':[2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(tree,params_grid,cv = 10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 2,min_samples_leaf = 1,
                              min_samples_split = 2)
tree_score = cross_val_score(tree,X_train,y_train, cv = 10)
print(np.mean(tree_score))

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test,y_pred))