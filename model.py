"""   HR Data analysis.py    """
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

IP_train_data = pd.read_csv('train.csv')
IP_test_data = pd.read_csv('test.csv')


# training dataframe
dum_train_data= IP_train_data.apply(lambda x : x.fillna(x.value_counts().index[0]))

# testing dataframe
dum_test_data= IP_test_data.apply(lambda x : x.fillna(x.value_counts().index[0]))

#columns of traing data
"""
dum_train_data.columns
['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel',
 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won?',
 'avg_training_score', 'is_promoted']

"""

# get the categorical columns
cat_cols = dum_train_data.select_dtypes(['object']).columns

#cat to num for training data
dum_train_data[cat_cols] = dum_train_data[cat_cols].apply(lambda x: pd.factorize(x)[0])

#cat to num for testing data
dum_test_data[cat_cols] = dum_test_data[cat_cols].apply(lambda x: pd.factorize(x)[0])

#importing RandomOverSampler for balancing the data
#from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

#splitting between Xtrain Ytrain from training data with oversampling
Xtrain = dum_train_data.drop(['employee_id', 'recruitment_channel','region','is_promoted'], axis=1)
Ytrain = dum_train_data['is_promoted']

Xresam , yresam = over_sampler.fit_resample(Xtrain,Ytrain)

Xtes = dum_test_data.drop(['employee_id', 'recruitment_channel','region'], axis=1)


#splitting training data into training and validation.
#from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(Xresam, yresam, test_size = 0.2, random_state = 2)

#normalization of data in 0-1 using StandardScaler
#from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
Xtr_norm = sc_x.fit_transform(X_train)
X_validation = sc_x.fit_transform(X_valid)
Xte_norm = sc_x.fit_transform(Xtes)

#use decision tree model to classify the result
"""
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report """

HR_model = DecisionTreeClassifier()
HR_model.fit(Xtr_norm, y_train)

# predection for confusion matrix
y_pred_for_validation = HR_model.predict(X_validation)

#getting confusion matrix

My_conf_metrics = confusion_matrix(y_valid, y_pred_for_validation)

# display confusion matrix
#print(My_conf_metrics)

# seaborn and matplotlib for visualization
#import seaborn as sns
#import matplotlib.pyplot as plt

#corr_mtrtix = input_train_df.corr().round(2)
#sns.heatmap(My_conf_metrics, annot = True,annot_kws={'size': 12},  fmt = '.8g')

metrics.accuracy_score(y_valid, y_pred_for_validation)

# predection for confusion matrix on test data
y_pred_for_testdata = HR_model.predict(Xtes.values)

#getting confusion matrix

#My_conf_metrics = confusion_matrix(y_valid, y_pred_for_testdata)

pickle.dump(HR_model, open('model.pkl','wb'))
#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))
