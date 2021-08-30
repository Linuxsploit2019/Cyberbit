from scipy.sparse import data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.decomposition import PCA 
from yellowbrick.model_selection import validation_curve 
import matplotlib as plt 
import pandas as pd    
import numpy as np    



dataset1 = pd.read_csv("phishing.csv") 
dataset2 = pd.read_csv("4.phishing.csv")   


fin_data = pd.concat([dataset1, dataset2], axis = 1)   
# Divide into attributes and label (prepare for training)

x = fin_data.iloc[:, 0:4].values   
y = fin_data.iloc[:,4].values   

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=0) 

sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# Training the algorithm  
classifier = RandomForestClassifier(bootstrap = True, max_depth = 4, max_features = 3, min_samples_leaf = 4,min_samples_split = 8, n_estimators = 100) 
# grid_param = { 
#     "bootstrap" : [True], 
#     "n_estimators" : [100],  
#     "max_depth" : (1,2,3,4), 
#     "min_samples_leaf" : (3,4,5), 
#     "min_samples_split" : (8, 10, 12), 
#     "max_features" : [2, 3]
# }

# grid_search = GridSearchCV(estimator = classifier, param_grid = grid_param, cv = 5, n_jobs = -1, verbose = 3) 
classifier.fit(x_train, y_train) 
# print(grid_search.best_params_) 
y_pred = classifier.predict(x_test) 

# Evaluating the algorithm   


print(accuracy_score(y_test, y_pred))      
print(classification_report(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))
# Validation curve for visualization 
print(validation_curve(classifier, x, y, param_name = "max_depth", n_jobs = -1, param_range = np.arange(1,11)))        




    



























































    

































































