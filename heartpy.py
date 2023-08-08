#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from dvclive import Live
import pickle
import json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import dvc.api
import sys
import mlem.api


####################################################### DVC PARAMS #############################################################
params = dvc.api.params_show()
params_test_size = params['test_size']
params_n_estim = params['n_estimators']
####################################################### DVC PARAMS #############################################################
#save_dvc_exp=True 

'''
def read_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Failed to read data from {file_path}: {str(e)}")
        return None
'''


excel_path = "C:/Users/sbittoun/Documents/main_fold/dvc_fold_2/heart2.csv"
#excel_path = "C:/Users/sbittoun/Documents/main_fold/creditcard.csv"

data = pd.read_csv(excel_path)


if data is not None:
        print("Data read successfully.")
        # Example: display the first few rows of the filtered data
        #print(data.head(5))
        
############################################ heart disease or ATTACK PREDICTION #########################################
        
        X = data.drop("HeartDiseaseorAttack", axis=1)  # Features
        y = data["HeartDiseaseorAttack"]  # Target column
        '''
        X = data.drop("class", axis=1)  # Features
        y = data["class"]  # Target column
        '''

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=params_test_size, random_state=42)

        
        # Random forest training
        model = RandomForestClassifier(n_estimators=params_n_estim)        
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        '''
        #print(train_y, "\n\n")
        print(test_y, "\n\n")
        print(test_X,"\n\n")
        print(predictions,"\n\n")
        '''
        
else:
    print("Failed to read data.")

    
############################################ heart disease or ATTACK PREDICTION #########################################

'''
######################################## COST AND LOSS FUNCTIONS #######################################################

# Calculer la fonction de coût (cost function) - log loss
cost = log_loss(test_y, predictions, labels=np.unique(test_y))

# Calculer la fonction de perte (loss function) - mean absolute error
loss = mean_absolute_error(test_y, predictions)

# Afficher les valeurs de la fonction de coût et de la fonction de perte
print("Cost function:", cost)
print("Loss function:", loss)
######################################## COST AND LOSS FUNCTIONS #######################################################
'''
'''

############################################### CONFUSION MATRIX AND METRICS ###########################################
# Create the confusion matrix
cm = confusion_matrix(test_y, predictions)

# Inversion des positions des classes dans la matrice de confusion
inv_con = cm[::-1, ::-1]

# Convert the confusion matrix to a DataFrame for easier visualization
cm_df = pd.DataFrame(inv_con, index=['Actual Class 1', 'Actual Class 0'], columns=['Predicted Class 1', 'Predicted Class 0'])

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
'''


# Calculate precision
precision_train = precision_score(train_y, model.predict(train_X))
precision = precision_score(test_y, predictions)

# Calculate recall
recall_train = recall_score(train_y, model.predict(train_X))
recall = recall_score(test_y, predictions)

#Calculate Accuracy
accuracy_train = accuracy_score(train_y, model.predict(train_X))
accuracy = accuracy_score(test_y, predictions)

#Calculate f1_score
f1_train = f1_score(train_y, model.predict(train_X))
f1 = f1_score(test_y,predictions)

#Calculate ROC_AUC
roc_auc_train = roc_auc_score(train_y, model.predict(train_X))
roc_auc = roc_auc_score(test_y, predictions)

print("\nPrecision_train:", precision_train,"\t\tPrecision_test:\t",precision,"\t\tPrecision_diff:\t",precision_train-precision)
print("Recall_train:\t", recall_train,"\t\tRecall_test:\t",recall,"\t\tRecall_diff:\t",recall_train-recall)
print("Accuracy_train:\t",accuracy_train,"\t\tAccuracy_test\t" ,accuracy,"\t\tAccuracy_diff:\t",accuracy_train-accuracy)
print("F1_train:\t",f1_train,"\t\tF1_test:\t" ,f1,"\t\tF1_diff:\t",f1_train-f1)
print("ROC_AUC_train:\t",roc_auc_train,"\t\tROC_AUC_test:\t" ,roc_auc,"\t\tROC_AUC_diff:\t",roc_auc_train-roc_auc,"\n" )


############################################### CONFUSION MATRIX AND METRICS #############################################


'''

################################################## PLOTS ####################################################################
# Plot the actual values with a specific color
plt.figure(figsize=(20, 6))
sns.scatterplot(x=test_y.index, y=test_y, color='blue', label='Actual')
plt.xlabel('Index')
plt.ylabel('Actual Values')

# Plot the predicted values with a specific color
sns.scatterplot(x=test_y.index, y=predictions, color='red', label='Predicted')

plt.title('Result Curve')
plt.legend()

# Specify the start and end indices for the portion to display
start_index = 0
end_index = 2500

# Limit the x-axis to the specified portion
plt.xlim(start_index, end_index)
plt.show()
'''
###################################################################################################################################################

############################################## DVC LOG_METRICS ####################################################################################
with Live() as live:
    live.log_metric("recall", recall)
    live.log_metric("precision", precision)
    live.log_metric("accuracy", accuracy)
    live.log_metric("f1_score", f1)
    live.log_metric("roc_auc", roc_auc)
############################################## DVC LOG_METRICS ####################################################################################

########################################### SAVE METRICS ##########################################################################################

scores = {"accuracy":accuracy,'recall': recall, 'precision':precision, 'f1_score':f1, 'roc_auc':roc_auc,'n_estims':params_n_estim}
with open ('scores.json', 'w')as file:
    json.dump(scores, file)

scores = [{"n_estimators": params_n_estim, "precision": precision, "recall": recall, "accuracy": accuracy,"f1_score":f1, "roc_auc":roc_auc}]

# The 'a-mode' : writes metrics at the end of the file 
with open('scores.csv', 'a', newline='') as file:
    #Creates a writer to write in the csv file
    writer = csv.DictWriter(file, fieldnames=["n_estimators", "precision", "recall", "accuracy","f1_score","roc_auc"])

    # Check if the is empty and add the header
    file.seek(0, 2)
    if file.tell() == 0:
        writer.writeheader()

    # Writes scores in a file
    writer.writerows(scores)
###################################################################################################################################################

################################################ MODEL ############################################################################################
excel_path2 = "C:/Users/sbittoun/Documents/main_fold/dvc_fold_2/heart3.csv"
data2 = pd.read_csv(excel_path)


with open('modeli.pkl', 'wb') as file:
    pickle.dump(model, file)

print('/n########################################/n')
mlem.api.save(model,"save_model")
mlem.api.apply(model,data,method='predict')

'''
# Load the model from the file
with open('modeli.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print(loaded_model)
print(type(loaded_model))
sys.exit()
#################################### PREDICTIONS WITH ANOTHER MODEL ############################################
################################################################################################################

new_excel_path = "C:/Users/sbittoun/Documents/main_fold/dvc_fold_2/heart2.csv"
new_data = pd.read_csv(new_excel_path)
if new_data is not None:
        print("Data read successfully.")
        # Example: display the first few rows of the filtered data
        print(new_data.head(5))

new_predictions = loaded_model.predict(new_data)
'''
