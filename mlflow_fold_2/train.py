import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from dvclive import Live
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import yaml
import time

with open('mlproject.yaml', 'r') as stream:
    try:
        mlproject_config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)



######################################################################################################################
################################# MLFLOW STARTING PARAMETERS #########################################################

# Experiment name
experiment_name =  mlproject_config['entry_points']['main']['parameters']['experiment_name']['default']
experiment = mlflow.get_experiment_by_name(experiment_name)
print(experiment_name)

if experiment is None:
    # Create a new experiment if it doesn't exist
    id_experiment = mlflow.create_experiment(experiment_name)
    print("The experiment '",experiment_name,"' doesn't exist. Experiment created :",id_experiment)
else:
    # Set the active experiment if it already exists
    id_experiment = experiment.experiment_id
    print("The experiment '",experiment_name,"' allready exists: ",id_experiment)

# Set the active experiment
mlflow.set_experiment(experiment_name)
name_run =  mlproject_config['entry_points']['main']['parameters']['run_name']['default']
mlflow.start_run(run_name = name_run, experiment_id =id_experiment)
print("Running :",name_run,"in [",experiment_name,";",id_experiment,"]\n")

################################# MLFLOW STARTING PARAMETERS ##########################################################
#######################################################################################################################
def read_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Failed to read data from {file_path}: {str(e)}")


excel_path =  mlproject_config['entry_points']['main']['parameters']['data_path']['default']
data = pd.read_csv(excel_path)

if data is not None:
        print("Data read successfully.")
        # Example: display the first few rows of the filtered data
        #print(data.head(5))

#######################################################################################################################
############################################ HEART DISEASEOR ATTACK PREDICTION ########################################
        target_name =  mlproject_config['entry_points']['main']['parameters']['target']['default']
        X = data.drop(target_name, axis=1)  # Features
        y = data[target_name]  # Target column
        params_test_size =  mlproject_config['entry_points']['main']['parameters']['test_size']['default']
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=params_test_size, random_state=42)



        start_time = time.time()
        print('\n######## Timer starts #########')

        

        # Random forest training

        #The number of trees in the forest.
        estims =  mlproject_config['entry_points']['main']['parameters']['n_estimators']['default']
        model = RandomForestClassifier(n_estimators = estims)        
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
        '''
        # Specify the path of the folder you want to create
        folder_path = "C:/Users/sbittoun/Documents/main_fold/mlflow_fold_2/mlruns/models/RF"
        folder_path2 = "C:/Users/sbittoun/Documents/main_fold/mlflow_fold_2/mlruns/models/newModel"
        # Check if the folder doesn't already exist
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            print("Folder created successfully.")
            mlflow.sklearn.save_model(model, folder_path)

        else:
            print("The folder already exists. Creates a new folder")
            os.makedirs(folder_path2)
            print("Folder created successfully.")
            mlflow.sklearn.save_model(model, folder_path2)
        '''
        
        print(test_y, "\n\n")
        print(test_X,"\n\n")
        print(predictions,"\n\n")
        
else:
    print("Failed to read data.")
    

############################################ HEART DISEASEOR ATTACK PREDICTION ########################################
#######################################################################################################################


#######################################################################################################################
######################################## COST AND LOSS FUNCTIONS ######################################################

# (cost function) - log loss
cost = log_loss(test_y, predictions, labels=np.unique(test_y))

#  (loss function) - mean absolute error
loss = mean_absolute_error(test_y, predictions)

# Afficher les valeurs de la fonction de coût et de la fonction de perte
print("Cost function:", cost)
print("Loss function:", loss)
######################################## COST AND LOSS FUNCTIONS ######################################################
#######################################################################################################################



#######################################################################################################################
#################################### CONFUSION MATRIX, PRECISION, RECALL ##############################################
# Create the confusion matrix
cm = confusion_matrix(test_y, predictions)
inv_con = cm[::-1, ::-1]

# Convert the confusion matrix to a DataFrame for easier visualization
cm_df = pd.DataFrame(inv_con, index=['Actual Class 1', 'Actual Class 0'], columns=['Predicted Class 1', 'Predicted Class 0'])

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')


# Calculate precision
precision = precision_score(test_y, predictions)

# Calculate recall
recall = recall_score(test_y, predictions)

# Calculate accuracy
accuracy = accuracy_score(test_y, predictions)

print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy", accuracy,"\n")

###################################### CONFUSION MATRIX, PRECISION, RECALL ############################################
#######################################################################################################################


end_time = time.time()
# Execution time in seconds
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

#######################################################################################################################
################################################ MLFLOW METRICS #######################################################

mlflow.log_metric("precision", precision) #metric logging
mlflow.log_metric("recall", recall) #metric logging
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("time", execution_time)
mlflow.log_param("test_size", params_test_size)
mlflow.log_param("n_estimators", estims)
mlflow.log_param("target",target_name)
#mlflow.sklearn.log_model(model, "model") #model logging

################################################ MLFLOW METRICS #######################################################
#######################################################################################################################







#######################################################################################################################
################################################## PLOTS ##############################################################
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

#######################################################################################################################
#######################################################################################################################

mlflow.end_run()