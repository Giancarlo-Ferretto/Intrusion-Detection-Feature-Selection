# Pre-proceso: Selecting variables for IDS

import numpy          as np
import inform_gain    as ig
import redundancy_svd as rsvd

from sklearn.preprocessing import LabelEncoder # for variables encoding

# File paths
file_data_path = './src/data/KDDTrain.txt' # data file path

# Class mapping dictionary
class_mapping = {
    'normal': 1,
    'neptune': 2,
    'teardrop': 2,
    'smurf': 2,
    'pod': 2,
    'back': 2,
    'land': 2,
    'apache2': 2,
    'processtable': 2,
    'mailbomb': 2,
    'udpstorm': 2,
    'ipsweep': 3,
    'portsweep': 3,
    'nmap': 3,
    'satan': 3,
    'saint': 3,
    'mscan': 3
}

# Load Parameters
def load_config():
    params = []
    return params

# Load data
def load_data():
    data = []
    
    # Read data file
    with open(file_data_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            data.append(fields)
    data = np.array(data)

    # Convert no-numeric vars to numeric
    for i in range(1, 4): # for each no-numerical variable
        no_numerical_var = np.array(data)[:, i] # get the no-numerical variable
 
        label_encoder = LabelEncoder() # initialize the label encoder
        no_numerical_var_encoded = label_encoder.fit_transform(no_numerical_var) # encode the no-numerical variable

        data[:, i] = no_numerical_var_encoded # replace the no-numerical variable with the encoded one

    # Convert the class labels to numeric
    class_labels = np.array(data)[:, 41] # get the class labels
    class_labels = np.array([class_mapping.get(label, label) for label in class_labels]) # encode the class labels

    data[:, 41] = class_labels # replace the class labels with the encoded ones

    return data

# selecting variables
def select_vars(X, params):
	return

#save results
def save_results():
    return

#-------------------------------------------------------------------
# Beginning...
def main():
    params = load_config()
    X = load_data()

    print(X[0])
    print(X[1])
    print(X[2])
    print(X[17])

    #[gain, idx, V] = select_vars(X, params)
    #save_results(gain, idx, V)
       
if __name__ == '__main__':
	 main()

#-------------------------------------------------------------------
