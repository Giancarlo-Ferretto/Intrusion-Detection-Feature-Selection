# Pre-proceso: Selecting variables for IDS

import numpy          as np
import inform_gain    as ig
import redundancy_svd as rsvd

from sklearn.preprocessing import LabelEncoder # for variables encoding

# Load parameters
def load_config(config_file_path = './src/config/cnf_sv.csv'):
    # Número de Muestras : 10000
    # Top-K de Relevancia : 30
    # Número de Vectores Singulares : 20
    # Clase Normal (s/n) : 1
    # Clase DOS (s/n) : 0
    # Clase Probe (s/n) : 1

    params = np.loadtxt(fname=config_file_path)  
    return params

# Load data
def load_data(file_data_path = './src/data/KDDTrain.txt'):
    data = []
    
    # Read data file
    with open(file_data_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            data.append(fields)
    data = np.array(data)

    data = data[:, :-1] # drop the last column

    # Convert no-numeric vars to numeric
    for i in range(1, 4): # for each no-numerical variable
        no_numerical_var = np.array(data)[:, i] # get the no-numerical variable
 
        label_encoder = LabelEncoder() # initialize the label encoder
        no_numerical_var_encoded = label_encoder.fit_transform(no_numerical_var) # encode the no-numerical variable

        data[:, i] = no_numerical_var_encoded # replace the no-numerical variable with the encoded one

    # Convert the class labels to numeric
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

    class_labels = np.array(data)[:, 41] # get the class labels
    class_labels = np.array([class_mapping.get(label, -1) for label in class_labels], dtype=int) # encode the class labels
    
    data[:, 41] = class_labels # replace the class labels with the encoded ones

    # Normalize each column
    # X = (X - min)/(max - min)*(b - a) + a
    # a = 0.1, b = 0.99
    #a = 0.1
    #b = 0.99
    #
    #for i in range(0, 41): # for each column
    #    column = np.array(data)[:, i]
    #    column = column.astype(float)
    #    max = np.max(column) + 1 # bins max + 1
    #    min = np.min(column)
    #    column = (column - min)/(max - min)*(b - a) + a
    #    data[:, i] = column

    return data

# Selecting variables
def select_vars(X, params):
    # Reorder the data randomly
    np.random.shuffle(X)
    # Randomly select "params[0]" samples
    samples = int(params[0])
    X = X[np.random.choice(X.shape[0], samples, replace=False)]

    # Initialize X, Y
    Y = X[:, -1] # classes
    X = X[:, :-1] # data

    X = X.astype(float)
    Y = Y.astype(float)

    # (Step 1, 2 and 3) Select most relevant variables with information gain
    idx = np.arange(41)
    gain = []
    
    for attribute in range(0, 41):
        gain.append(ig.inform_gain(X[:, attribute], Y)) # calculate the information gain of each variable

    idx = np.array(idx)[np.argsort(gain)[::-1]] # order the variables by information gain (descending)

    # (Step 4) Drop redundant variables with SVD
    V = rsvd.svd_data(X, params)

    return [gain, idx, V]

# Save results
def save_results():
    return

#-------------------------------------------------------------------
# Beginning...
def main():
    params = load_config()
    X = load_data()

    [gain, idx, V] = select_vars(X, params)
    #save_results(gain, idx, V)
       
if __name__ == '__main__':
	 main()

#-------------------------------------------------------------------
