# Pre-proceso: Selecting variables for IDS

import numpy          as np
import inform_gain    as ig
import redundancy_svd as rsvd

# Load Parameters
def load_config():
    ...
    return()

# Load data 
def load_data():
	...
	return()

# selecting variables
def select_vars():
	...
	return()

#save results
def save_results():
    ...
    return

#-------------------------------------------------------------------
# Beginning ...
def main():
    param        = load_config()            
    X            = load_data()   
    [gain idx  V]= select_vars(X,param)                 
    save_results(gain,idx,V)
       
if __name__ == '__main__':   
	 main()

#-------------------------------------------------------------------
