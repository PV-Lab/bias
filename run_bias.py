import NN_training
import bias
from pathlib import Path
import datetime
import os

# If you want to use a trained neural network, set nn_trained to 'yes'. If you want to train a new neural network, set it to 'no'.

nn_trained = 'yes' # 'yes', 'no'

# Define the paths for the directories and dataset
dir_path = os.getcwd()  # Current working directory
Results_path = Path(dir_path) / "Results"  # Results directory where all results will be saved
dataset = Path("./Dataset/0102025_JV_226580_dataset_8_parameters.h5")  # Dataset for the NN training
exp_path = Path("./Experimental_data/0102025_exp_JV_dataset_12_1_065_025_sun.h5")      # Experimental data path



# Extract the last part of the exp_path and create a file name
h5_name = exp_path.stem 
last_six_chars = h5_name[-8:]
print("Last 8 characters of h5_name:", last_six_chars)
sim_name = "_" + last_six_chars

print("Generated file name:", sim_name)

nwalkers = 512      # Number of parallel samplers for the MCMC
sigma = 1e-4    # Noise level for the experimental data
min_prob = 700  # Minimum probability a combination of parameters must have to be accepted as a valid solution

# Neural Network training

# If nn_trained is 'no', it will train a new neural network and save the model, training data, and scaler.
if nn_trained == 'no':
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path, reg_path,train_path, scaler_path = NN_training.main("Training", timestamp, dataset)
    print('reg_path', reg_path, train_path, scaler_path)
    print('train_path', train_path)
    print('scaler_path', scaler_path)


# if nn_trained is 'yes', it will use the trained model and scaler to perform bias correction on the experimental data.
elif nn_trained == 'yes':
    reg_path = Results_path / 'Training/JV_dataset_8_parameters_trained_model.h5'
    train_path = Results_path / 'Training/JV_dataset_8_parameters_train_test.h5'
    scaler_path = Results_path / 'Training/input_transform_scaler.joblib'

    
bias.main(Results_path, sim_name, reg_path, train_path, scaler_path, exp_path, sigma, min_prob,nwalkers)


