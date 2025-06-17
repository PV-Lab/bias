import NN_training
import bias
from pathlib import Path
import datetime
import os

#exp_dir = Path('./Cubic_Deg_data/')

#h5_files = list(exp_dir.glob("*.h5"))
#print("Found h5 files:", h5_files)

#for h5_file in h5_files:
#    exp_path = exp_dir / h5_file.name
#    print(f"Processing file: {exp_path}")
    
    # The rest of the code will use the updated exp_path



nn_trained = 'yes' # 'yes', 'no'
dataset = Path("./Dataset/params_8_sun_1_for_cubic_output_transformed.h5")
exp_path = Path("./Cubic_Deg_data/Device_EFJ1-169-3_Pixel_1_Sweep_1_degtest_Age_50.0.h5")

# Extract the last part of the exp_path and create a file name
h5_name = exp_path.stem 
last_six_chars = h5_name[-8:]
print("Last 8 characters of h5_name:", last_six_chars)
sim_name = "_" + last_six_chars

print("Generated file name:", sim_name)

nwalkers = 512
sigma = 1e-4

if nn_trained == 'no':
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path, reg_path,train_path, scaler_path = NN_training.main(sim_name, timestamp, exp, dataset)
    print('reg_path', reg_path, train_path, scaler_path)
    print('train_path', train_path)
    print('scaler_path', scaler_path)
elif nn_trained == 'yes':
    dir_path = Path("C:/Users/basit/Codes/BayesMC_04112024/BayesMC/Results/Cubic_degtest")
    reg_path = dir_path/('20250510-211703_cubic_1_sun_sigma=1e-4_trained_model.h5')
    train_path = dir_path/('20250510-211703_cubic_1_sun_sigma=1e-4_train_test.h5')
    scaler_path = dir_path/('20250510-211703_scaler.joblib')

    
#if start_condition == 'random':                
    bias.main(dir_path, sim_name, reg_path, train_path, scaler_path, exp_path, sigma, nwalkers)


