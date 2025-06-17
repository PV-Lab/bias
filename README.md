# bias: Bayesian Inference with accelerated sampling
### Developed by Dr. Basita Das
This code is used to perform parameter estimation from experimental JV data of solar cells. 

# File download 
1. Clone the code from GitHub
2. Download **Dataset** and **Results** directory from https://osf.io/jcqke/?view_only=6c63f45e6097491e9625688aa816b794.
3. Put the two folders inside the directory where the rest of the code is. **Code won't run until you download this two folder and put then inside the directory**

Example: 
If **bias** is the main directory where you want to put everything, then **bias** should have the following files/Folders
1. Dataset : directory for NN training data
2. Cubic_Deg_data : directory for expeimental data
3. Results : directory for results
4. bias.py
6. de.py
7. de_snooker.py
8. h5.py
9. NN_training.py
10. run_bias.py

# Code environment
To run the code one need to create a python environment with the following libraries:
install miniforge from Releases · conda-forge/miniforge
Create a file called foo.yml and paste the following in it 
name: biasenv
channels:
  - conda-forge
  - defaults
dependencies:
  - pymoo~=0.6.0.1
  - tensorflow~=2.10.0
  - addict
  - emcee~=3.1.6
  - python~=3.10
  - corner~=2.2.2
  - joblib~=1.3.2
  - scikit-learn~=1.3.2
  - statsmodels~=0.14.4
  - mpmath~=1.3.0
Open a  miniforge promt (search it in the task bar) and then in that prompt window navigate to the directory where foo.yml file is located. And then run this command:
'mamba env create -f foo.yml' to create the environment


# How to run and save results
The code runs from **run_bias.py** and saves all results from the current run in a subfolder with a time stamp inside the **Results** folder. All data is saved and graphs corner plot are generated automatically.
During every run the following files/folders are generated in a :
If NN training is performed
1. "**_timestamp__simname_**_trained_model.h5" stores the weights and biases trained NN model. 
2. "**_timestamp__simname_**_train_test.h5" stores the data used for training and testing the NN netwrok.
3. "**_timestamp__simname_**_scaler.joblib" stores the scaler used to perform transformation of the training data.
4. A new folder is created "**_timestamp__simname_**" which saves the corner plot as well as the MCMC chains generated from the current run.

# Steps to run the code without training the NN
It is possible to run the code without training the NN surrogae model. A pre-trained NN has been provided in **20220826-040829one_diode.zip**
1. Unzip the zip file **20220826-040829one_diode.zip**
2. The folder **20220826-040829one_diode.zip** contains 3 files required to run a the code without training the NN surrogate model:

    (a) A hdf5 file **20220826-040829one_diode_trained_model.h5** which has the weight and biases of the trained model.

    (b) A hdf5 file **20220826-040829one_diode_trained_model.h5** which contains the train and test data needed to verify if the NN is working correctly.

    (c) A scaler file **20220826-040829scaler.joblib** containing the scalers used for data transformation, required to inverse transform the data at the end of the run.
3. Open the **run.py** file in an editor and change **_nn_trained_** to **yes**. The _elif_ statement will be executed. The Paths to the files for this have already been set to the three files mentioned above for the code to run without training the NN.


