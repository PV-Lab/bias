# bias: Bayesian Inference with accelerated sampling
### Developed by Dr. Basita Das
This code is used to perform parameter estimation from experimental JV data of solar cells. 

# File download 
1. Clone the code from GitHub
2. Download **Dataset** from [Zenodo](https://zenodo.org/records/15875229?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjZjRkODM5LWMxMzktNDQzNy05NzQ1LTg2NmU0YjQ2M2RmMyIsImRhdGEiOnt9LCJyYW5kb20iOiJlNGE5NTBmMzhjODZmNTY1YzEzOGMyYmZhODNiNDhlYSJ9.gWw0G0oTJHrPNScyKjJIQEbZ_jfEdduDouB2aik_7X0fKmlox4OLMF4RtfSwqSYmUmc54bmwEt4rRvZ8-S2U2A)
4. Put the three folders **Dataset**, **Experimental_data**, and **Results** directly inside the folder **bias** where the rest of the code is.
**Code won't run until you download this two folder and put then inside the directory**

Example: 
If **bias** is the main directory where you want to put everything, then **bias** should have the following files/Folders
1. Dataset : directory for NN training data - to be downloaded from Zenodo
2. Results : directory for results - to be downloaded from Zenodo
3. Experimental_data : directory for experimental data - to be downloaded from Zenodo
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
  - mpmath~=1.3.0\
Open a  miniforge promt (search it in your task bar) and then in that prompt window navigate to the directory where foo.yml file is located. \
And then run this command:
'mamba env create -f foo.yml' to create the environment


# How to run and save results
The code runs from **run_bias.py** and saves all results from the current run in a subfolder with a time stamp inside the **Results** folder. All data is saved and graphs corner plot are generated automatically.
During every run the following files/folders are generated in a :
If NN is not trained, nn_trained = "No"
1. "**timestamp_NN_trained_model.h5**" stores the weights and biases trained NN model. 
2. "**_timestamp_NN_train_test.h5**" stores the data used for training and testing the NN netwrok.
3. "**_timestamp_NN_scaler.joblib**" stores the scaler used to perform transformation of the training data.
4. A new folder is created "**timestamp_training**" which saves the NN training

# Steps to run the code without training the NN
It is possible to run the code without training the NN surrogae model. A pre-trained NN has been provided in **Results/Training**(download from zenodo)
1. To run the code with the pre-tranied NN, download the data and trained NN files availble in Zenodo
2. Make sure the variable **nn_trained='yes'** in file "**run_bias.py**"
3. Then run the code from **run_bias.py**

# checking


