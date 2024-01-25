#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python submit_multiple_runs.ipynb

# ## Import Packages

# In[1]:


import os
import os.path as path


# ## Action To Be Taken

# In[2]:


action = "Extract" # "Extract", "Prep", "Train", "Analyze"


# ## Simulation Directory

# In[3]:


sim_dir = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI'


# ## `sbatch` Scripts

# In[4]:


sbatch_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_extract.sh'
sbatch_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_prep.sh'
sbatch_script_train = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_train.sh'


# ## `python` Scripts

# In[5]:


python_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step1_ExtractData/Extract_DFM_Data.py'
python_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step2_PrepareData/Prepare_TrainTest_Data.py'
python_script_train = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step3_TrainModel/TrainModel.py'


# ## `json` Input Files

# In[6]:


json_extract_base = os.path.join(sim_dir, 'InputJson/Extract/json_extract_data')
#json_extract_counts = [0] #[0, 1]
json_extract_counts = range(15, 39)


# In[7]:


json_extract_counts


# In[8]:


json_prep_base = os.path.join(sim_dir, 'InputJson/Prep/json_prep_data_label')
json_prep_counts = [1, 2] #[1, 2, 3]


# In[9]:


json_train_base = os.path.join(sim_dir, 'InputJson/Train/json_train_model')
json_train_counts = [3, 4]


# ## Generate and Execute `command`

# In[10]:


for data_count in json_extract_counts:
    json_extract = '%s_%03d.json'%(json_extract_base, data_count)
    #print(json_extract)
    if (action == 'Extract'):
        sbatch_submit_command = 'sbatch {} {} {}'.format(                                                      sbatch_script_extract,
                                                      python_script_extract,
                                                      json_extract)
        print(sbatch_submit_command, '\n')
        #os.system(sbatch_submit_command)
        continue
        
    for label_count in json_prep_counts:
        json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
        #print(json_prep)
        if (action == "Prep"):
            sbatch_submit_command = 'sbatch {} {} {} {}'.format(                                                      sbatch_script_prep,
                                                      python_script_prep,
                                                      json_extract,
                                                      json_prep)
            print(sbatch_submit_command, '\n')
            #os.system(sbatch_submit_command)
            continue
            
        for train_count in json_train_counts:
            json_train   = '%s_%03d.json'%(json_train_base, train_count)
            #print(json_train)
            if (action == "Train"):
                sbatch_submit_command = 'sbatch {} {} {} {} {}'.format(                                                      sbatch_script_train,
                                                      python_script_train,
                                                      json_extract,
                                                      json_prep,
                                                      json_train)
            print(sbatch_submit_command, '\n')
            #os.system(sbatch_submit_command)


# In[ ]:




