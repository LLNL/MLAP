#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python submit_multiple_runs.ipynb

# ## Import Packages

# In[ ]:


import os
import os.path as path


# ## Action To Be Taken

# In[ ]:


action = "Extract" # "Extract", "Prep", "Train", "Analyze"


# In[ ]:


print_command = True
submit_job    = False


# In[ ]:


exempt_flag = '--qos=exempt'


# ## Simulation Directory

# In[ ]:


sim_dir = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI'


# ## `sbatch` Scripts

# In[ ]:


sbatch_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_extract.sh'
sbatch_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_prep.sh'
sbatch_script_train = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/sbatch_script_train.sh'


# ## `python` Scripts

# In[ ]:


python_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step1_ExtractData/Extract_DFM_Data.py'
python_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step2_PrepareData/Prepare_TrainTest_Data.py'
python_script_train = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step3_TrainModel/TrainModel.py'


# ## `json` Input Files

# In[ ]:


json_extract_base = os.path.join(sim_dir, 'InputJson/Extract/json_extract_data')
json_prep_base = os.path.join(sim_dir, 'InputJson/Prep/json_prep_data_label')
json_train_base = os.path.join(sim_dir, 'InputJson/Train/json_train_model')


# In[ ]:


#json_extract_counts = [0]
json_extract_counts = list (set (range(41, 59)) - set([43, 47]))
json_prep_counts = [2] #[1, 2, 3]
json_train_counts = [3, 5]


# In[ ]:


json_extract_counts


# ## Generate and Execute `command`

# In[ ]:


for data_count in json_extract_counts:
    json_extract = '%s_%03d.json'%(json_extract_base, data_count)
    #print(json_extract)
    if (action == 'Extract'):
        sbatch_submit_command = 'sbatch {} {} {} {}'.format(                                                      exempt_flag,
                                                      sbatch_script_extract,
                                                      python_script_extract,
                                                      json_extract)
        if (print_command):
            print('\n', sbatch_submit_command)
        if (submit_job):
            os.system(sbatch_submit_command)
        continue
        
    for label_count in json_prep_counts:
        json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
        #print(json_prep)
        if (action == "Prep"):
            sbatch_submit_command = 'sbatch {} {} {} {} {}'.format(                                                      exempt_flag,
                                                      sbatch_script_prep,
                                                      python_script_prep,
                                                      json_extract,
                                                      json_prep)
            if (print_command):
                print('\n', sbatch_submit_command)
            if (submit_job):
                os.system(sbatch_submit_command)
            continue
            
        for train_count in json_train_counts:
            json_train   = '%s_%03d.json'%(json_train_base, train_count)
            #print(json_train)
            if (action == "Train"):
                sbatch_submit_command = 'sbatch {} {} {} {} {} {}'.format(                                                      exempt_flag,
                                                      sbatch_script_train,
                                                      python_script_train,
                                                      json_extract,
                                                      json_prep,
                                                      json_train)
            if (print_command):
                print('\n', sbatch_submit_command)
            if (submit_job):
                os.system(sbatch_submit_command)

