#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import os.path as path


# In[ ]:


sim_dir = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI'


# In[ ]:


sbatch_script = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/SimulationScripts/run_python_from_conda_env_2024_01_07.sh'


# In[ ]:


python_script = '/g/g92/jha3/Codes/Wildfire_ML/SJSU/Step3_TrainModel/TrainModel.py'


# In[ ]:


json_extract_base = os.path.join(sim_dir, 'InputJson/Extract/json_extract_data')
json_extract_counts = [2, 3, 4, 5, 6, 7, 8]


# In[ ]:


json_prep_base = os.path.join(sim_dir, 'InputJson/Prep/json_prep_data_label')
json_prep_counts = [1] #[1, 2, 3]


# In[ ]:


json_train_base = os.path.join(sim_dir, 'InputJson/Train/json_train_model')
json_train_counts = [1, 2, 3, 4, 5]


# In[ ]:


for data_count in json_extract_counts:
    for label_count in json_prep_counts:
        for train_count in json_train_counts:
            json_extract = '%s_%03d.json'%(json_extract_base, data_count)
            json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
            json_train   = '%s_%03d.json'%(json_train_base, train_count)
            #print(json_extract)
            #print(json_prep)
            #print(json_train)
            sbatch_submit_command = 'sbatch {} {} {} {} {}'.format(                                                      sbatch_script,
                                                      python_script,
                                                      json_extract,
                                                      json_prep,
                                                      json_train)
            #print(sbatch_submit_command)
            os.system(sbatch_submit_command)


# In[ ]:




