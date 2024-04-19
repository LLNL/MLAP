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


print_interactive_command = False
print_sbatch_command = True
run_interactively = False
submit_job    = False


# In[ ]:


exempt_flag = ''#'--qos=exempt'


# ## Simulation Directory

# In[ ]:


sim_dir = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI'


# ## `sbatch` Scripts

# In[ ]:


sbatch_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/SimulationScripts/sbatch_script_extract.sh'
sbatch_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/SimulationScripts/sbatch_script_prep.sh'
sbatch_script_train = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/SimulationScripts/sbatch_script_train.sh'


# ## `python` Scripts

# In[ ]:


python_script_extract = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/Step1_ExtractData/Extract_DFM_Data.py'
python_script_prep = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/Step2_PrepareData/Prepare_TrainTest_Data.py'
python_script_train = '/g/g92/jha3/Codes/Wildfire_ML/MachineLearningAutomationPipleline/Step3_TrainModel/TrainModel.py'


# ## `json` Input Files

# In[ ]:


json_extract_base = os.path.join(sim_dir, 'InputJson/Extract/json_extract_data')
json_prep_base = os.path.join(sim_dir, 'InputJson/Prep/json_prep_data_label')
json_train_base = os.path.join(sim_dir, 'InputJson/Train/json_train_model')


# In[ ]:


#json_extract_counts = [39]
json_extract_counts = range(79, 94)
json_prep_counts = [7] #[1, 2, 3]
#json_prep_counts = [1, 2, 4] #[1, 2, 3]
json_train_counts = [13]


# ## Generate and Execute `command`

# In[ ]:


def get_commands (exempt_flag, sbatch_script, base_command):
    run_command = 'python {}'.format(base_command)
    sbatch_submit_command = 'sbatch {} {} {}'.format(                                exempt_flag, sbatch_script, base_command)
    
    return run_command, sbatch_submit_command
    


# In[ ]:


def print_and_execute (print_interactive_command, print_sbatch_command,                        run_interactively, submit_job,                        run_command, sbatch_submit_command):
    if (print_interactive_command):
        print('\n', run_command)
    if (print_sbatch_command):
        print('\n', sbatch_submit_command)
    if (run_interactively):
        os.system (run_command)
    if (submit_job):
        os.system (sbatch_submit_command)


# In[ ]:


for data_count in json_extract_counts:
    json_extract = '%s_%03d.json'%(json_extract_base, data_count)
    #print(json_extract)
    if (action == 'Extract'):
        base_command = '{} {}'.format(python_script_extract,
                                      json_extract)
        run_command, sbatch_submit_command = get_commands (                                    exempt_flag, sbatch_script_extract, base_command)
        print_and_execute (print_interactive_command, print_sbatch_command,                            run_interactively, submit_job,                            run_command, sbatch_submit_command)
        continue
        
    for label_count in json_prep_counts:
        json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
        #print(json_prep)
        if (action == "Prep"):
            base_command = '{} {} {}'.format(python_script_prep,
                                             json_extract,
                                             json_prep)
            run_command, sbatch_submit_command = get_commands (                                        exempt_flag, sbatch_script_prep, base_command)
            print_and_execute (print_interactive_command, print_sbatch_command,                                run_interactively, submit_job,                                run_command, sbatch_submit_command)
            continue
            
        for train_count in json_train_counts:
            json_train   = '%s_%03d.json'%(json_train_base, train_count)
            #print(json_train)
            if (action == "Train"):
                base_command = '{} {} {} {}'.format(python_script_train,
                                                    json_extract,
                                                    json_prep,
                                                    json_train)
                run_command, sbatch_submit_command = get_commands (                                        exempt_flag, sbatch_script_train, base_command)
                print_and_execute (print_interactive_command, print_sbatch_command,                                    run_interactively, submit_job,                                    run_command, sbatch_submit_command)
                continue

