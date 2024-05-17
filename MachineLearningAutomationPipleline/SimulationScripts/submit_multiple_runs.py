#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python submit_multiple_runs.ipynb

# ## Import Packages

# In[ ]:


import os
import sys
import os.path as path
import json


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_simulate = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Simulate/json_simulate_000.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_simulate = sys.argv[1]


# ### Load the JSON file for simulation

# In[ ]:


print('Loading the JSON file for simulation: \n {}'.format(json_file_simulate))


# In[ ]:


with open(json_file_simulate) as json_file_handle:
    json_content_simulate = json.load(json_file_handle)


# ## Action To Be Taken

# In[ ]:


action = json_content_simulate['action']


# ## Execution Options

# In[ ]:


execution_options = json_content_simulate['execution_options']
print_interactive_command = execution_options['print_interactive_command']
print_sbatch_command = execution_options['print_sbatch_command']
run_interactively = execution_options['run_interactively']
submit_job    = execution_options['submit_job']


# In[ ]:


exempt_flag = json_content_simulate['exempt_flag'] #'--qos=exempt'


# ## Simulation Directory

# In[ ]:


sim_dir = json_content_simulate['paths']['sim_dir']


# ## `sbatch` Scripts

# In[ ]:


sbatch_scripts = json_content_simulate['paths']['sbatch_scripts']


# In[ ]:


sbatch_script_extract = os.path.join(sbatch_scripts['base'], sbatch_scripts['extract'])
sbatch_script_prep = os.path.join(sbatch_scripts['base'], sbatch_scripts['prep'])
sbatch_script_train = os.path.join(sbatch_scripts['base'], sbatch_scripts['train'])


# ## `python` Scripts

# In[ ]:


python_scripts = json_content_simulate['paths']['python_scripts']


# In[ ]:


python_script_extract = os.path.join(python_scripts['base'], python_scripts['extract'])
python_script_prep = os.path.join(python_scripts['base'], python_scripts['prep'])
python_script_train = os.path.join(python_scripts['base'], python_scripts['train'])


# ## `json` Input Files

# In[ ]:


json_base = json_content_simulate['paths']['json_base']


# In[ ]:


json_extract_base = os.path.join(sim_dir, json_base['extract'])
json_prep_base = os.path.join(sim_dir, json_base['prep'])
json_train_base = os.path.join(sim_dir, json_base['train'])


# ## `json` Collections

# In[ ]:


collection_options = json_content_simulate['collection_options']
json_extract_counts = collection_options['json_extract_counts']
json_prep_counts = collection_options['json_prep_counts']
json_train_counts = collection_options['json_train_counts']


# In[ ]:


json_extract_counts, json_prep_counts, json_train_counts


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

