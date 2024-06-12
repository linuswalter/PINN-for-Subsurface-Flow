#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:08:39 2022

@author: Linus Walter
"""

import time
import os
import numpy as np
import random
import sciann as sn
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorflow.python.keras import backend

# %% Class for Experimental Setup
class experiment_setup:
    def __init__(self,delay=1,test_run=False):
        self.dict_experiment_params = {}
        self.delay = delay
        self.test_run = test_run

    def add_param(self,experiment_param=[],fixed_var=None,default_var=0, name=None,doc=None,main_var=False):
        '''0
        fixed var : Int
                    If this Variable is set, it is always chosen
            
        default_var : Int
                    Default value for the variable during the test_run
            
        name :      Str
                    Name of the Variable
        
        doc :      Str, Optional
                   Documentation for the Variable if needed

         main_var: Boolean
                 If a Variable is 'main_var' this information will be indicated in the output dictionary with all variables.
        '''
        self.experiment_param = experiment_param
        # print(experiment_param)

        assert isinstance(experiment_param,list), "Please provide 'experiment_param' as list"
        if name != None: print("============================================================================\n   Name of Variable: {}\n============================================================================".format(name))
        if doc!=None:
            assert isinstance(doc,str), "Please provide 'doc' as string"
            print("     Experiment Doc: {}".format(doc))
        print("Choice of Variables:")
        for i in range(len(experiment_param)): print("                 {} : {}".format(i, experiment_param[i]))

        # allow Custom values if it is about float or integers
        if isinstance(experiment_param[0],(float,int)) and type(experiment_param[0])!=bool :
            print("                 {} : Custom value".format(i+1))

        if self.test_run==False and fixed_var==None:
            experiment_nr = input("     Enter integer : ")
            if len(experiment_nr)==0: self.experiment_param_choice = experiment_param[0]
            #Assign Custom variable
            elif int(experiment_nr)>len(self.experiment_param)-1 and isinstance(experiment_param[0],(int,float)):
                if type(self.experiment_param[0]) == float: custom_val = float(input("Enter custom value : "))
                else: custom_val = int(input("Enter custom value : "))
                assert isinstance(custom_val, type(self.experiment_param[0])), "Please provide a value in the same format as the given values"
                self.experiment_param_choice = custom_val
            else: self.experiment_param_choice = experiment_param[int(experiment_nr)]
        elif fixed_var != None:
            assert isinstance(fixed_var,int), "Please provide 'fixed_var' as integer"
            self.experiment_param_choice = experiment_param[fixed_var]
        else:
            assert isinstance(default_var,int), "Please provide 'default_var' as integer"
            self.experiment_param_choice = experiment_param[default_var]
        print("\n>            Result: {} ".format(self.experiment_param_choice))
        print("\n \n")
        # print("======================================\n \n")
        time.sleep(self.delay)
        self.dict_experiment_params[name]=self.experiment_param_choice
        if main_var==True:
            self.dict_experiment_params["ex_var_name"]=name
            self.dict_experiment_params["ex_var_value"]=experiment_param
        return self.experiment_param_choice

    def save_experiment_params(self,path=""):
        import pickle
        file_path = "{}/experiment_params.pkl".format(path)
        f = open(file_path,"wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.dict_experiment_params,f)
        f.close()
        print("Experiment Parameters saved to Path:\n {}".format(file_path))

    def print_summary(self):
        print("============================================================================\nSummary:\n")
        for i in range(len(self.dict_experiment_params)):
            print("{} = {}".format(list(self.dict_experiment_params.keys())[i],list(self.dict_experiment_params.values())[i]))
        print("\n============================================================================ \n\n")

    def abort_if_x(self):
        #Make code wait to check experimental setup
        print("Check the exeperimental setup.\n Enter 'x' to abort run.")
        i = input("Press enter to continue: ")
        if i == "x": import sys; sys.exit() ; print("Execution aborted")

def np_growing_steps(xmin,xmax,nr,power=2,endpoint=True,growth_dir_neg=False,flip_order=False,symmetric_growth=False,random=False,plot=True,plot_log=False):
    '''
    We create points with a growing spacing. Interally we start with the linspace always at 0 because otherwise we can not apply fractured powers.
    
    Example: x1 = np_growing_steps(200,940,nr=50,power=1.5)
    
    Parameters
    ----------
    xmin : float
        Minimal value
    xmax : float
        Maximum Value
    nr : int
        Number of Points
    power : float
        Power of growth
    endpoint : TYPE, optional
        DESCRIPTION. The default is True.
    growth_dir_neg : bool, optional
        Decision if growth should increase or decrease. The default is False.
    flip_order : bool, optional
        Order the number from largest to smallest.
    symmetric_growth : bool, optional
        Decision if growth should be symmetric. The default is False.
    random : bool, optional
        If yes, the x-array is created randomly with a uniform distribution
    plot : bool, optional
        If yes, we return a plot of the points.
    
    Returns
    -------
    x : Numpy array
        1D-Array of points
    
    '''
    assert xmax - xmin > 0, "xmax has to be larger than xmin"
    x_range = xmax-xmin
    if symmetric_growth==False:
        if random==True: x = np.random.uniform(0,x_range, nr)
        else: x = np.linspace(0,x_range, nr, endpoint=endpoint)
        x_exp = np.power(x,power)
        #Scale back via min/max scaling
        x = x_exp/max(x_exp)*x_range
        x = x + xmin #translate
        if growth_dir_neg==True: x= x*(-1) + (xmax+xmin)
        if flip_order==True: x=np.flip(x)
    else:
        assert nr % 2 != 0, "Please provide an odd number for a symmtric growth array"
        if random==True: x = np.random.uniform(0,x_range/2, int(nr/2)+1)
        else: x = np.linspace(0,x_range/2, int(nr/2)+1, endpoint=endpoint)
        x_exp = np.power(x,power)
        x = x_exp/max(x_exp)*x_range/2
        x = np.unique(np.concatenate((
            np.sort(x*(-1)),
            x)))
        x = x + (xmax - xmin)/2 + xmin  #translate
    # Reverse Growth
    if plot==True: plt.figure(figsize=(10,2)),plt.scatter(x,np.full(x.shape,1),s=10,color="red"), plt.grid()
    if min(x)>=0 and plot_log==True: plt.xscale("log")
    plt.show()
    return x

# %% Save Models

class save_models:
    def __init__(self,main_dir="02_model_output",model_prefix="model",dir_size_tresh=1000):
        """
        This class helps to save each model in a new directory

        Parameters
        ----------
        main_dir : str, optional
            Main directory in which all models are saved, by default "02_model_output"
        model_prefix : str, optional
            Prefix of the individual modeling directories, by default "model_"
        dir_size_tresh : int, optional
            Threshold value for total of modeling directory in bytes, by default 1000 = 1KB
        """
        from datetime import datetime
        t1 = datetime.now()
      
        self.main_dir = main_dir
        # Check if main dir exists
        if os.path.exists(self.main_dir)==False:  os.mkdir(self.main_dir); print("Created Directory: {}".format(self.main_dir))
        #Remove empty Directories
        self.remove_emtpy_dirs(dir_size_tresh)
        #Create new subdirectories:
        self.model_dir = "{}_{}".format(model_prefix,t1.strftime("%Y%m%d_%H%M%S"))
        self.model_path ="{}/{}".format(self.main_dir,self.model_dir)
        # Create list of all model paths
        self.subdir_list = [f.path for f in os.scandir(self.main_dir) if f.is_dir()]
        self.subdir_list.sort()
        if os.path.exists(self.model_path)==False:  os.mkdir(self.model_path); print("Created Directory: {}".format(self.model_path))
        
    def remove_emtpy_dirs(self,dir_size_tresh):
        from pathlib import Path
        import shutil
        # for dir_i in os.listdir("{}/".format(self.main_dir)):
        
        for entry in os.scandir(self.main_dir):
            root_directory = Path(entry.path)
            size = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
            if entry.is_dir() and size < dir_size_tresh:
                shutil.rmtree(entry.path)
                print("Removed Directory {} with size {}".format(root_directory,size))
    
    def get_path(self):
        """
        Returns path of current model
        """
        return self.model_path
                
    def get_model_list(self):
        """
        Returns path of all models in model dir.
        """
        return os.listdir(self.main_dir)
    
    def get_last_model_path(self):
        """
        Returns path of the last/previously saved model
        """
        return self.subdir_list[-1]

# %% os Support funcs

def create_dir(path):
    """
    Summary
    ------
    This function creates a directory under the given path if it
    is not created yet

    Args:
    ------
        path (list): _description_
    """
    import os
    if os.path.exists(path) == False: os.mkdir(path)

def empty_dir(path):
    """
    _summary_
    

    Parameters
    ----------
    path : str
        Example: "test/"
    """
    import os
    for file in os.listdir(path): 
        os.remove("%s%s"%(path,file))

# %% Numpy Support Funcs

class set_seeds:
    """
    This function (re-)sets seeds of the system (os). 
    """
    def __init__(self,my_seed=120):
        self.np_seed = np.random.seed(my_seed)
        self.np_random_state = np.random.RandomState(1)
        
        np.random.uniform(0,10,20)

        rng = np.random.default_rng(12345)
        rng.uniform(0,10,3)

        random.seed(self.np_seed)
        os.environ['PYTHONHASHSEED'] = str(my_seed)

# %% Custom Loss Functions

def mean_absolute_logarithmic_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
  second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
  return backend.mean(backend.abs(first_log-second_log), axis=-1)

def rmsle(y_true, y_pred):
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
  second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
  return backend.sqrt(backend.mean(
      math_ops.squared_difference(first_log, second_log), axis=-1))


# %% Log Scaler

class log_scaler:
    def __init__(self,base=None):
        self.base=base
    def log_transform(self,x):
        x=x+1 #We use x+1 to receive only positive log values
        if self.base==None: return np.log(x)
        else: return np.log(x)/np.log(self.base)
    def inverse_transform(self,x):
        x=x+1 #We use x+1 to receive only positive log values
        #Invserse Transformation for SciANN Functionals
        if isinstance(x,sn.functionals.mlp_functional.MLPFunctional):
            if self.base==None: return sn.exp(x)
            else: sn.pow(self.base,x)
        #Inverse transform any other dataset
        else:
            if self.base==None: return np.exp(x)
            else: return self.base**(x)


def stepwise_learning_rate(lr_start,lr_finish,freq_ep,epochs,plot=False):
    steps = int(epochs/freq_ep)
    y = np.concatenate((np.linspace(lr_start,lr_finish,steps),np.linspace(lr_start,lr_finish,steps)))
    y=-np.sort(-y)
    x = np.concatenate((
        np.linspace(0,epochs,steps+1),
        np.linspace(0,epochs,steps+1)[1:steps]
                        ))
    x.sort()
    if plot==True: 
        plt.close()
        plt.plot(x,y)
        plt.yscale("log"), plt.grid()
        plt.show()
    return [list(x),list(y)]





