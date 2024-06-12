#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:14:29 2022

@author: Linus Walter
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class assign_targets_to_input_dim:
    """
    This function interpolates the collocation points for a function f(x) where x can be any space dimension or the time dimension.

    Example Input:
        target_data[0] = (assign_targets_to_input_dim(
            input_dataset = input_data[0],
            target=target_data[0],
            input_GT=x_k_min/x_c,
            target_GT=k_k_min/k_c)
            )

    Parameters
    ----------
    input_dataset : Array
        List with one dimension of collocation points.
        Example: input_data[0]    
    target : Array
        Specific target from the target dataset, e.g., target_dataset[4]
    input_GT : List/Array
        Must be dimensionless!!
        Array that contains a list of values of the independent variable (e.g., x or t)
    target_GT : TYPE
        Must be dimensionless!!
        Array that contains a list of values of the dependent variable (e.g.. k)

    Returns
    -------
    input_CP : Array
    target : Array
    """
    def __init__(self,input_dataset,target,input_GT,target_GT,log=False,base=2):
        self.input_dataset = input_dataset
        self.target=target
        self.input_GT=input_GT
        self.target_GT = target_GT
        self.log = log
        self.base = base

    def replace_targets(self):
        """
        This method interpolates the target values of each collocation point between the points of the groundtruth

        Returns
        -------
        TYPE: Array
            Array of input values of the independent variable
        TYPE: Array
            Array of target values for each input value

        """
        input_CP = np.array([],dtype=np.int64)
        for i in range(len(self.target[0])):
            index = self.target[0][i]
            input_CP = np.append(input_CP,self.input_dataset[index])
        if self.log== True:
            input_CP_transformed = np.power(self.base,input_CP)-1
            self.target_CP = np.interp(input_CP_transformed,self.input_GT,self.target_GT)
            self.input_CP = input_CP_transformed
        else:
            self.target_CP = np.interp(input_CP,self.input_GT,self.target_GT)
            self.input_CP = input_CP
        return self.target[0], self.target_CP

    
    def replace_targets_with_threshold(self,dimless_threshold):
        """
        This method interpolates the target values of each collocation point between the points of the groundtruth

        Returns
        -------
        TYPE: Array
            Array of input values of the independent variable
        TYPE: Array
            Array of target values for each input value
        """
        input_CP = np.array([],dtype=np.int64)
        input_CP_tresh = np.array([],dtype=np.int64)
        input_nr_tresh = np.array([],dtype=np.int64)
        target_CP_tresh = np.array([],dtype=np.int64)

        for i in range(len(self.target[0])):
            index = self.target[0][i]
            input_CP = np.append(input_CP,self.input_dataset[index])
        self.target_CP = np.interp(input_CP,self.input_GT,self.target_GT)
        # Selecting targets based on the threshold:
        for i in range(len(self.target[0])):
            if abs(self.target_CP[i]) > dimless_threshold:
                input_CP_tresh = np.append(input_CP_tresh, input_CP[i])
                input_nr_tresh = np.append(input_nr_tresh, self.target[0][i])
                target_CP_tresh = np.append(target_CP_tresh,self.target_CP[i])
        self.input_nr_tresh = input_nr_tresh
        self.target_CP = target_CP_tresh
        self.input_CP = input_CP_tresh
        return self.input_nr_tresh, self.target_CP

    def get_inputs_and_targets(self):
        return self.input_CP, self.target_CP

    def get_inputs_and_targets_dim(self,factor_input,factor_target):
        return self.input_CP*factor_input,  self.target_CP*factor_target

    def get_Input_and_Target_for_GT_and_CP(self):
        return self.input_GT, self.target_GT, self.input_CP, self.target_CP

# test = np.array([np.zeros(50)+1,np.random.uniform(0, 1, 50)])

    def validation_data(self,CP_ratio=0.2,range_x=[0,1],range_t=[0,1]):
        # print("Length Validation Dataset: %i"%int(len(self.input_CP)*CP_ratio))
        if range_x[0]==range_x[1]:
            input_val = np.concatenate(([np.zeros(int(len(self.input_CP)*CP_ratio))+range_x[0]],
                                  [np.random.uniform(range_t[0], range_t[1], int(len(self.input_CP)*CP_ratio))]),axis=0)
            target_val =  np.interp(input_val[1],self.input_GT,self.target_GT)

        elif range_t[0]==range_t[1]:
            input_val = np.concatenate(([np.random.uniform(range_x[0], range_x[1], int(len(self.input_CP)*CP_ratio))],
                                        [np.zeros(int(len(self.input_CP)*CP_ratio))+range_t[0]]),axis=0)
            target_val =  np.interp(input_val[0],self.input_GT,self.target_GT)
        else:
            print("Error: This Function can only interpolate for 1D inputs.")

        return input_val.transpose(),target_val

    def plot_data(self,x_label,y_label,plt_title,semilogy=False,xlim=(0,1)):
        sns.set_style("darkgrid")
        sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
        plt.figure(figsize=(10,6))
        plt.plot(self.input_GT,self.target_GT,label="GT",marker=".",zorder=1)
        plt.scatter(self.input_CP, self.target_CP,label="CP Targets",s=5,color="red",zorder=2)
        if semilogy == True: plt.yscale("log")
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(xlim)
        plt.title("Dimensionless "+plt_title)


    def plot_hist(self,bins,x_label,
                  title="Distribution of Collocation Points over ",log=False):
      sns.set_style("darkgrid")
      sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
      plt.figure(figsize=(10,6))
      plt.hist(self.target_CP,bins=bins,log=log)
      plt.xlabel(x_label)
      plt.ylabel("Nr")
      plt.title(title+x_label)

    def plot_data_dim(self,x_label,y_label,plt_title,factor_input,factor_target,semilogy=False):
        """
        Plots Groundtruth and Collocation Points dimensional
        """
        sns.set_style("darkgrid")
        sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
        plt.figure(figsize=(10,6))
        plt.plot(self.input_GT*factor_input,self.target_GT*factor_target,label="GT",marker=".",zorder=1)
        plt.scatter(self.input_CP*factor_input, self.target_CP*factor_target,label="CP Targets",s=5,color="red",zorder=2)
        if semilogy == True: plt.yscale("log")
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Dimensional "+plt_title)