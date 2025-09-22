# Ipython startup command
# ipython --no-autoindent --matplotlib=agg --InteractiveShellApp.exec_lines="%matplotlib auto" --InteractiveShellApp.exec_lines="%load_ext autoreload" --InteractiveShellApp.exec_lines="%autoreload 2"

# %% Import
import numpy as np
import pickle
import pandas
import os
from datetime import datetime
import time
import ast # for converting string lists to real lists

import sciann as sn
from utils.sciann_datagenerator import *

import tensorflow as tf
# from tensorflow.python.ops import math_ops
# from tensorflow.python.keras import backend as K

# import tensorflow_probability as tfp
from scipy.interpolate import griddata

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm, Normalize
matplotlib.use('agg')
# matplotlib.use('tkagg')
plt.style.use("default")

# Own Files
from utils.SciANN_custom_targets import assign_targets_to_input_dim
# from utils.template_values import *
from utils.my_func import *
from utils.my_plots import *
matplotlib.rcParams.update(params)

#Set Random Seeds
set_seeds(123)

# %%% SI Units
SI_s, SI_minute, SI_hr, SI_day = 1., 60., 60.*60, 24*60.*60
SI_g, SI_kg = 1.e-3, 1.
SI_mm, SI_cm, SI_m, SI_km = 1e-3, 1e-2, 1.0, 1e3
SI_Pa, SI_kPa, SI_MPa, SI_GPa = 1.0, 1.e3, 1.e6, 1.e9
SI_micro, SI_milli, SI_centi, SI_kilo, SI_mega = 1e-6, 1e-3, 1e-2, 1e3, 1e6
poise = 0.1*SI_Pa*SI_s
Darcy = 9.869233e-13*SI_m**2

# %% Experiment Setup

# The `test_run` option allows to run the whole model with a small amount of epochs to test the workflow 
# test_run=True
test_run=False

experiment_FlowPINN = experiment_setup(delay=0.1)

experiment_param_GT_dataset = experiment_FlowPINN.add_param(["ogs_output_a_1e0","ogs_output_a_5e0"],
                # fixed_var=1,
                default_var=1,
                name="Name Ground Truth Dataset")

load_weights_kd = experiment_FlowPINN.add_param([False,True],
                fixed_var = 0,
                # fixed_var = 1,
                default_var=0,
                name="Load weights for kd?")

load_weights_pd_pinn = experiment_FlowPINN.add_param([False,True],
                fixed_var = 0,
                # fixed_var = 1,
                default_var=0,
                name="Load weights for pd_pinn?")

if test_run==True:
    experiment_param_epochs_kd = experiment_FlowPINN.add_param([10,50,100,200],
                                            fixed_var=0,
                                            default_var=0,
                                            name="Number Epochs Training kd")

    experiment_param_epochs_pd_pinn = experiment_FlowPINN.add_param([10,15,30,50,100],
                                            # fixed_var = 0,
                                            fixed_var = 3,
                                            # fixed_var = 3,
                                            name="Number Epochs pd(xd) for test run")

else:
    experiment_param_epochs_kd = experiment_FlowPINN.add_param([100,300,500],
                                            fixed_var=1,
                                            default_var=0,
                                            name="Number Epochs Training kd")

    experiment_param_epochs_pd_pinn = experiment_FlowPINN.add_param([100,500,1000,2000,3000,],
                                            # fixed_var = 0,
                                            default_var=0,
                                            name="Number Epochs pd")

    experiment_param_pd_pinn_retraining = experiment_FlowPINN.add_param([False,True],
                                            # fixed_var = 1,
                                            default_var=0,
                                            name="Perform retraining for too high loss values?")
    
    if experiment_param_pd_pinn_retraining==True:
        experiment_param_epochs_pd_pinn_retraining_treshold = experiment_FlowPINN.add_param([1e-3,8e-4,5e-4,1e-4],
                                                fixed_var = 3,
                                                default_var=3,
                                                doc="This threshold value decides if the PINN needs to be retrained with more epochs.",
                                                name="Error Treshold for Retraining PINN")
    else: experiment_param_epochs_pd_pinn_retraining_treshold=0.0001

experiment_param_pd_serial_iterx = experiment_FlowPINN.add_param([1,20],
                doc="Through how many retraining iterations should the PINN go?",
                default_var=0,
                name="Number of Iterations")

experiment_param_nr_observation_points = experiment_FlowPINN.add_param([25,10,4,0],
                default_var=0,
                name="Nr observation points")

experiment_param_sigma_noise = experiment_FlowPINN.add_param([0,0.01,0.02,0.03,0.04,0.1],
                # fixed_var = 0,
                default_var=0,
                name="Standard Deviation of the Gaussian Noise")

experiment_param_pinn_obs = experiment_FlowPINN.add_param([False,True],
                doc="This should be set 'True' for Experiment Nr1 and 'False' for Experiment Nr2",
                default_var=0,
                name="PINN with observation loss term?")

experiment_param_learning_rate_pd_VALUES = experiment_FlowPINN.add_param([
                                                    [1e-3, 1e-5],
                                                    [1e-4, 1e-6],
                                                    ],
                    fixed_var = 0,
                    default_var=0,
                    name="Learning Rate values")

experiment_param_learning_rate_pd_method = experiment_FlowPINN.add_param(["constant","linear","step","ExponentialDecay"],
                    fixed_var = 2,
                    default_var=3,
                    name="Learning Rate method")

experiment_param_adaptive_weights = experiment_FlowPINN.add_param([
                    None,
                    {'method': 'GN', 'freq': 20, 'use_score': True, 'alpha': 2.0},
                    {'method': 'NTK', 'freq': 20, 'use_score': True},
                    {'method': 'GP', 'freq': 20, 'use_score': True},
                    {'method': 'GP', 'freq': 300, 'use_score': True},
                    {'method': "inversedirischlet", 'freq': 500, 'use_score': True},
                    {'method': "self_adaptive_sample_weight", 'freq': 4, 'use_score': True},
                    ],
                fixed_var = 0,
                # fixed_var = 2,
                # fixed_var = 4,
                default_var=4,
                name="Adaptive Weighting")

experiment_param_safe_figs = experiment_FlowPINN.add_param([False,True],
                fixed_var=1,
                default_var=0,
                name="Safe Plots as PNG?")

experiment_param_create_gif = experiment_FlowPINN.add_param([False,True],
                # fixed_var=0,
                fixed_var=1,
                default_var=0,
                name="Create GIF of PNG?")


experiment_FlowPINN.print_summary()
experiment_FlowPINN.abort_if_x()

# %%% Paths to Save and Load Models
# We create for each model a path where it can be saved
save_model = save_models()
model_path_save = save_model.get_path()

experiment_FlowPINN.save_experiment_params(model_path_save)
experiment_FlowPINN.save_to_text(model_path_save)

# load_weights_pd_dat = True
load_weights_pd_dat = False
model_path_load_pd_dat ="saved_models/{}".format("")
model_path_load_kd ="saved_models/{}".format("")
model_path_load_pd_pinn ="saved_models/{}".format("")

# %% Classes and Functions

def equiv_frac_perm(b,a,km):
	"""
    Scalar formulation of the equivalent fracture permeability
	b ... Type: FLOAT
		Fracture width in [m]
	a ... Type: FLOAT
		Mean fracture distance in [m]. This should ne 
	km ... Type: Float
		Matrix permeability
 
	"""
	return km + b/a * (b**2/12 - km)

def Gaussian_permeability(k_max,k_min,b_1,x_k,mu,si):
    """
    Representation of an equivalent fracture via a Gaussian distribution
    
    b_1 ... width of the equivalent fracture
    si  ... standard deviation
    
    """
    m = (k_max-k_min)*b_1
    return m * 1/(si*np.sqrt(2*np.pi))*np.exp(-0.5*((x_k-mu)/si)**2)+k_min

class DataGenerator_p_xt:
    def __init__(self,
                 X=[0,1],
                 T=[0,1],
                 nr_obs_space=100,
                 nr_obs_time=100,
                 logT=False,
                 si_noise=0.1,
                 dataset_type=None,
                 load_weights_pd_dat=False
                 ):
        
        self.dataset_type = dataset_type
        self.nr_obs_time=nr_obs_time
        # =============================================================================
        # Observation points over space and time
        # =============================================================================
        
        np.random.seed(int(time.time()))
        # Random timesteps for each each well

        if nr_obs_space<5:self.x = np.concatenate((np.array([X[0]]),np.array([X[1]*0.57]),np.array([X[1]*0.86]),np.array([X[1]])))
        else: self.x = np.concatenate((np.array([X[0]]),np.array([X[1]*0.57]),np.array([X[1]*0.86]),np.array([X[1]]),np.random.uniform(xd_min,xd_max,nr_obs_space-3)))
        x_stacked = np.meshgrid(self.x,np.ones(nr_obs_time))[0].flatten()
        self.t_stacked = np_growing_steps(T[0],T[1],len(x_stacked),plot=False)

        # We create normally distributed noise with mu=0 and si=VARIABLE
        self.noise = np.random.normal(0,si_noise,len(self.x))
        # Set noise for injection point zero:
        self.noise[0] = 0

        self.noise_stacked = np.meshgrid(self.noise,np.ones(nr_obs_time))[0].flatten()
        self.x_stacked = np.meshgrid(self.x,np.ones(nr_obs_time))[0].flatten()

        self.inputs = [self.x_stacked.reshape(-1,1),self.t_stacked.reshape(-1,1)]
        # self.targets = [(np.arange(len(self.x_stacked)),np.zeros(len(self.x_stacked)))]
        self.targets = [(np.arange(len(self.x_stacked)),np.zeros(len(self.x_stacked)))]

    def get_data(self):
        return self.inputs, self.targets
    
    def get_obs_x(self):
        return self.x

    def plot_obs(self):
        plt.scatter(self.x_stacked,self.t_stacked,s=10)
        plt.xlabel(r"$\bar{x}$")
        plt.ylabel(r"$\bar{t}$")
        plt.xlim(np.min(self.x_stacked),np.max(self.x_stacked))
        plt.ylim(np.min(self.t_stacked),np.max(self.t_stacked))
        plt.title(r"Observation Points for $p(\bar{x},\bar{t})$")
        plt.grid()
        # plt.show()
        
class DataGenerator_DataDriven_kd:
    def __init__(self,
                 X=[0,1],
                 T=[0,1],
                 nr_CP=100,
                 logT=False,
                 focus_frac=False,
                 ):
        if focus_frac==True:
            x = np.clip(np.random.normal(0.5,0.2*dict_equiv_frac_params["a"],int(0.5*nr_CP)),X[0],X[1])
            self.x = np.concatenate((x,np.random.uniform(xd_min,xd_max,int(nr_CP-len(x)))))
        else: self.x =np.random.uniform(xd_min,xd_max,int(nr_CP))
        
        self.t = np.random.uniform(td_min,td_max,int(nr_CP))
        # x_stacked,t_stacked = np.meshgrid(self.x,self.t)
        # x_stacked,t_stacked = x_stacked.reshape(-1,1),t_stacked.reshape(-1,1)
        
        self.inputs = [self.x.reshape(-1,1),self.t.reshape(-1,1)]
        self.targets = [(np.arange(len(self.x)),np.zeros(len(self.x)))]
    
    def get_data(self,dims="xt"):
        if dims=="xt": return self.inputs, self.targets
        elif dims=="x": return [self.inputs[0]], self.targets
        else: raise ValueError("Please insert either 'xt' or 'x'")
    
    
    def get_obs_x(self):
        return self.x

class Data_Driven_ANN:
    def __init__(self,nr_obs_wells=100,si_noise=0):
        self.nr_obs_wells = nr_obs_wells
        
        DTYPE = 'float32'
        xd_dat = sn.Variable('xd', dtype=DTYPE)
        td_dat = sn.Variable('td', dtype=DTYPE)
        
        self.pd = sn.Functional('pd', [xd_dat,td_dat],4*[100], "tanh",output_activation="softplus")
        
        # The Data-Driven Model has just one loss term
        self.mod_pd_ANN = sn.SciModel([xd_dat,td_dat],[self.pd], optimizer="adam",loss_func="mse")
        
    def train(self,inputs,targets,val=None,
              test_run=False,epochs=200,path_load=None,path_save=None):
        
        self.inputs = inputs
        self.targets = targets
        batch_size =  int(len(self.inputs[0])*epochs/50e3)
        
        if test_run==True:
            epochs = 20
            batch_size =  64
        
        if isinstance(path_load,str):
            self.mod_pd_ANN.load_weights("{}/weights_pd_dat".format(path_load))
            self.history = pandas.read_csv("{}/loss_hist_pd_dat.csv".format(path_load,index_col=0))
        else:
            self.Hist_pd_dat = self.mod_pd_ANN.train(self.inputs, self.targets,
                                           epochs=epochs,
                                           learning_rate={"scheduler": "ExponentialDecay",
                                                               "initial_learning_rate": 1e-3,
                                                               "final_learning_rate": 1e-5,
                                                               "decay_epochs": epochs
                                                               },
                                           batch_size=batch_size,
                                           # verbose="True",
                                           save_weights={"path":"%s/checkpoints/"%path_save,
                                                         "freq":2},
                                           validation_data=val
                                           )
            self.history = self.Hist_pd_dat.history

        self.pd.set_trainable(False)
        self.mod_pd_ANN.compile()
        if isinstance(path_save,str):
            self.mod_pd_ANN.save_weights("%s/weights_pd_dat"%(path_save))
            pandas.DataFrame(data=self.history).to_csv("%s/loss_hist_pd_dat.csv"%path_save)

    def get_pd(self):
        return self.pd
    
    def get_SciMod(self):
        return self.mod_pd_ANN
    
    def get_hist(self):
        return self.Hist_pd_dat
    
    def plot_training(self,gif_duration=500):
        dir_list = os.listdir("%s/checkpoints/"%model_path_save)
        dir_list.sort()
        t_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for wei in range(len(dir_list)-2):
            print(dir_list[wei])
            self.mod_pd_ANN.load_weights("%s/checkpoints/%s"%(model_path_save,dir_list[wei]))
            self.mod_pd_ANN.compile()
            
            fig, ax1 = plt.subplots(1,1,
                                    # figsize=(10,7)
                                    )
            x_test = np.linspace(0,25)
            ax1.set_title("Epoch  = {}".format(dir_list[wei][1:6]))
            for t_test_i in range(len(t_test_p)):
                #  Prediction
                ax1.plot(x_test/x_c, self.pd.eval([x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),color=my_cmap["pred"],label="Prediction",
                         zorder=3,linestyle="dashed",linewidth=3)
                # GT
                dataset = GT_OGS[["x","p"]].loc[GT_OGS["t"]==t_test_p[t_test_i]].sort_values(by=["x"])
                ax1.plot(dataset["x"]/x_c,dataset["p"]/p_c,color=my_cmap["GT"],zorder=2,label="Ground Truth",linewidth=3)
                # Training Data Points
                ax1.scatter(self.dg_p_dat.get_obs_x(),
                          griddata(dataset["x"].values/x_c,dataset["p"].values/p_c,self.dg_p_dat.get_obs_x()).flatten(),
                          s=20,color="yellow",ec="black",alpha=1,zorder=5,label="Training Points")
                if t_test_i==0: ax1.legend()
                # =============================================================================
                # Annotate Timesteps
                # =============================================================================
                step_index = 50+int(t_test_i*len(dataset["x"])/(len(t_test_p)))
                # step_index = 1+int(t_test_i)*4
            
            ax1.grid()
            ax1.set_xlabel("xd")
            ax1.set_ylabel("pd")
            ax1.set_ylim(0,1)
            
            if os.path.exists(model_path_save+"/training_figs")==False:  os.mkdir(model_path_save+"/training_figs")
            if os.path.exists(model_path_save+"/training_figs/%s"%t_plot)==False:  os.mkdir(model_path_save+"/training_figs/%s"%t_plot)
            plt.savefig(model_path_save+"/training_figs/%s/Train_pd_dat_ep_%s.png"%(t_plot,dir_list[wei][1:6]))
            plt.close()
        
        from PIL import Image
        gif = []  
        image_path = model_path_save+"/training_figs/%s/"%t_plot
        if os.path.exists(image_path+"/gif")==False:  os.mkdir(image_path+"/gif")
        images = os.listdir(image_path)[2:]
        images.sort()
        for image in images[:len(images)-1]:
            gif.append(Image.open(image_path+image))
        
        gif[0].save('%s/gif/temp_result.gif'%image_path, save_all=True,
                    optimize=False, append_images=gif[1:],
                    loop=0, duration=gif_duration)

class log_scale_zero_to_one():
    """
    This class performs a log-scaling of a certain dataset.
    
    Parameters:
    -----------
    x_GT : Is the Ground Truth dataset of the parameter. It is used for calculating the x_min and x_max values.
    m    : Base for the logarithm
    n    : vertical shift
    """
    
    def __init__(self,func,x_GT,):
        self.x_GT = x_GT
        self.func = func
        self.x_GT_min = np.min(self.x_GT.flatten())
        self.x_GT_max = np.max(self.x_GT.flatten())
        self.m = self.x_GT_max/self.x_GT_min
        self.n = - np.log(self.x_GT_min)/np.log(self.m)
    
    def get_func_scaled(self):
        '''
        Returns scaled function
        -------
        TYPE
            DESCRIPTION.
        '''
        self.func_scaled = sn.log(self.func)/np.log(self.m)+self.n
        return self.func_scaled
    
    def scale_data(self,x,plot=False):
        self.fx = np.log(x)/np.log(self.m)+self.n
        if plot==True:
            plt.close()
            plt.figure(figsize=(10,7))
            plt.hist(x,log=True,bins=100,label="Data Distribution before Scaling",alpha=0.5)
            plt.hist(self.fx,log=True,bins=100,label="Data Distribution after Scaling",alpha=0.5)
            plt.legend(),plt.grid()
            # plt.show()
        return self.fx
    
    def backscale_data(self,fx):
        x = (self.x_GT_max/self.x_GT_min)**fx * self.x_GT_min
        return x
    
    def backscale_func(self):
        """
        Scales back a SciANN functional.
        """
        x = sn.pow(self.x_GT_max/self.x_GT_min,self.func) * self.x_GT_min
        return x
    
# %% Preprocessing


# %%% Load Ground Truth

GT_OGS = pandas.read_csv("01_data/{}.csv".format(experiment_param_GT_dataset),index_col=0)
t_start_filter = 1000 #s #If the first few seconds have a convergence performance, we filter these values
GT_OGS = GT_OGS.loc[GT_OGS["t"]>t_start_filter]

# =============================================================================
# Assign Material Properties of Factures to Dictionary
# =============================================================================
frac_matrix = np.array([
            # File name            a       start_si    final_si      timesteps
            ["ogs_output_a_1e0",   1e0 ,   0.8,        0.15,         "[10,18,25,27,30]"],
            ["ogs_output_a_5e0",   5e0 ,   0.2,        0.2,          "[2,9,15,28]"],
    ])

frac_idx = list(frac_matrix[:,0]).index(experiment_param_GT_dataset)

dict_equiv_frac_params = {"k_m":1e-18,"a":float(frac_matrix[frac_idx,1]) ,"b":10e-6,
                            "si_start":float(frac_matrix[frac_idx,2]),
                            "si_final":float(frac_matrix[frac_idx,3]),
                            "ts":list(ast.literal_eval(frac_matrix[frac_idx,4])),
                                }
if dict_equiv_frac_params["si_final"] == dict_equiv_frac_params["si_start"]:sigma_k_reduction=[dict_equiv_frac_params["si_final"]]
else:
    sigma_k_reduction = np_growing_steps(dict_equiv_frac_params["si_final"],dict_equiv_frac_params["si_start"],
                                        experiment_param_pd_serial_iterx,power=1.5,flip_order=True,plot=True)

GT_OGS_t = pandas.DataFrame(data={"t":np.sort(GT_OGS["t"].unique())})
t_test_p = GT_OGS_t.values.flatten()[dict_equiv_frac_params["ts"]]

GT_OGS_x = np.sort(GT_OGS["x"].unique())
GT_OGS_y = np.sort(GT_OGS["y"].unique())
GT_OGS_k = GT_OGS[["x","k"]].loc[(GT_OGS["t"]==GT_OGS_t.iloc[1].values[0])&(GT_OGS["y"]==GT_OGS_y[0])].sort_values(by=["x"])


# %%%% Plot Ground Truth
plt.close()
fig, ax1 = plt.subplots(1)
for i in range(len(GT_OGS_t)):
    timestep = list(GT_OGS_t.iloc[i].values)[0]
    dataset_GT_plot = GT_OGS[["x","p"]].loc[GT_OGS["t"]==timestep].sort_values(by=["x"])
    if timestep in list(t_test_p): 
        color="red"
    else: color=my_cmap["GT"]
    ax1.plot(dataset_GT_plot["x"],dataset_GT_plot["p"],color=color,label="Selected timesteps for Visualization" if timestep==t_test_p[0] else None
             )
    step_index = 1+int(i*len(dataset_GT_plot["x"])/(len(GT_OGS_t)-1)*0.7)
    ax1.annotate(r"$\bar{t}=%.2e\,$"%(GT_OGS_t.values[i]),
                    xy = (dataset_GT_plot["x"].iloc[step_index],dataset_GT_plot["p"].iloc[step_index]),
                   xytext = (0.5,0),
                  textcoords='offset points',
                  bbox={"boxstyle":"round", "fc":"w","ec":"teal"},
                  horizontalalignment='center',fontsize=10,zorder=6,
                  )
ax1.legend(loc="upper right")
ax2 = ax1.twinx()
ax2.plot(GT_OGS_k["x"],GT_OGS_k["k"]/max(GT_OGS_k["k"]),linewidth=3,label=r"$k(x)$",color="gray",linestyle="dashed")
ax2.legend(loc="upper center")
ax1.grid()
ax1.set_zorder(ax2.get_zorder() - 1)
ax1.set_ylabel("P /Pa")
ax1.set_xlabel("x /m")
plt.savefig("{}/Fig_01_Groundtruth_dataset.png".format(model_path_save),dpi=300)
# plt.show()

# %%%% Equivalent Gaussian Permeability Distribution

x = np.linspace(0,1,1000)
k_min = min(GT_OGS_k["k"].values)
# k_max = max(GT_OGS_k["k"].values)
k_max = equiv_frac_perm(dict_equiv_frac_params["b"],dict_equiv_frac_params["a"],dict_equiv_frac_params["k_m"],)
x_c = max(GT_OGS_x)

a = dict_equiv_frac_params["a"]/x_c
mu = 12.5/x_c
si = dict_equiv_frac_params["si_start"] * a

# =============================================================================
# Assign changed permeability
# =============================================================================

GT_OGS_k_Gauss = pandas.DataFrame(data={'x':GT_OGS_k['x'],
                                    'k': Gaussian_permeability(k_max,k_min,dict_equiv_frac_params["a"],GT_OGS_k['x'],12.5,
                                                                dict_equiv_frac_params["si_start"]*dict_equiv_frac_params["a"])
                                    })

plt.close()
plt.figure(figsize=(9,6),tight_layout=True)

plt.plot(GT_OGS_k_Gauss['x'],GT_OGS_k_Gauss['k'],linewidth=3,label=r"Ground Truth Plot",color="black",zorder=3)
plt.plot(x*x_c,Gaussian_permeability(k_max,k_min,a,x,mu,si),linewidth=3,label=r"$k_\mathregular{g}(x)$",color=my_cmap["v"],zorder=3,linestyle="dashed")
# plt.plot([min(x),mu-a/2,mu-a/2,mu+a/2,mu+a/2,max(x)],[k_min,k_min,k_max,k_max,k_min,k_min] ,linewidth=3,label=r"$k(x)$",color=my_cmap["GT"])
plt.plot(GT_OGS_k["x"],GT_OGS_k["k"],linewidth=3,label=r"$k(x)$",color=my_cmap["GT"])
plt.yscale("log")
plt.grid(True, which="both",color="lightgrey")
plt.legend()
plt.savefig("{}/Fig_02_GT_equivalent_permeability.png".format(model_path_save),dpi=300)
# plt.show()


# %%% Plot Full Permeability Distribution
# Here we print all disributions of k(x) for each iteration

fig, ax = plt.subplots(figsize=(11,6),tight_layout=False)
ax.plot(GT_OGS_k["x"],GT_OGS_k["k"],linewidth=3,label=r"$k(x)$",color=my_cmap["GT"])
fig.subplots_adjust(left=None, bottom=0, right=30, top=None, wspace=None, hspace=None)
cmap_kx = plt.get_cmap("flare",len(sigma_k_reduction))

for i_sigma_k_reduction in range(len(sigma_k_reduction)):
    GT_OGS_k_Gauss_step = pandas.DataFrame(data={'x':GT_OGS_k['x'],
                                        'k': Gaussian_permeability(k_max,k_min,dict_equiv_frac_params["a"],GT_OGS_k['x'],12.5,
                                                                    sigma_k_reduction[i_sigma_k_reduction]*dict_equiv_frac_params["a"])
                                        })

    ax.plot(GT_OGS_k_Gauss_step['x'],GT_OGS_k_Gauss_step['k'],linewidth=1,label=r"$k_\mathregular{g}(x)$",color=cmap_kx(i_sigma_k_reduction),zorder=3)
    # plt.plot(x*x_c,Gaussian_permeability(k_max,k_min,a,x,mu,si),linewidth=3,label=r"$k_\mathregular{g}(x)$",color=my_cmap["v"],zorder=3,linestyle="dashed")
    if i_sigma_k_reduction==0:plt.legend(fontsize=12)

plt.yscale("log")
plt.xlabel(r"$x$ /m")
plt.ylabel(r"$k(x)$ /m²")
plt.grid(True, which="both",color="lightgrey")

if len(sigma_k_reduction)>1:
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("flare_r",len(sigma_k_reduction)),norm=Normalize(vmin=min(sigma_k_reduction), vmax=max(sigma_k_reduction)))
    # cax = plt.axes([1, 0.13, 0.01, 0.8])

    cb = plt.colorbar(sm, 
                    #ticks=np.linspace(max(sigma_k_reduction),min(sigma_k_reduction), 9),
                label=r"Standard Deviation $\sigma \; /a$",
                ax=ax)
plt.savefig("{}/Fig_03_Stepwise Permeability.png".format(model_path_save),dpi=300)
# plt.show()
plt.close()
# %%% Physical Parameters
# Domain
Lx = 25 /SI_m
Ly = 1 /SI_m
p_ini= 0 * SI_MPa
p_max = 1*SI_MPa

k_max = np.round(max(GT_OGS_k['k']),18)
k_min = np.round(min(GT_OGS_k['k']),20)

#Material
rho_s = 2690
phi = 0.01

rho_fr_ref_cond = 1e5 #Pa
rho_fr_ref = 998.2  # for P=1e5 and T=20°C=193.15K
k_fr = (rho_fr_ref-1002.7)/((rho_fr_ref_cond)-(10.1e6)) / rho_fr_ref

mu = 1.006e-3 #dyn viscosity
g = 9.81

K_sr = 45e9
biot = 0.2 #determined by Braun (2007)
Ss = (k_fr * phi + (biot-phi) / K_sr)

# %%% Characteristic values

if load_weights_pd_pinn==True or load_weights_kd==True:
    experiment_param_dict = pickle.load(open("{}/experiment_param_dict.pkl".format(model_path_load_kd),"rb"))

    p_c = experiment_param_dict["p_c"]
    k_c = experiment_param_dict["k_c"]
    t_c = experiment_param_dict["t_c"]
    x_c = experiment_param_dict["x_c"]
    v_c = k_c * p_c / mu / x_c

    x_end = Lx
    x_scaler=x_end/x_c
    print("x_end/x_c=",x_scaler)

    t_end = Ss * mu / k_min * (Lx**2)
    t_scaler = t_end / t_c
    print("t_scaler=t_end/t_c=",t_scaler)
    experiment_param_dict_file = open("{}/experiment_param_dict.pkl".format(model_path_save),"wb")
else:
    experiment_param_dict = {}
    experiment_param_dict_file = open("{}/experiment_param_dict.pkl".format(model_path_save),"wb")
    p_c = p_max
    k_c = max(GT_OGS_k_Gauss["k"])
    x_end = Lx
    x_c= x_end
    x_scaler=x_end/x_c
    print("x_end/x_c=",x_scaler)

    t_max = max(GT_OGS['t'])
    t_c = Ss * mu / k_c * (x_c**2)
    t_end = Ss * mu / k_min * (Lx**2)
    t_scaler = t_end / t_c
    print("t_scaler=t_end/t_c=",t_scaler)

    v_c = k_c * p_c / mu / x_c

xd_min, xd_max = 0. , x_end/x_c
yd_min, yd_max = 0. , Ly/x_c
td_min, td_max = 0., t_end/t_c
pd_min,pd_max = 0, p_max / p_c

# Save model parameters
experiment_param_dict["p_c"]=p_c
experiment_param_dict["k_c"]=k_c
experiment_param_dict["t_c"]=t_c
experiment_param_dict["x_c"]=x_c
pickle.dump(experiment_param_dict,experiment_param_dict_file)
experiment_param_dict_file.close()

# %%% Reset SciANN
sn.reset_session()
sn.set_random_seed(124)

# %% Generate Training Data

nr_obs_wells = experiment_param_nr_observation_points

dg_p_dat = DataGenerator_p_xt(X=[xd_min,xd_max], T=[td_min,td_max],nr_obs_space = nr_obs_wells,nr_obs_time=100,logT=False,si_noise=experiment_param_sigma_noise,dataset_type="Training")
dg_input_p, dg_target_p = dg_p_dat.get_data()
dg_p_dat.plot_obs()

dg_p_dat_val = DataGenerator_p_xt(X=[xd_min,xd_max], T=[td_min,td_max],nr_obs_space = nr_obs_wells if nr_obs_wells<5 else int(nr_obs_wells*0.2),
                                          nr_obs_time=int(100*0.2),logT=False,si_noise=experiment_param_sigma_noise,dataset_type="Validation")
# dg_p_dat_val = DataGenerator_p_xt(X=[xd_min,xd_max], T=[td_min,td_max],nr_obs_space = nr_obs_wells,nr_obs_time=100,logT=False,si_noise=experiment_param_sigma_noise,dataset_type="Training")
dg_input_p_val, dg_target_p_val = dg_p_dat_val.get_data()

# Interpolate targets from Groundtruth dataset
for inp,tar,_class in zip([dg_input_p,dg_input_p_val],
                    [dg_target_p,dg_target_p_val],
                    [dg_p_dat,dg_p_dat_val]):
    tar[0] = (tar[0][0],griddata(GT_OGS[["x","t"]].values,GT_OGS["p"].values,
                                        np.concatenate((inp[0]*x_c,inp[1]*t_c),axis=1),
                                        method = "linear",fill_value=0.0).flatten()/p_c
                )
    # add noise
    if isinstance(experiment_param_sigma_noise,(float,int)): tar[0] = (tar[0][0],tar[0][1]+_class.noise_stacked)

plt.close()
plt.scatter(dg_input_p[0],dg_target_p[0][1],s=1), plt.xscale("linear"), plt.title("Subsampled target data for Data-Driven NN"), plt.grid()

# %% ANN

# %%% Train ANN

create_dir(model_path_save+"/checkpoints")
pd_ann = Data_Driven_ANN(experiment_param_nr_observation_points,si_noise=experiment_param_sigma_noise)
pd_ann.train(inputs=dg_input_p,targets=dg_target_p,
             val=(dg_input_p_val,dg_target_p_val),
             epochs=20 if test_run==True else 200,
            #  path_load=model_path_save,
             path_save=model_path_save,
             test_run=test_run
             )

# %%% Plot ANN Result

# history_in_plot=False
history_in_plot=True

if history_in_plot==True:
    fig, [ax1,ax2] = plt.subplots(2,1,height_ratios=[3,1],gridspec_kw={"hspace":0.3})

else: fig, ax1 = plt.subplots(1,1,figsize=(6,4),tight_layout=True)
x_test = np.linspace(0,25)

for t_test_i in range(len(t_test_p)):
    #  Prediction
    linewidth = 3
    ax1.plot(x_test/x_c, pd_ann.pd.eval([x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),color=my_cmap["pred"],label="Prediction",
                zorder=3,linestyle="dashed",linewidth=linewidth)
    # GT
    dataset = GT_OGS[["x","p"]].loc[GT_OGS["t"]==t_test_p[t_test_i]].sort_values(by=["x"])
    ax1.plot(dataset["x"]/x_c,dataset["p"]/p_c,color=my_cmap["GT"],zorder=2,label="Ground Truth",linewidth=linewidth)
    # Training Data Points
    ax1.scatter(dg_p_dat.get_obs_x(),
                griddata(dataset["x"].values/x_c,dataset["p"].values/p_c,dg_p_dat.get_obs_x()).flatten()+dg_p_dat.noise,
                s=20,color="yellow",ec="black",linewidth=0.7,alpha=1,zorder=5,label="Training Points")
    if t_test_i==0: ax1.legend()
    # =============================================================================
    # Annotate Timesteps
    # =============================================================================
    step_index = 40+int(t_test_i*len(dataset["x"])/(len(t_test_p)-1)*0.9)
    # step_index = 1+int(t_test_i)*4
    ax1.annotate(r"$\bar{t}=%.2f\,$"%(t_test_p[t_test_i]/t_c),
                    xy = (dataset["x"].iloc[step_index]/x_c,dataset["p"].iloc[step_index]/p_c),
                    # xytext = (0.5,0),
                    textcoords='offset points',bbox={"boxstyle":"round", "fc":"w","ec":"teal"},
                    horizontalalignment='center',fontsize=9,zorder=6
                    )
ax1.grid()
ax1.set_xlabel(r"$\bar{x}$")
ax1.set_ylabel(r"$\bar{p}(\bar{x},\bar{t})$")
ax1.set_ylim(-0.1,1.1)

if history_in_plot==True:
    ax2.plot(pd_ann.history["loss"]), ax2.grid(), ax2.set_xlabel("Epochs"), ax2.set_ylabel("Loss"), ax2.set_yscale("log")
    if experiment_param_safe_figs==True: plt.savefig("{}/Fig_04_ANN_Regression.png".format(model_path_save),dpi=300)
else:
    if experiment_param_safe_figs==True: plt.savefig("{}/Fig_04_ANN_Regression.png".format(model_path_save),dpi=300)

# %% PINN

# %%% NN Setup

#dimensionless variables
xd = sn.Variable('x')
yd = sn.Variable('y')
td = sn.Variable('t')

kd = sn.Functional('k', [xd/x_scaler,td/t_scaler], 4*[20], "tanh", output_activation="softplus", res_net=True)

kd_scaler = log_scale_zero_to_one(kd,GT_OGS_k["k"].values/k_c)
kd_pde = kd_scaler.backscale_func()

mod_kd = sn.SciModel([xd,td],[kd], optimizer="adam",loss_func="mse")

pd_pinn = sn.Functional('p', [xd/x_scaler,td/t_scaler], 4*[100], "tanh", output_activation="softplus", res_net=True)

# %% Iteration

for iteration_nr in range(experiment_param_pd_serial_iterx):
    print(iteration_nr)

    kd_GT_sigma = sigma_k_reduction[iteration_nr]

    if iteration_nr>0:
        # dict_equiv_frac_params["si_start"] = 0.6*dict_equiv_frac_params["si_start"]
        x = np.linspace(0,1,1000)
        
        k_min = min(GT_OGS_k["k"].values)
        # k_max = max(GT_OGS_k["k"].values)
        k_max = equiv_frac_perm(dict_equiv_frac_params["b"],dict_equiv_frac_params["a"],dict_equiv_frac_params["k_m"],)
        x_c = max(GT_OGS_x)
        
        a = dict_equiv_frac_params["a"]/x_c
        mu = 12.5/x_c
        si = kd_GT_sigma * a
        
        # =============================================================================
        # Assign changed permeability
        # =============================================================================
        GT_OGS_k_Gauss = pandas.DataFrame(data={'x':GT_OGS_k['x'],
                                            'k': Gaussian_permeability(k_max,k_min,dict_equiv_frac_params["a"],GT_OGS_k['x'],12.5,kd_GT_sigma*dict_equiv_frac_params["a"])
                                            })
        plt.figure(figsize=(9,6),tight_layout=True)
        plt.plot(GT_OGS_k_Gauss['x']/x_c,GT_OGS_k_Gauss['k'],linewidth=3,label=r"Ground Truth Plot",color="black",zorder=3)
        plt.plot(x,Gaussian_permeability(k_max,k_min,a,x,mu,si),linewidth=3,label=r"$k_\mathregular{g}(x)$",color=my_cmap["v"],zorder=3,linestyle="dashed")
        plt.plot(GT_OGS_k["x"]/x_c,GT_OGS_k["k"],linewidth=3,label=r"$k(x)$",color=my_cmap["GT"])
        plt.yscale("log")
        plt.grid(True, which="both",color="lightgrey")
        plt.legend()
        plt.savefig("{}/Fig_05_{:2d}_plot_GT_equivalent_permeability.png".format(model_path_save,iteration_nr),dpi=300)

    # %%% Train kd
    Nr_CP_kd = 6400
    data_k_x = DataGenerator_DataDriven_kd(X=[xd_min,xd_max],nr_CP = Nr_CP_kd,focus_frac=True)
    dg_input_data_k_x, dg_target_data_k_x = data_k_x.get_data(dims="xt")
    data_k_x_val = DataGenerator_DataDriven_kd(X=[xd_min,xd_max],nr_CP = Nr_CP_kd*0.2,focus_frac=True)
    dg_input_data_k_x_val, dg_target_data_k_x_val = data_k_x_val.get_data(dims="xt")
    # Interpolate data from groundtruth
    for inp,tar in zip([dg_input_data_k_x,dg_input_data_k_x_val],
                    [dg_target_data_k_x,dg_target_data_k_x_val]):
        p_interpolation =  kd_scaler.scale_data(griddata(GT_OGS_k_Gauss["x"].values,GT_OGS_k_Gauss["k"].values,inp[0]*x_c,
                                                        method = "linear",fill_value=0.0).flatten()/k_c,plot=True)
        tar[0] = (tar[0][0],p_interpolation)
    
    plt.scatter(GT_OGS_k_Gauss["x"].values/x_c,GT_OGS_k_Gauss["k"].values/(k_c),s=1)
    plt.scatter(dg_input_data_k_x[0],dg_target_data_k_x[0][1],s=1,alpha=0.5)
    plt.close()
    
    plt.hist(dg_input_data_k_x[0],bins=50), plt.close()
    plt.hist(dg_target_data_k_x[0][1],bins=50), plt.close()
    
    epochs_kd = experiment_param_epochs_kd
    batch_size_kd =  int(len(dg_input_data_k_x[0])*epochs_kd/50e3)

    if iteration_nr>0: epochs_kd=int(epochs_kd/5)

    if test_run ==True: batch_size_kd= 640
    
    # =============================================================================
    # Train
    # =============================================================================
    
    learning_rate_kd = stepwise_learning_rate(1e-3,1e-5,epochs_kd/int(10),epochs_kd)
    # learning_rate_kd = {"scheduler": "ExponentialDecay","initial_learning_rate": 1e-3,"final_learning_rate": 1e-5,"decay_epochs": epochs_kd}
    
    if load_weights_kd==True and iteration_nr==0:
        mod_kd.load_weights("{}/weights_kd".format(model_path_load_kd))
        loss_kd=pandas.read_csv("%s/loss_hist_kd.csv"%model_path_load_kd,index_col=0)
    else:
        kd.set_trainable(True)
        mod_kd.compile()

        Hist_kd = mod_kd.train(dg_input_data_k_x, dg_target_data_k_x,
                                       epochs=epochs_kd,
                                       learning_rate=learning_rate_kd,
                                       batch_size=batch_size_kd,
                                       # verbose="True",
                                       validation_data=(dg_input_data_k_x_val, dg_target_data_k_x_val)
                                       )
        mod_kd.save_weights("%s/weights_kd"%(model_path_save))
        pandas.DataFrame(data=Hist_kd.history).to_csv("%s/loss_hist_kd.csv"%model_path_save)
    kd.set_trainable(False)
    mod_kd.compile()

    # %%%% Eval kd
    # =============================================================================
    # Plots
    # =============================================================================
    switch_plot_history=True
    
    if switch_plot_history==True:
        fig, [ax1,ax2] = plt.subplots(2,1,height_ratios=[3,1],gridspec_kw={"hspace":0.3})
    
    else: fig, ax1 = plt.subplots(figsize=(12,6),tight_layout=True)
    
    x_test = np.linspace(0,25,2000)
    t_test = np.full(x_test.shape,0.5)
    
    #Original
    mu_frac = 12.5/x_c
    a_frac = dict_equiv_frac_params["a"]/x_c
    k_max_frac = equiv_frac_perm(dict_equiv_frac_params["b"],dict_equiv_frac_params["a"],dict_equiv_frac_params["k_m"])
    k_min_frac =dict_equiv_frac_params["k_m"]
    ax1.plot(GT_OGS_k["x"]/x_c,
             GT_OGS_k["k"]/k_c ,
             linewidth=3,label=r"$k(x)$",color=my_cmap["CP"],zorder=2)
    # GT
    ax1.plot(GT_OGS_k_Gauss["x"]/x_c,GT_OGS_k_Gauss["k"]/k_c,color=my_cmap["GT"],zorder=3,label=r"$k_\mathregular{g}(x)$",
             linewidth=3)
    #  Prediction
    ax1.plot(x_test/x_c, kd_pde.eval(mod_kd,[x_test/x_c,t_test]),color=my_cmap["pred"],label=r"$\hat{k}_\mathregular{g}(x)$ ANN",
            zorder=3,linestyle="dashed",linewidth=4)
    
    ax1.legend(fontsize="large")
    ax1.grid()
    ax1.set_yscale("log")
    
    if switch_plot_history==True and load_weights_kd==False:
        ax2.plot(Hist_kd.history["loss"])
        # ax2.grid()
        ax2.legend()
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_yscale("log")

    plt.grid(True, which="both",color="lightgrey",zorder=-1)
    plt.savefig("{}/Fig_05_kd_trained.png".format(model_path_save),dpi=300)

    # %%% Train pd
    pd_xd, pd_xdxd =sn.diff(pd_pinn,xd),  sn.diff(pd_pinn,xd,order=2)
    
    L1 = sn.rename(( sn.diff(pd_pinn,td) - (sn.diff(kd_pde,xd) * pd_xd + kd_pde* pd_xdxd)),"PDE")

    targets_p_xt_table = np.array([
        # 1. Loss Term | 2. Weight | 3. Domain Section | 4. Loss Metric | 
        # =============================================================================
        # Domain Constraints
        # =============================================================================
        [L1,                                        1,   "domain",       "mse", "PDE"        , "pde"       ],
        [sn.rename(1*(pd_pinn),"BC_p_xmax")   ,     1,   "bc-left",      "mse", "BC_p_xmax"  , "bco"       ],
        [sn.rename(1*(pd_pinn),"BC_p_xmin")   ,     1,   'bc-right',     "mse", "BC_p_xmin"  , "bci"       ],
        [sn.rename(1*(pd_pinn),"IC_p_tmin")   ,     1,   'ic',           "mse", "IC_p_tmin"  , "ic"        ],
        # =============================================================================
        ])
    
    if experiment_param_pinn_obs==True:
        targets_p_xt_table = np.concatenate((targets_p_xt_table,
                    np.array([[sn.rename(1*(pd_pinn),"data_p"),           1,   'domain',       "mse", "data_p"  , "o"]])
                    ))
    
    # %%%% Generate Data
    # Nr_CP = int(6400*3)
    Nr_CP = 6400*2

    if experiment_param_pinn_obs==True:
        Nr_CP = int(6400*3)
    
    data_p_xt = DataGeneratorXT(X=[xd_min,xd_max], T=[td_min,td_max],num_sample = Nr_CP,targets=targets_p_xt_table[:,2],logT=False)
    data_p_xt.plot_data(name="Training_Data")
    dg_input_data_p_xt, dg_target_data_p_xt = data_p_xt.get_data()
    
    data_p_xt_val = DataGeneratorXT(X=[xd_min,xd_max], T=[td_min,td_max],num_sample = Nr_CP*0.2,targets=targets_p_xt_table[:,2],logT=False)
    data_p_xt_val.plot_data(name="Validation_Data")
    dg_input_data_p_xt_val, dg_target_data_p_xt_val = data_p_xt_val.get_data()
    
    # Assign PDE targets
    dg_target_data_p_xt[0] = (dg_target_data_p_xt[0][0],np.zeros(len(dg_target_data_p_xt[0][0])))
    dg_target_data_p_xt_val[0] = (dg_target_data_p_xt_val[0][0],np.zeros(len(dg_target_data_p_xt_val[0][0]),dtype=np.float64))
    
    # Assign Pressure BC and IC
    target_nr = 1
    target_BC_p_x_min = assign_targets_to_input_dim(dg_input_data_p_xt[1],target=dg_target_data_p_xt[target_nr],
                                    input_GT=GT_OGS["t"].loc[(GT_OGS['x']==0.0)].values/t_c,
                                    target_GT=GT_OGS["p"].loc[(GT_OGS['x']==0.0)].values/p_c,
                                    # log=True,base=td_max+1,
                                    )
    dg_target_data_p_xt[target_nr] = (target_BC_p_x_min.replace_targets())
    target_BC_p_x_min_val = assign_targets_to_input_dim(dg_input_data_p_xt_val[1],target=dg_target_data_p_xt_val[target_nr],
                                    input_GT=GT_OGS["t"].loc[(GT_OGS['x']==0.0)].values/t_c,
                                    target_GT=GT_OGS["p"].loc[(GT_OGS['x']==0.0)].values/p_c,
                                    # log=True,base=td_max+1
                                    )
    dg_target_data_p_xt_val[target_nr] = (target_BC_p_x_min_val.replace_targets())
    
    # =============================================================================
    # Assign Observation points
    # We use the same training data-points that we used for the Data-driven Neural Network
    # =============================================================================
    if "data_p" in list(targets_p_xt_table[:,4]):
        target_nr = list(targets_p_xt_table[:,4]).index("data_p")
        dg_input_data_p_xt = [
            np.concatenate((dg_input_data_p_xt[0],dg_input_p[0])),
            np.concatenate((dg_input_data_p_xt[1],dg_input_p[1])),
                                ]
        dg_target_data_p_xt[target_nr]=(dg_target_p[0][0]+Nr_CP,dg_target_p[0][1])
        
        dg_input_data_p_xt_val = [
            np.concatenate((dg_input_data_p_xt_val[0],dg_input_p_val[0])),
            np.concatenate((dg_input_data_p_xt_val[1],dg_input_p_val[1])),
                                ]
        dg_target_data_p_xt_val[target_nr] = (dg_target_p_val[0][0]+int(Nr_CP*0.2),dg_target_p_val[0][1])

    # Sample Weights
    sample_weights_pd_pinn = np.full(dg_input_data_p_xt[0].shape,1.)
    
    # %%%% Train pd pinn

    hist_dict_pd_pinn = {"loss":np.array([1])}
    repetion_pd_pinn = 0
    # we repeat the training if the loss value did not reach below a certain value
    while hist_dict_pd_pinn["loss"][len(hist_dict_pd_pinn["loss"])-1] > experiment_param_epochs_pd_pinn_retraining_treshold:
        print("check")
        mod_pd_pinn = sn.SciModel([xd,td],list(targets_p_xt_table[:,0]), optimizer="adam",loss_func=list(targets_p_xt_table[:,3]))
        pd_pinn.set_trainable(True)
        epochs_pd_pinn = experiment_param_epochs_pd_pinn
        batch_size_pd_pinn= int(len(dg_input_data_p_xt[0])*epochs_pd_pinn/50e3)

        # =============================================================================
        # Assign Epochs
        # =============================================================================

        # Iteration Epochs
        if iteration_nr>0 and test_run==False: epochs_pd_pinn=500
        # if the training is repeated because of a too high loss, then the 
        if repetion_pd_pinn > 0: epochs_pd_pinn = int(epochs_pd_pinn * 2 * repetion_pd_pinn ** 2)

        save_weights_freq = int(epochs_pd_pinn/10)
        if test_run ==True:
            epochs_pd_pinn, batch_size_pd_pinn= experiment_param_epochs_pd_pinn, 640
            save_weights_freq = 2

        #Assign Learning Rate
        initial_lr = experiment_param_learning_rate_pd_VALUES[0]
        final_lr = experiment_param_learning_rate_pd_VALUES[1]

        if experiment_param_learning_rate_pd_method =="constant" : learning_rate_pd_pinn=initial_lr
        elif experiment_param_learning_rate_pd_method =="linear" : [[0,epochs_pd_pinn],[1e-3,1e-5]]
        elif experiment_param_learning_rate_pd_method =="step" :
            learning_rate_pd_pinn = stepwise_learning_rate(initial_lr,final_lr,epochs_pd_pinn/int(5),epochs_pd_pinn,plot=True)
        elif experiment_param_learning_rate_pd_method =="ExponentialDecay" :
            learning_rate_pd_pinn = {"scheduler": "ExponentialDecay",
                                "initial_learning_rate": initial_lr,
                                "final_learning_rate": final_lr,
                                "decay_epochs": epochs_pd_pinn
                                }
        else: raise TypeError("Plesase enter learning rate in an appropriate format")

        start_time = time.time()
        
        if load_weights_pd_pinn==True and iteration_nr==0 and repetion_pd_pinn==0:
            hist_dict_pd_pinn=pandas.read_csv("%s/loss_hist_pd_pinn.csv"%model_path_load_pd_pinn,index_col=0)
            mod_pd_pinn.load_weights("{}/weights_pd_pinn".format(model_path_load_pd_pinn))
            pd_pinn.set_trainable(False)
            mod_pd_pinn.compile()
            H_pd_pinn = mod_pd_pinn.train(dg_input_data_p_xt,dg_target_data_p_xt,epochs=1)
        else:
            kd.set_trainable(False)
            pd_pinn.set_trainable(True)
            mod_pd_pinn.compile()

            # =============================================================================
            # Safe weights
            # =============================================================================
            create_dir("%s/checkpoints/"%model_path_save)
            
            H_pd_pinn = mod_pd_pinn.train(dg_input_data_p_xt,dg_target_data_p_xt,
                                            epochs=epochs_pd_pinn,
                                            learning_rate=learning_rate_pd_pinn,
                                            batch_size=batch_size_pd_pinn,
                                            weights=sample_weights_pd_pinn,
                                            verbose = 1,
                                            save_weights={"path":"%s/checkpoints/"%model_path_save,
                                                            "freq":save_weights_freq},
                                            validation_data=(dg_input_data_p_xt_val,dg_target_data_p_xt_val),
                                            validation_freq=1,
                                            adaptive_weights=experiment_param_adaptive_weights
                                            )
            pd_pinn.set_trainable(False)
            mod_kd.compile(),mod_pd_pinn.compile()
    
            hist_dict_pd_pinn=H_pd_pinn.history
    
            mod_pd_pinn.save_weights("%s/weights_pd_pinn"%(model_path_save))
            pandas.DataFrame(data=hist_dict_pd_pinn).to_csv("%s/loss_hist_pd_pinn.csv"%model_path_save)

        print('\n Model completed in {:0.2f}s'.format(time.time() - start_time))
        # %%%% Eval pd
        
        # switch_plot_history=False
        switch_plot_history=True
        if switch_plot_history==True:
            fig, [ax1,ax3] = plt.subplots(1,2,width_ratios=[1,1],gridspec_kw={"hspace":0.4},figsize=(18,6))
        
        else: fig, ax1 = plt.subplots(1,1,figsize=(10,6),tight_layout=True)

        x_test = np.linspace(0,25,1000)
        for t_test_i in range(len(t_test_p)):
            #  Prediction
            ax_pred = ax1.plot(x_test/x_c, pd_pinn.eval(mod_pd_pinn,[x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),
            # ax1.plot(x_test/x_c, pd_pinn.eval(mod_pd_pinn,[x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),
                        # color=my_cmap["pred"],label="Prediction",
                        zorder=3,**kwargs_plot["pred"])
            # GT
            dataset = GT_OGS[["x","p"]].loc[GT_OGS["t"]==t_test_p[t_test_i]].sort_values(by=["x"])
            ax_gt = ax1.plot(dataset["x"]/x_c,dataset["p"]/p_c,
                        # color=my_cmap["GT"],label="Ground Truth",
                        **kwargs_plot["GT"],
                        zorder=2)
            
            if experiment_param_pinn_obs==True:
                # Training Data Points
                ax1.scatter(dg_input_p[0],
                            griddata(dataset["x"].values/x_c,dataset["p"].values/p_c,dg_input_p[0]).flatten()+dg_p_dat.noise_stacked,
                            s=20,color="yellow",ec="black",linewidth=0.7,alpha=1,zorder=5,label="Training Points")
            else:
                ax1.scatter([xd_min,xd_max],[1,0],
                            s=60,color="yellow",ec="black",linewidth=0.7,alpha=1,zorder=5,label="Training Points")

            if t_test_i==0: ax1.legend(loc='upper left', bbox_to_anchor=(0.65, 0.99 ),facecolor="white")

            # =============================================================================
            # Annotate Timesteps
            # =============================================================================
            step_index = 40+int(t_test_i*len(dataset["x"])/(len(t_test_p)-1)*0.9)
            # step_index = 1+int(t_test_i)*4
            ax1.annotate(r"$\bar{t}=%.2f\,$"%(t_test_p[t_test_i]/t_c),
                            xy = (dataset["x"].iloc[step_index]/x_c,dataset["p"].iloc[step_index]/p_c),
                            # xytext = (0.5,0),
                            textcoords='offset points',bbox={"boxstyle":"round", "fc":"w","ec":"teal","alpha":0.7},
                            horizontalalignment='center',fontsize=9,zorder=6,
                            )
            # =============================================================================
            # Plot Permeability Field
            # =============================================================================
            if t_test_i==1:
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                color = "darkslategrey"
                #  Prediction
                ax2.plot(x_test/x_c, kd_pde.eval(mod_kd,[x_test/x_c,np.full(x_test.shape,0.5)]),color=color,
                            label=r"$\hat{k}_\mathregular{g}(x)$ ANN",zorder=0,linestyle="dashed",
                            linewidth=2,alpha=0.45)
                ax2.tick_params(axis='y', labelcolor=color)
        
                ax2.plot(GT_OGS_k["x"]/x_c,
                            GT_OGS_k["k"]/k_c ,
                            linewidth=2,label=r"$k(x)$ Ground Truth",color=color,zorder=1,alpha=0.5)
                ax2.legend(loc='upper left', bbox_to_anchor=(0.65, 0.75 ),facecolor="white",framealpha=1.)

        ax1.grid(zorder=0)
        ax1.set_title(r"$\sigma = {:.2f}$".format(kd_GT_sigma))
        ax1.set_xlabel(r"$\bar{x}$")
        ax1.set_ylabel(r"$\bar{p}(\bar{x},\bar{t})$")
        ax1.set_ylim(-0.1,1.1)
        ax1.patch.set_visible(False)
        ax1.set_zorder(ax2.get_zorder() + 1)
        
        ax2.set_ylabel(r'$\bar{k}(\bar{x})$', color=color,labelpad=-15)  # we already handled the x-label with ax1
        ax2.patch.set_facecolor('white')
        ax2.patch.set_visible(True)
        ax2.set_ylim(0.64*k_min_frac/k_c,np.round(max(GT_OGS_k["k"])/k_c*2,0))
        ax2.set_yscale("log")
        
        # =============================================================================
        # Add Histogram
        # =============================================================================
        # ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("axes", 1.15))
        # color_hist = "brown"
        # ax3.hist(dg_input_data_p_xt[0],bins=50,color=color_hist,alpha=0.3)
        # # ax3.yaxis.label.set_color(color_hist)
        # ax3.set_ylabel("Nr CP", color=color_hist)
        # ax3.tick_params(axis='y', labelcolor=color_hist)
    
        if switch_plot_history==True:
            ax3.plot(hist_dict_pd_pinn["loss"],label=r"$\mathcal{L}$")
            for i in range(len(targets_p_xt_table[:,4])):
                if targets_p_xt_table[:,5][i]=="pde": alpha_loss = 1.
                else: alpha_loss = 0.3
                ax3.plot(list(hist_dict_pd_pinn["{}_loss".format(targets_p_xt_table[:,4][i])]),label=r"$\mathcal{L}_{\mathregular{%s}}$"%(targets_p_xt_table[:,5][i]),
                            alpha=alpha_loss)
            ax3.grid()
            ax3.legend(fontsize=15)
            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Loss (MSE)",labelpad=0)
            ax3.set_yscale("log")
            ax3.set_ylim(1e-6,1e0)

            ax_lr = ax3.twinx()
            ax_lr.plot(hist_dict_pd_pinn["lr"],color="black",linestyle="dashed",label="Learning Rate")
            ax_lr.set_ylabel("Learning Rate",color="black")
            ax_lr.tick_params(axis='y', labelcolor="black")
            ax_lr.set_yscale("log")
            ax_lr.legend(bbox_to_anchor=(0.8,0.999))
            if experiment_param_safe_figs==True: plt.savefig("{}/Fig_06_PINN_training_iteration_{}.png".format(model_path_save,iteration_nr),dpi=300,bbox_inches='tight')
        else:
            if experiment_param_safe_figs==True: plt.savefig("{}/Fig_06_PINN_training_iteration_{}.png".format(model_path_save,iteration_nr),dpi=300,bbox_inches='tight')
            plt.close()
            fig,ax = plt.subplots(figsize=(10,7))
            ax.plot(hist_dict_pd_pinn["loss"],label=r"$\mathcal{L}$")
            for i in range(len(targets_p_xt_table[:,4])):
                ax.plot(list(hist_dict_pd_pinn["{}_loss".format(targets_p_xt_table[:,4][i])]),label=r"$\mathcal{L}_{\mathregular{%s}}$"%(targets_p_xt_table[:,5][i]))
            ax.grid()
            ax.legend(fontsize=15)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.set_ylim(1e-6,1e0)

            ax_lr = ax.twinx()
            ax_lr.plot(hist_dict_pd_pinn["lr"],color="black",linestyle="dashed",label="Learning Rate")
            ax_lr.set_ylabel("Learning Rate",color="black")
            ax_lr.tick_params(axis='y', labelcolor="black")
            ax_lr.legend(bbox_to_anchor=(1,0.65))

            plt.savefig("{}/Fig_07_PINN_training_history_{}.png".format(model_path_save,iteration_nr),dpi=300,bbox_inches='tight')
        
        # %%%% Print Error Map

        xd_error = np.meshgrid(np.linspace(xd_min,xd_max,200),np.linspace(td_min,td_max,200))[0]
        td_error = np.meshgrid(np.linspace(xd_min,xd_max,200),np.linspace(td_min,td_max,200))[1]

        pd_gt_map = griddata(GT_OGS[["x","t"]],GT_OGS["p"]/p_c,
                                    np.concatenate((xd_error.reshape(-1,1)*x_c,td_error.reshape(-1,1)*t_c),axis=1),
                                    method="linear",fill_value=0
                                    ).reshape(xd_error.shape)

        pd_pinn_map = pd_pinn.eval(mod_pd_pinn,[xd_error,td_error])
        pd_error_map = abs(pd_gt_map-pd_pinn_map)
        pd_pinn_residual = mod_pd_pinn.losses_cp[0][dg_target_data_p_xt[0][0]]

        # =============================================================================
        # Plotting
        # =============================================================================

        fig, [ax1,ax3,ax4,] = plt.subplots(1,3,figsize=(18,6),
                        #tight_layout=True,
                        # subplot_kw={"projection": "3d"},
                        #height_ratios=[1,2],
                        width_ratios=[3,1,1],
                        tight_layout=True,
                        gridspec_kw={"hspace":0,"wspace":0.5},
                        )

        x_test = np.linspace(0,25,1000)
        for t_test_i in range(len(t_test_p)):
            #  Prediction
            ax_pred = ax1.plot(x_test/x_c, pd_pinn.eval(mod_pd_pinn,[x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),
            # ax1.plot(x_test/x_c, pd_pinn.eval(mod_pd_pinn,[x_test/x_c,np.full(x_test.shape,t_test_p[t_test_i]/t_c)]),
                        # color=my_cmap["pred"],label="Prediction",
                        zorder=3,**kwargs_plot["pred"])
            # GT
            dataset = GT_OGS[["x","p"]].loc[GT_OGS["t"]==t_test_p[t_test_i]].sort_values(by=["x"])
            ax_gt = ax1.plot(dataset["x"]/x_c,dataset["p"]/p_c,
                        # color=my_cmap["GT"],label="Ground Truth",
                        **kwargs_plot["GT"],
                        zorder=2)
            
            if experiment_param_pinn_obs==True:
                # Training Data Points
                ax1.scatter(dg_input_p[0],
                            griddata(dataset["x"].values/x_c,dataset["p"].values/p_c,dg_input_p[0]).flatten()+dg_p_dat.noise_stacked,
                            s=20,color="yellow",ec="black",linewidth=0.7,alpha=1,zorder=5,label="Training Points")
            else:
                ax1.scatter([xd_min,xd_max],[1,0],
                            s=60,color="yellow",ec="black",linewidth=0.7,alpha=1,zorder=5,label="Training Points")

            if t_test_i==0: ax1.legend(loc='upper left', bbox_to_anchor=(0.71, 0.99 ),facecolor="white")

            # =============================================================================
            # Annotate Timesteps
            # =============================================================================
            step_index = 20+int(t_test_i*len(dataset["x"])/(len(t_test_p)-1)*0.9)
            # step_index = 1+int(t_test_i)*4
            ax1.annotate(r"$\bar{t}=%.2f\,$"%(t_test_p[t_test_i]/t_c),
                            xy = (dataset["x"].iloc[step_index]/x_c,dataset["p"].iloc[step_index]/p_c),
                            # xytext = (0.5,0),
                            textcoords='offset points',bbox={"boxstyle":"round", "fc":"w","ec":"teal","alpha":0.8},
                            horizontalalignment='center',fontsize=9,zorder=6,
                            )

            # =============================================================================
            # Plot Permeability Field
            # =============================================================================
            if t_test_i==1:
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                color = "darkslategrey"
                #  Prediction
                ax2.plot(x_test/x_c, kd_pde.eval(mod_kd,[x_test/x_c,np.full(x_test.shape,0.5)]),color=color,
                            label=r"$\hat{k}_\mathregular{g}(x)$ ANN",zorder=0,linestyle="dashed",
                            linewidth=2,alpha=0.45)
                ax2.tick_params(axis='y', labelcolor=color)
        
                ax2.plot(GT_OGS_k["x"]/x_c,
                            GT_OGS_k["k"]/k_c ,
                            linewidth=2,label=r"$k(x)$ Ground Truth",color=color,zorder=1,alpha=0.5)
                ax2.legend(loc='upper left', bbox_to_anchor=(0.71, 0.83 ),facecolor="white")

        ax1.grid(zorder=0)
        ax1.set_xlabel(r"$\bar{x}$")
        ax1.set_ylabel(r"$\bar{p}(\bar{x},\bar{t})$")
        ax1.set_ylim(-0.1,1.1)
        ax1.patch.set_visible(False)
        ax1.set_zorder(ax2.get_zorder() + 1)
        
        ax2.set_ylabel(r'$\bar{k}(\bar{x})$', color=color,labelpad=-10)  # we already handled the x-label with ax1
        ax2.patch.set_facecolor('white')
        ax2.patch.set_visible(True)
        ax2.set_ylim(0.64*k_min_frac/k_c,np.round(max(GT_OGS_k["k"])/k_c*2,0))
        ax2.set_yscale("log")

        # =============================================================================
        # Absolute Error
        # =============================================================================

        data = ax3.pcolor(xd_error,td_error,abs(pd_gt_map-pd_pinn_map),cmap="jet",norm=LogNorm(vmin=1e-5, vmax=1))
        plt.colorbar(data,ax=ax3,label="Absolute Error",pad=0.06)
        ax3.plot([xd_max/2-dict_equiv_frac_params["a"]/x_c,xd_max/2-dict_equiv_frac_params["a"]/x_c],[td_min,td_max],linestyle="dashed",color="black" )
        ax3.plot([xd_max/2+dict_equiv_frac_params["a"]/x_c,xd_max/2+dict_equiv_frac_params["a"]/x_c],[td_min,td_max],linestyle="dashed",color="black" )
        ax3.annotate("Equivalent Fracture",xy=((xd_max/2)-0.05-dict_equiv_frac_params["a"]/x_c,td_max*0.98),ha="right",va="top",
                        rotation=90,bbox=dict(boxstyle="round", fc="w",ec="black",alpha=0.7))
        ax3.grid()
        ax3.set(xlabel=r"$\bar{x}$")
        ax3.set_ylabel(r"$\bar{t}$",labelpad=-5)

        # ============================================================================
        # Abs Residual
        # =============================================================================

        data = ax4.tricontourf(
            dg_input_data_p_xt[0][dg_target_data_p_xt[0][0]].flatten(),
            dg_input_data_p_xt[1][dg_target_data_p_xt[0][0]].flatten(),
            pd_pinn_residual,
            np.logspace(np.log10(1e-5),np.log10(1), 100),
            locator=mpl.ticker.LogLocator(),
            # cmap="gist_ncar"
            cmap="jet"
            )
        ax4.plot([xd_max/2-dict_equiv_frac_params["a"]/x_c,xd_max/2-dict_equiv_frac_params["a"]/x_c],[td_min,td_max],linestyle="dashed",color="black" )
        ax4.plot([xd_max/2+dict_equiv_frac_params["a"]/x_c,xd_max/2+dict_equiv_frac_params["a"]/x_c],[td_min,td_max],linestyle="dashed",color="black" )
        ax4.annotate("Equivalent Fracture",xy=((xd_max/2)-0.05-dict_equiv_frac_params["a"]/x_c,td_max*0.98),ha="right",va="top",
                        rotation=90,bbox=dict(boxstyle="round", fc="w",ec="black",alpha=0.7))
        cbar = plt.colorbar(data,ax=ax4,label="Absolute Residual",pad=0.06)
        cbar.locator = mpl.ticker.LogLocator(10)
        cbar.minorticks_on()
        ax4.grid()
        ax4.set(xlabel=r"$\bar{x}$",
                ylim=(td_min,td_max),xlim=(0,1)
                )
        ax4.set_ylabel(r"$\bar{t}$",labelpad=-5)
        
        plt.savefig("{}/Fig_08_PINN_plot_Error_Map_Iteration {}.png".format(model_path_save,iteration_nr ),dpi=300,bbox_inches="tight")
        ax1.set_title(r"$\sigma = {:.2f}$".format(kd_GT_sigma))
        plt.close()

        # =============================================================================
        # Plot Error Distribution
        # =============================================================================
        plt.hist(pd_error_map.flatten(),
                    bins=np.logspace(np.log10(1e-5),np.log10(np.max(pd_error_map).flatten()),50).flatten()
                    )
        plt.xscale("log")
        plt.yscale("log")
        plt.close()
        
        repetion_pd_pinn +=1

# %% Print Paths
print("###############################")
print("Path saved model: {}".format(model_path_save))
print("###############################")
