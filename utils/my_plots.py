import matplotlib
import matplotlib.pyplot as plt


# %% Functions

def colorbar(mappable,**kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.minorticks_on()
    plt.sca(last_axes)
    return cbar

# %% Colormaps

from cmcrameri import cm as cmcrameri
import seaborn as sns

my_cmap = {"p":"#009EB3", # some kind of teal
           "v":"chocolate",
           "k":"orange",
           # "GT":"#107B9E",
           "GT":"#5FB5CF", # gray
           "pred":"#EB174F",
           # "CP":"#EBAC00",
           "CP":"#6E655D",
           "p_map":cmcrameri.vik,
           # "k_map":cm.bamako,
           "k_map":sns.color_palette("flare",as_cmap=True),
           # "error_map":cm.buda,
           "error_map":cmcrameri.acton,
           "error_map":sns.color_palette("rainbow",as_cmap=True),
           }


# %% Settings

kwargs_plot = {
"GT":{"linestyle":"solid","color":my_cmap["GT"],"label":"Groundtruth","linewidth":3},
"pred":{"linestyle":"dashed","color":my_cmap["pred"],"label":"Prediction","linewidth":3},
"interpolation":{"s":1,"color":my_cmap["pred"],"label":"Regression"},
    }

kwargs_plot_pred = {"linestyle":"dashed","color":my_cmap["pred"],"label":"Prediction"}
kwargs_plot_interpolation = {"s":1,"color":my_cmap["pred"],"label":"Regression"}

params = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    # 'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    # 'axes.titlesize': 12,
    # 'font.size': 12, # was 10
    'font.weight': 400,
    'legend.facecolor': "white", # was 10
    'legend.title_fontsize': "x-small", # was 10

    # 'xtick.labelsize': 12,
    # 'ytick.labelsize': 12,
    'text.usetex': False,
    # 'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{amsmath}',
    'figure.figsize': [16,9],
    'font.family': 'sans', # "serif", "sans"
}
matplotlib.rcParams.update(params)