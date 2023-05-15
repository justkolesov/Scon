import termcolor
import argparse
import wandb
import os
import numpy as np
from PIL import Image
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img( fig ):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    

    
def pca_plot(y_samples, xn, stage ):
    
 
    fig,axes = plt.subplots( 1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(y_samples.cpu().numpy())
    map_data_emb = pca.transform(xn)
    real_data_emb = pca.transform(y_samples.cpu().numpy())
    
    axes[0].scatter( map_data_emb[:,0], map_data_emb[:,1], c="bisque", edgecolor = 'black', label = r'$x\sim T(x)$', s =30)
    axes[1].scatter( real_data_emb[:,0], real_data_emb[:,1], c="salmon", edgecolor = 'black', label = r'$x\sim Q(x)$', s =30)
    axes[2].scatter( map_data_emb[:,0], map_data_emb[:,1], c="bisque", edgecolor = 'black', label = r'$x\sim T(x)$', s =30)
    axes[2].scatter( real_data_emb[:,0], real_data_emb[:,1], c="salmon", edgecolor = 'black', label = r'$x\sim Q(x)$', s =30)

    fig.tight_layout(pad=0.5)
    wandb.log({ 'Plot PCA samples' : [wandb.Image(fig2img(fig))]}, step=stage)
 