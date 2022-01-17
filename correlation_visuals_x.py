# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:58:15 2021

@author: rankankan
"""
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import compltoolconst
from numpy import std
import umap as um


from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log_file = open(compltoolconst.LOG_FILE, 'w')

#Utility functions
def read_data(name):
    return pd.read_csv(name)

def write_data(df, filename):
    df.to_csv(filename, header=True)

def clean_up(df):
    df.drop(df.columns[0:5], axis=1, inplace=True)
    print("df aft: ",df.columns)
    df.drop(['volume'], axis=1, inplace=True)
    df=df.dropna()
    return(df)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.drop(df.columns[0], axis=1, inplace=True)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

###
### *** Main Program *** ###
###
    
log_file = open(compltoolconst.LOG_FILE_CORR, 'w')
consolidated_df = read_data(compltoolconst.CONSOLIDATED_ALL_LARGE)
# smaller sample as requested by Sven
consolidated_df = consolidated_df.sample(n=10000, random_state=22)

#There were some issues because this column was not numeric and - for lack of knowing better - had to switch back and
# forth; would do it differently now.
consolidated_df.loc[(consolidated_df.proofread == 't'), 'proofread'] = 1
consolidated_df.loc[(consolidated_df.proofread == 'f'), 'proofread'] = 0
consolidated_df = clean_dataset(consolidated_df)
target = consolidated_df.loc[:,'proofread']
consolidated_df.drop(['proofread'], axis=1, inplace=True)


##PCA
scaler = StandardScaler()
scaler.fit(consolidated_df)
scaled_consolidated=scaler.transform(consolidated_df)    
pca = PCA(n_components=4)
x_new = pca.fit_transform(scaled_consolidated)
print("\nVariance ratio: \n",pca.explained_variance_ratio_)
print("\nVariance ratio: \n",pca.explained_variance_ratio_,file=log_file)
print("\n\nAbsolutes: \n", abs(pca.components_))
print("\n\nAbsolutes: \n", abs(pca.components_),file=log_file)

# define list of column names which after analysis of the absolutes showed to be the most relevant
cols = ['i_da_mean', 'i_ach_std', 'o_oct_mean', 'o_ser_mean', 'o_da_mean']

# Create a Df by slicing the source DataFrame
pca_subset_df = consolidated_df[cols]


##UMAP
try:
    # neighbors to be set lower to 20 as requested by Sven
    feat_umap = um.UMAP(random_state=999, n_neighbors=50, n_components= 3, min_dist=0) 
    # Fit UMAP and extract latent vars 1-2
    embedding = pd.DataFrame(feat_umap.fit_transform(consolidated_df), columns = ['UMAP1','UMAP2','UMAP3'])
    print("UMAP: ", embedding.head())
    print("UMAP: ", embedding.head(),file=log_file)
    
    # Produce sns.scatterplot and pass metadata.subclasses as color
    #UMAP1 vs UMAP2
    sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding, hue=target, 
                               style=target, palette=['blue','red'], edgecolor='black', alpha=.5)
    # Adjust legend
    sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.figure(figsize = (30,16))
    # Save PNG
    sns_plot.figure.savefig('umap12_X_scatter_pca_50_neighbors_0_dist.png', bbox_inches='tight', dpi=500)
    
    #UMAP1 vs UMAP3
    sns_plot = sns.scatterplot(x='UMAP1', y='UMAP3', data=embedding, hue=target, 
                              style=target, palette=['blue','red'], edgecolor='black', alpha=.5)
    sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.figure(figsize = (30,16))
    sns_plot.figure.savefig('umap13_X_scatter_pca_50_neighbors_0_dist.png', bbox_inches='tight', dpi=500)
    
    
    #UMAP2 vs UMAP3
    sns_plot = sns.scatterplot(x='UMAP2', y='UMAP3', data=embedding, hue=target, 
                              style=target, palette=['blue','red'], edgecolor='black', alpha=.5)
    sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.figure(figsize = (30,16))
    sns_plot.figure.savefig('umap23_X_scatter_pca_50_neighbors_0_dist.png', bbox_inches='tight', dpi=500)
    

    # PairGrid plot
    fig = plt.gcf()
    fig.set_size_inches(15, 10)   
    g = sns.PairGrid(embedding)
    g.map_upper(sns.scatterplot, hue=target, style=target, palette=['blue','red'], edgecolor='black', alpha=.5)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2, fill=True)
    g.add_legend()
    plt.show()

except Exception as e:
    print(e)
    print(e, file=log_file)
finally:

    plt.savefig('umap123_XX_scatter.png', bbox_inches='tight', dpi=500)
    print("\nFINALLY: ", file=log_file)
    log_file.close()

