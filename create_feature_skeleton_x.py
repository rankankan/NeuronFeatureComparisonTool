# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:41:26 2021

@author: rankankan
"""

# create empty feature dataframe / file

import pandas as pd
import numpy as np
import compltoolconst

#Create dataframe from csv file
#Place-holder function for the moment; additional functionality?
def read_data(name):
    return pd.read_csv(name)

#Store dataframe contents into a csv file
#It allows to work with a subset of the data and avoid loading the backend with requests
#ToDo: error/exception handling
def write_data(df, filename, extract=False, table=None, start=0, end=999):
    if extract:
        table_name = client.info.get_datastack_info()[table]
        df = client.materialize.query_table(table_name)
        df = df.loc[start:end]
    df.to_csv(filename, header=True)
    
from caveclient import CAVEclient
datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)
log_file = open(compltoolconst.LOG_FILE, 'w+')

#Create feature Dataframe (probably there are better data structures to accomplish this)
feature_df = pd.DataFrame(columns = ['root_id','incoming', 'i_gaba_mean', 'i_ach_mean', 'i_glut_mean', 'i_oct_mean', 'i_ser_mean', 'i_da_mean', 'i_gaba_std', 'i_ach_std', 'i_glut_std', 'i_oct_std', 'i_ser_std', 'i_da_std','outgoing', 'o_gaba_mean', 'o_ach_mean', 'o_glut_mean', 'o_oct_mean', 'o_ser_mean', 'o_da_mean', 'o_gaba_std', 'o_ach_std', 'o_glut_std', 'o_oct_std', 'o_ser_std', 'o_da_std', 'volume', 'partition'])
#Get the nucleus data
nuc_df = read_data(compltoolconst.NUC_DATA_ALL)
#..and the partitions
nuc_partition_df = read_data(compltoolconst.NUC_PARTITIONS)
merged_nuc_df = pd.merge(left=nuc_df, right=nuc_partition_df, how='left', left_on='id', right_on='id')
feature_df.insert(0, 'id', range(0, len(merged_nuc_df)))
feature_df.root_id = merged_nuc_df.pt_root_id.astype(np.int64)
feature_df.set_index('root_id')
feature_df.partition = merged_nuc_df.partition.astype(np.int64)
write_data(feature_df, compltoolconst.FEATURE_DATA_ALL_LATEST)
