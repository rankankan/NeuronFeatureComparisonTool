# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:22:29 2021

@author: rankankan
"""
import pandas as pd
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
log_file = open(compltoolconst.LOG_FILE, 'w')

#extract the proofreading data and store it
proof_df = client.materialize.query_table("proofreading_status_public_v1")
write_data(proof_df,compltoolconst.NUC_DATA_PROOF_LARGE)
print("length: ",len(proof_df))
