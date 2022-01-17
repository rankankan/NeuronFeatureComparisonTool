# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:31:16 2021

@author: rankankan
"""
import pandas as pd
import numpy as np
import compltoolconst


def read_data(name):
    return pd.read_csv(name)

def write_data(df, filename):
    df.to_csv(filename, header=True)
    

###
### *** Main Program *** ###
###
    
log_file = open(compltoolconst.LOG_FILE, 'w')
ground_df = pd.DataFrame()

#All feature data
feature_df = read_data(compltoolconst.FEATURE_DATA_ALL_LATEST)
#feature_df = feature_df.drop(feature_df[feature_df['incoming'] < 1].index) #watch
write_data(feature_df,compltoolconst.FEATURE_DATA_ALL_LATEST_LARGE)
print("inc: ", feature_df.size)

#Get the nucleus data
proof_df = read_data(compltoolconst.NUC_DATA_PROOF_LARGE)

#create subset with merge
proof_df_revised = proof_df.drop_duplicates('root_id')                          #To be adapted after talk with Sven
merged_data_df = pd.merge(left=feature_df, right=proof_df_revised, how='inner', on='root_id')
write_data(merged_data_df,compltoolconst.CONSOLIDATED_GROUND_SUBSET_LARGE_X)


## Create subset WITHOUT merge - not used because data quality looks better with merge
#for ind in range(0,len(proof_df)):
#    current_id = proof_df.at[ind, 'root_id'].astype(np.int64)
#    if current_id != 0: 
#        found = feature_df[(feature_df["root_id"] == current_id)]
#        ground_df = ground_df.append(found) # index?
#        write_data(ground_df,compltoolconst.CONSOLIDATED_GROUND_SUBSET_LARGE)
#


# Further "massaging" of the data to test different scenarios and/or to subset data
# Some of this code was commented out depending on what was needed and or tried out
all_df = read_data(compltoolconst.FEATURE_DATA_ALL_LATEST_LARGE)
# this was used to make sure that the proofread column was correctly aligned. 
all_df["proofread"] = 'f'
for i in range(len(merged_data_df)):
    all_df.loc[merged_data_df.at[i, 'id_x'], 'proofread'] = 't'

all_df = all_df.drop(all_df[all_df['incoming'] < 1].index) #Adapt: according to Sven 0 incoming should also be considered
consolidated_df = all_df
# clean up; removing unnamed columns
consolidated_df.drop(consolidated_df.columns[0:5], axis=1, inplace=True)
# clean up to remove volume until feature data is available
consolidated_df.drop(['volume'], axis=1, inplace=True)
write_data(consolidated_df,compltoolconst.CONSOLIDATED_ALL_LARGE)        