# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:15:36 2021

@author: rankankan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import compltoolconst


# Perform cleanup and hygiene
# It expects a datafrane; possible actions are parameterized
# It returns the optimized data frame

def clean_dataset(df, cleft_threshold=50, connection_threshold=1, distance_filter=True, null_ids = True):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    
    #filter out null values, infinite values
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    
    # invoke filter for close synapses
    if distance_filter:
        close_mask = distance_filter_syn_mask(df)
        df = df[close_mask]
        
    # apply cleft threshold
    df.drop(df[df.cleft_score < cleft_threshold].index, inplace=True)
    
    # appy connection threshold
    df.drop(df[df.connection_score < connection_threshold].index, inplace=True)
    if null_ids:
        df.drop(df[df.pre_pt_root_id == 0].index, inplace=True)    
        df.drop(df[df.post_pt_root_id == 0].index, inplace=True)  
    return df[indices_to_keep]#.astype(np.uint64)


# Reshapes the dataframe to work with the required columns
# It expects a datafrane; possible actions are parameterized
# It returns the denormalized data frame
def clean_schema(df, drop_unused = True, split_arrays = False):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    #Discard "unused" columns
    if drop_unused:
        df.drop('id',axis='columns', inplace=True)
        df.drop('valid',axis='columns', inplace=True)
        df.drop('valid_nt',axis='columns', inplace=True)
        df.drop('pre_pt_supervoxel_id',axis='columns', inplace=True)
        df.drop('post_pt_supervoxel_id',axis='columns', inplace=True)
    print("clean_schema: ", df.columns)        
    # denormalize positions column
    #(I found out there is an option in query that performs this..)     
    if split_arrays:
        pre_x = []
        pre_y = []
        pre_z = []
        coordinates = [[]]
        for i in range(len(df)):
            coordinates = (df.loc[i,"pre_pt_position"])
            coordinates = coordinates[1:len(coordinates)-1]
            coordinates = coordinates.split()
            pre_x.append(coordinates[0])
            pre_y.append(coordinates[1])
            pre_z.append(coordinates[2])
        df["pre_x"] = pre_x;
        df["pre_y"] = pre_y;
        df["pre_z"] = pre_z;
        return df


#Filters close synapses
# It expects a valid datafrake; returns a mask array of removed ids.
def distance_filter_syn_mask(synapse_df):
    assert isinstance(synapse_df, pd.DataFrame), "synapse_df needs to be a pd.DataFrame"

    syn_mask = np.ones(len(synapse_df), dtype=np.bool)
    pre_coords = np.array(synapse_df[["pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z"]])
    if len(pre_coords) > 0:
        connections = np.array(synapse_df[["pre_pt_root_id", "post_pt_root_id"]], dtype=np.uint64)
        connections = np.ascontiguousarray(connections)
        connections_v = connections.view(dtype='u8,u8').reshape(-1)
        pre_syn_kdtree = sp.spatial.cKDTree(pre_coords)
    
        clustered_syn_ids = pre_syn_kdtree.query_ball_point(pre_coords, r=150)
        removed_ids = set()
        valid_ids = []
        for i_cl in range(len(clustered_syn_ids)):
            if connections[i_cl, 0] == connections[i_cl, 1]:
                removed_ids.add(i_cl)
                continue
            
            if i_cl in removed_ids:
                continue
    
            if len(clustered_syn_ids[i_cl]) > 1:
                local_cluster_ids = np.array(clustered_syn_ids[i_cl])
                conn_m = connections_v[local_cluster_ids] == connections_v[i_cl]
                for id_ in local_cluster_ids[conn_m]:
                    if id_ == i_cl:
                        continue
                    removed_ids.add(id_)                
            valid_ids.append(i_cl)
        valid_ids = np.array(valid_ids)
        removed_ids = np.array(list(removed_ids))
        assert len(valid_ids) + len(removed_ids) == len(clustered_syn_ids)
        if len(removed_ids) > 0:
            syn_mask[removed_ids] = False
    return syn_mask

#Create dataframe from csv file
#Place-holder function for the moment; additional functionality?
def read_data(name):
    return pd.read_excel(name)

#Store dataframe contents into a csv file
#It allows to work with a subset of the data and avoid loading the backend with requests
#ToDo: error/exception handling
def write_data(df, filename, extract=False, table=None, start=0, end=999):
    if extract:
        table_name = client.info.get_datastack_info()[table]
        df = client.materialize.query_table(table_name)
        df = df.loc[start:end]
    df.to_csv(filename, header=True)
  
    def get_synapse_info(direction, feature_df, ind):
    print ("direction: ", direction)
    if direction == "incoming":
        output_df = pd.DataFrame(client.materialize.synapse_query(post_ids=current_id, split_positions=True))
    else:
        output_df = pd.DataFrame(client.materialize.synapse_query(pre_ids=current_id, split_positions=True))       
    assert isinstance(output_df, pd.DataFrame), "output df needs to be a pd.DataFrame" 

#    print("HEAD BEFORE processing: ", output_df.head(2), file=log_file)
    if len(output_df) > 0:
        output_df.dropna(inplace=True)  # this 3 lines come from cleandataset
        indices_to_keep = ~output_df.isin([np.nan, np.inf, -np.inf]).any(1)
        output_df = output_df[indices_to_keep]
        
        close_mask = distance_filter_syn_mask(output_df)
        output_df = output_df[close_mask]    
                
        gaba_mean = direction == "incoming" and "i_gaba_mean" or "o_gaba_mean"
        ach_mean = direction == "incoming" and "i_ach_mean" or "o_ach_mean"
        glut_mean = direction == "incoming" and "i_glut_mean" or "o_glut_mean"
        oct_mean = direction == "incoming" and "i_oct_mean" or "o_oct_mean"
        ser_mean = direction == "incoming" and "i_ser_mean" or "o_ser_mean"
        da_mean = direction == "incoming" and "i_da_mean" or "o_da_mean"  
        
        gaba_std = direction == "incoming" and "i_gaba_std" or "o_gaba_std"
        ach_std = direction == "incoming" and "i_ach_std" or "o_ach_std"
        glut_std = direction == "incoming" and "i_glut_std" or "o_glut_std"
        oct_std = direction == "incoming" and "i_oct_std" or "o_oct_std"
        ser_std = direction == "incoming" and "i_ser_std" or "o_ser_std"
        da_std = direction == "incoming" and "i_da_std" or "o_da_std"
         
        feature_df.at[ind,gaba_mean] = np.mean(output_df[["gaba"]])[0]
        feature_df.at[ind,ach_mean] = np.mean(output_df[["ach"]])[0]
        feature_df.at[ind,glut_mean] = np.mean(output_df[["glut"]])[0]
        feature_df.at[ind,oct_mean] = np.mean(output_df[["oct"]])[0]
        feature_df.at[ind,ser_mean] = np.mean(output_df[["ser"]])[0]
        feature_df.at[ind,da_mean] = np.mean(output_df[["da"]])[0]
      
        feature_df.at[ind,gaba_std] = np.std(output_df[["gaba"]])[0]
        feature_df.at[ind,ach_std] = np.std(output_df[["ach"]])[0]
        feature_df.at[ind,glut_std] = np.std(output_df[["glut"]])[0]
        feature_df.at[ind,oct_std] = np.std(output_df[["oct"]])[0]
        feature_df.at[ind,ser_std] = np.std(output_df[["ser"]])[0]
        feature_df.at[ind,da_std] = np.std(output_df[["da"]])[0]
        
        feature_df.at[ind,direction] = len(output_df)

###
### *** MAIN PROGRAM *** ###
###
        
#Set up CAVE client
from caveclient import CAVEclient
datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)
log_file = open(compltoolconst.LOG_FILE, 'w')

feature_df = pd.read_pickle(compltoolconst.SYNAPSES_PICKLE)
try:        
    ind2 = 0
    for ind in range(137524,len(feature_df)):
        current_id = feature_df.at[ind, 'root_id']
        print("\nrecord_id: ", current_id, ind, file=log_file)        
        if current_id != 0:
            get_synapse_info("incoming", feature_df, ind)
            get_synapse_info("outgoing", feature_df, ind)
        ind2 += 1
        if ind2 == compltoolconst.PICKLE_LIMIT:
            ind2 = 0
            print("\nPICKLE: ", current_id, ind, file=log_file)
            feature_df.to_pickle(compltoolconst.SYNAPSES_PICKLE)
            feature_df = pd.read_pickle(compltoolconst.SYNAPSES_PICKLE)
            write_data(feature_df,compltoolconst.FEATURE_DATA_ALL_LATEST)
            
    print("\nFEATURE: ", feature_df.head(), file=log_file)
    
    #some cleanup just in case
    feature_df['partition'] = feature_df['partition'].fillna(0)
    feature_df['incoming'] = feature_df['incoming'].fillna(0)
    
except Exception as e:
    print(e)
    print(e, file=log_file)
finally:
    ##Store feature table
    write_data(feature_df,compltoolconst.FEATURE_DATA_ALL_LATEST)
    feature_df.to_pickle(compltoolconst.SYNAPSES_PICKLE)
    print("\nFINALLY: ", file=log_file)
    log_file.close()





 
