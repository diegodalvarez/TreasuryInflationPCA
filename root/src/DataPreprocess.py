# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:51:02 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from CollectData import DataManager
from sklearn.decomposition import PCA

class InflationPCA(DataManager):
    
    def __init__(self) -> None: 
        
        super().__init__()        
        self.pca_path = os.path.join(self.data_path, "Signal")
        if os.path.exists(self.pca_path) == False: os.makedirs(self.pca_path)
        
        self.n_comps = 3
        
    def pre_process(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "PreprocessedData.parquet")
        
        try:
            
            if verbose == True: print("Trying to find PreprocessData")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find the data collecting it it")
        
            df_breakeven = (self.get_breakeven().drop(
                columns = ["Description"]).
                assign(security = lambda x: x.security.str.split(" ").str[0]))
            
            df_inflation = (self.get_inflation_swap().drop(
                columns = ["Description"]).
                assign(security = lambda x: x.security.str.split(" ").str[0]))
            
            df_out = (pd.concat([
                df_breakeven.assign(curve = "breakeven"), 
                df_inflation.assign(curve = "inflation")]).
                rename(columns = {"value": "raw_value"}).
                assign(log_value = lambda x: np.log(x.raw_value)).
                melt(id_vars = ["date", "security", "curve"]))
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_pca(self, df: pd.DataFrame) -> pd.DataFrame:

        df_wider = (df.drop(
            columns = ["group_var"]).
            pivot(index = "date", columns = "security", values = "value").
            dropna())
        
        df_out = (pd.DataFrame(
            data    = PCA(n_components = self.n_comps).fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(self.n_comps)],
            index   = df_wider.index))
        
        return df_out
    
    def _lag_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(lag_spread = lambda x: x.spread.shift()).
            dropna())
        
        return df_out
    
    def pca_signal(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "PCASignal.parquet")
        
        try:
            
            if verbose == True: print("Trying to find PreprocessData")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find the data collecting it it")
            
        
            df_out = (self.pre_process().assign(
                group_var = lambda x: x.curve + " " + x.variable).
                drop(columns = ["curve", "variable"]).
                groupby("group_var").
                apply(self._get_pca).
                reset_index().
                assign(
                    curve     = lambda x: x.group_var.str.split(" ").str[0],
                    input_val = lambda x: x.group_var.str.split(" ").str[1]).
                drop(columns = ["group_var"]).
                melt(id_vars = ["date", "curve", "input_val"]).
                pivot(index = ["date", "input_val", "variable"], columns = "curve", values = "value").
                dropna().
                reset_index().
                assign(spread = lambda x: x.breakeven - x.inflation).
                assign(group_var = lambda x: x.input_val + " " + x.variable).
                groupby("group_var").
                apply(self._lag_signal).
                reset_index(drop = True).
                drop(columns = ["group_var"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> pd.DataFrame: 
       
    df = InflationPCA().pre_process(verbose = True)
    df = InflationPCA().pca_signal(verbose = True)
    
if __name__ == "__main__": main()