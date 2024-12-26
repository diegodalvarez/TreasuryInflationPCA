# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 07:31:23 2024

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
        
    def _get_pca(self, df: pd.DataFrame) -> pd.DataFrame:
      
        df_wider = (df.drop(
            columns = ["curve"]).
            pivot(index = "date", columns = "security", values = "value").
            dropna())
        
        df_out = (pd.DataFrame(
            data    = PCA(self.n_comps).fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(self.n_comps)],
            index   = df_wider.index))
        
        return df_out
    
    def _lag_spread(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(lag_spread = lambda x: x.spread.shift()).
            dropna())
        
    def get_pca(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "PCASignal.parquet")
        try:
            
            if verbose == True: print("Trying to find PCA data")
            df_combined = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting it")
            df_inflation = (self.get_inflation_swap().drop(
                columns = ["Description"]).
                assign(
                    curve    = "inflation_swap",
                    security = lambda x: x.security.str.split(" ").str[0]))
            
            df_breakeven = (self.get_breakeven().drop(
                columns = ["Description"]).
                assign(
                    curve    = "tsy_breakeven",
                    security = lambda x: x.security.str.split(" ").str[0]))
            
            df_combined = (pd.concat([
                df_inflation, df_breakeven]).
                groupby("curve").
                apply(self._get_pca).
                reset_index().
                melt(id_vars = ["date", "curve"]).
                pivot(index = ["date", "variable"], columns = "curve", values = "value").
                dropna().
                reset_index().
                assign(spread = lambda x: x.tsy_breakeven - x.inflation_swap).
                groupby("variable").
                apply(self._lag_spread).
                reset_index(drop = True))
        
            if verbose == True: print("Saving data\n")
            df_combined.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_combined
    
    def get_log_pca(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "LogPCASignal.parquet")
        try:
            
            if verbose == True: print("Trying to find Log PCA data")
            df_combined = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting it")
            df_inflation = (self.get_inflation_swap().drop(
                columns = ["Description"]).
                assign(
                    security = lambda x: x.security.str.split(" ").str[0],
                    value    = lambda x: np.log(x.value),
                    curve    =  "inflation_swap"))
            
            df_breakeven = (self.get_breakeven().drop(
                columns = ["Description"]).
                assign(
                    security = lambda x: x.security.str.split(" ").str[0],
                    value    = lambda x: np.log(x.value),
                    curve    = "tsy_breakeven"))
            
            df_combined = (pd.concat([
                df_inflation, df_breakeven]).
                groupby("curve").
                apply(self._get_pca).
                reset_index().
                melt(id_vars = ["date", "curve"]).
                pivot(index = ["date", "variable"], columns = "curve", values = "value").
                dropna().
                reset_index().
                assign(spread = lambda x: x.tsy_breakeven - x.inflation_swap).
                assign(spread = lambda x: np.where(x.variable == "PC3", -x.spread, x.spread)).
                groupby("variable").
                apply(self._lag_spread).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_combined.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_combined
    
def main() -> None:
    
    InflationPCA().get_pca(verbose = True)
    InflationPCA().get_log_pca(verbose = True)

if __name__ == "__main__": main()
