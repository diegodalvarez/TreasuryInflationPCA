# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:41:45 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from DataPreprocess import InflationPCA


class PCASpreadOLS(InflationPCA):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.ols_path  = os.path.join(self.data_path, "OLSModel")
        self.num_comps = 3
        
        if os.path.exists(self.ols_path) == False: os.makedirs(self.ols_path)
        
    def _full_sample_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        model = (sm.OLS(
            endog = df.PX_bps,
            exog  = sm.add_constant(df[["PC{}".format(i + 1) for i in range(self.num_comps)]])).
            fit())
        
        df_out = (df.assign(
            predict     = model.predict(),
            lag_predict = lambda x: x.predict.shift()))
        
        return df_out
        
    def full_sample_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_path, "FullSampleModel.parquet")
        
        try:
            
            if verbose == True: print("Trying to find Full Sample OLS outputs")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find PCA OLS Full Sample Model, collecting it")
                
            df_pc_wider = (self.get_log_pca()[
                ["date", "variable", "spread"]].
                pivot(index = "date", columns = "variable", values = "spread"))
            
            df_tsy_rtn = (self.get_tsy_fut()[
                ["date", "security", "PX_bps"]].
                assign(security = lambda x: x.security.str.split(" ").str[0]))
            
            df_out = (df_pc_wider.merge(
                right = df_tsy_rtn, how = "inner", on = ["date"]).
                groupby("security").
                apply(self._full_sample_ols).
                reset_index(drop = True).
                dropna().
                assign(signal_rtn = lambda x: np.sign(x.lag_predict) * x.PX_bps))
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _rolling_ols(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (RollingOLS(
            endog  = df_tmp.PX_bps,
            exog   = sm.add_constant(df_tmp.drop(columns  = ["PX_bps"])),
            window = window).
            fit().
            params.
            merge(right = df_tmp[["PX_bps"]], how = "inner", on = ["date"]).
            reset_index().
            melt(id_vars = ["date", "const", "PX_bps"]))
        
        return df_out
    
    def get_rolling_regression(self, window: int = 90, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_path, "RollingOLSMOdel.parquet")
        
        try:
            
            if verbose == True: print("Trying to find Rolling OLS outputs")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find PCA OLS Full Sample Model, collecting it")
            df_pc_wider = (self.get_log_pca()[
                ["date", "variable", "spread"]].
                pivot(index = "date", columns = "variable", values = "spread"))
            
            df_tsy_rtn = (self.get_tsy_fut()[
                ["date", "security", "PX_bps"]].
                assign(security = lambda x: x.security.str.split(" ").str[0]))
            
            df_longer = (df_pc_wider.reset_index().melt(
                id_vars = "date"))
            
            df_out = (df_pc_wider.merge(
                right = df_tsy_rtn, how = "inner", on = ["date"]).
                groupby("security").
                apply(self._rolling_ols, window, include_groups = False).
                reset_index().
                drop(columns = ["level_1"]).
                rename(columns = {
                    "value": "beta",
                    "const" : "alpha"}).
                dropna().
                merge(right = df_longer, how = "inner", on = ["date", "variable"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:         

    df = PCASpreadOLS().full_sample_ols(verbose = True)
    df = PCASpreadOLS().get_rolling_regression(verbose = True)
    
if __name__ == "__main__": main()