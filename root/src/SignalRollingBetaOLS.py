# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 06:59:49 2025

@author: Diego
"""

import os
import sys
import numpy as np
import pandas as pd
from   tqdm import tqdm

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

tqdm.pandas()
from DataPreprocess import InflationPCA

class RollingBetaOLS(InflationPCA):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.beta_bootstrap_path = os.path.join(self.data_path, "SignalBetaOLS")
        if os.path.exists(self.beta_bootstrap_path) == False: os.makedirs(self.beta_bootstrap_path)
        self.windows = {
            "weekly"   : 5,
            "two_week" : 5 * 10,
            "monthly"  : 20,
            "two_month": 20 * 2}
        
        self.proportions = [0.10, 0.15, 0.3, 0.5]
        
    def prep_data(self) -> pd.DataFrame:
        
        df_tsy = (self.get_tsy_fut().assign(
            security = lambda x: x.security.str.split(" ").str[0])
            [["date", "security", "PX_bps"]])
        
        df_combined = (self.pca_signal()[
            ["date", "input_val", "variable", "lag_spread", "spread"]].
            merge(right = df_tsy, how = "inner", on = ["date"]).
            assign(group_var = lambda x: x.security + " " + x.variable + " " + x.input_val))
        
        return df_combined
    
    def _rolling_ols(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        df_tmp = (df.sort_values(
            "date").
            set_index("date"))
        
        df_out = (RollingOLS(
            endog  = df_tmp.PX_bps,
            exog   = sm.add_constant(df_tmp.lag_spread),
            window = window).
            fit().
            params.
            rename(columns = {
                "const"     : "alpha",
                "lag_spread": "beta"}).
            sort_index().
            assign(lag_beta = lambda x: x.beta.shift()).
            dropna().
            merge(right = df_tmp, how = "inner", on = ["date"]))
        
        return df_out
    
    def _get_proportion_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        proportions = ({
            value: int(len(df) * value)
            for value in self.proportions})
        
        df_out = (pd.concat([
            self._rolling_ols(df, proportions[key]).assign(prop = key, window = proportions[key])
            for key in proportions.keys()]))
        
        return df_out
    
    def get_proportion(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.beta_bootstrap_path, "RollingProportionBetaBootstrap.parquet")
        try:
        
            if verbose == True: print("Trying to find Rolling Proportion OLS Beta")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, getting it now")
        
            df_out = (self.prep_data().groupby(
                "group_var").
                progress_apply(lambda group: self._get_proportion_ols(group)).
                drop(columns = ["group_var"]).
                reset_index())
            
            if verbose == True: print("Saving Data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_rolling_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._rolling_ols(df, self.windows[window]).assign(period = window, window = self.windows[window])
            for window in self.windows.keys()]))
        
        return df_out
    
    def get_window(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.beta_bootstrap_path, "RollingWindowBetaBootstrap.parquet")
        try:
        
            if verbose == True: print("Trying to find Rolling Window OLS Beta")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, getting it now")
        
            df_out = (self.prep_data().groupby(
                "group_var").
                apply(self._get_rolling_ols).
                drop(columns = ["group_var"]).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
    
    df = RollingBetaOLS().get_window(verbose = True)
    df = RollingBetaOLS().get_proportion(verbose = True)
    
if __name__ == "__main__": main()