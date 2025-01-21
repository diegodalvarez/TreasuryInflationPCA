# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:02:28 2025

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

class BootstrapBetaOLS(InflationPCA):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.beta_bootstrap_path = os.path.join(self.data_path, "SignalBetaOLS")
        if os.path.exists(self.beta_bootstrap_path) == False: os.makedirs(self.beta_bootstrap_path)
        
    def _beta(self, df: pd.DataFrame, sample_size: float) -> float:
        
        df_tmp = df.sample(frac = sample_size)
        alpha, beta = (sm.OLS(
            endog = df_tmp.PX_bps,
            exog  = sm.add_constant(df_tmp.lag_spread)).
            fit().
            params)
        
        return beta
    
    def _sample_beta(self, df: pd.DataFrame, sample_size: float, sims: int) -> pd.DataFrame: 
        
        betas = [self._beta(df, sample_size) for i in tqdm(range(sims), desc = "Sampling {}".format(df.name))]
        
        df_out = (pd.DataFrame({
            "beta": betas,
            "sim" : [i + 1 for i in range(sims)]}))
        
        return df_out
    
    def sample_betas(
            self, 
            sample_size: float = 0.3, 
            sims       : int = 10_000,
            verbose    : bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.beta_bootstrap_path, "OLSBetaBootstrap.parquet")
        try:
        
            if verbose == True: print("Trying to find Bootstrapped OLS Beta")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, getting it now")
        
            df_tmp = (self.pca_signal().drop(
                columns = ["breakeven", "inflation"]).
                merge(right = InflationPCA().get_tsy_fut(), how = "inner", on = ["date"]).
                drop(columns = ["PX_LAST", "PX_dur", "PX_cnvx", "PX_rtn", "PX_diff"]).
                assign(
                    security  = lambda x: x.security.str.split(" ").str[0],
                    group_var = lambda x: x.input_val + " " + x.security + " " + x.variable))
        
            df_out = (df_tmp.groupby(
                "group_var").
                apply(self._sample_beta, sample_size, sims).
                reset_index().
                drop(columns = ["level_1"]).
                assign(
                    input_val = lambda x: x.group_var.str.split(" ").str[0],
                    security  = lambda x: x.group_var.str.split(" ").str[1],
                    variable  = lambda x: x.group_var.str.split(" ").str[2]).
                drop(columns = ["group_var"]))
            
            if verbose == True: print("\nSaving Data\n")
            df_out.to_parquet(file_path, engine = "pyarrow")
            
        return df_out
    
def main() -> None: 
    
    df = BootstrapBetaOLS().sample_betas(verbose = True)
    
if __name__ == "__main__": main()