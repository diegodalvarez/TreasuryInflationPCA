# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:41:45 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   tqdm import tqdm

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from DataPreprocess import InflationPCA

tqdm.pandas()

class PCASpreadOLS(InflationPCA):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.ols_path  = os.path.join(self.data_path, "OLSModel")
        self.num_comps = 3
        
        if os.path.exists(self.ols_path) == False: os.makedirs(self.ols_path)
        
    def _get_signal_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        model = (sm.OLS(
            endog = df.PX_bps,
            exog  = sm.add_constant(df.lag_spread)).
            fit())
    
        alpha, beta = model.params
        palpha, pbeta = model.pvalues
        
        df_out = (df.assign(
            alpha   = alpha,
            beta    = beta,
            p_alpha = palpha,
            p_beta  = pbeta))
        
        return df_out
        
    def get_signal_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_path, "SignalOLS.parquet")
        try:
            
            if verbose == True: print("Seaching for Signal OLS futures")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting") 
        
            df_out = (self.pca_signal().drop(
                columns = ["inflation", "breakeven", "spread"]).
                merge(right = self.get_tsy_fut(), how = "inner", on = ["date"]).
                drop(columns = ["PX_rtn", "PX_diff", "PX_cnvx", "PX_dur", "PX_LAST"]).
                assign(
                    security  = lambda x: x.security.str.split(" ").str[0],
                    group_var = lambda x: x.security + " " + x.input_val + " " + x.variable).
                groupby("group_var").
                apply(self._get_signal_ols).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_rolling_signal_ols(self, df: pd.DataFrame, ratio: int) -> pd.DataFrame:
        
        df_tmp = (df[
            ["date", "lag_spread", "PX_bps"]].
            set_index("date"))
        
        window = int(len(df) * ratio)
        
        df_out = (RollingOLS(
            endog  = df_tmp.PX_bps,
            exog   = sm.add_constant(df_tmp.lag_spread),
            window = window).
            fit().
            params.
            rename(columns = {"lag_spread": "roll_beta"}).
            assign(lag_beta = lambda x: x.roll_beta.shift()).
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out
        
    def rolling_ols_signal(self, ratio: float = 0.3, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_path, "SignalRollingOLS.parquet")
        try:
            
            if verbose == True: print("Seaching for Rolling Signal OLS futures")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting") 
        
            df_out = (self.pca_signal().drop(
                columns = ["inflation", "breakeven", "spread"]).
                merge(right = self.get_tsy_fut(), how = "inner", on = ["date"]).
                drop(columns = ["PX_rtn", "PX_diff", "PX_cnvx", "PX_dur", "PX_LAST"]).
                assign(
                    security  = lambda x: x.security.str.split(" ").str[0],
                    group_var = lambda x: x.security + " " + x.input_val + " " + x.variable).
                groupby("group_var").
                progress_apply(lambda group: self._get_rolling_signal_ols(group, ratio)).
                reset_index(drop = True).
                drop(columns = ["group_var"]).
                dropna().
                assign(signal_bps = lambda x: np.sign(x.lag_beta * x.lag_spread) * x.PX_bps))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    

def main() -> None:
            
    df = PCASpreadOLS().get_signal_ols(verbose = True)
    df = PCASpreadOLS().rolling_ols_signal(verbose = True)
    
if __name__ == "__main__": main()