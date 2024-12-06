# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 07:23:40 2024

@author: Diego
"""

import os
import pandas as pd

class DataManager:
    
    def __init__(self) -> None:
        
        self.root_path      = os.path.abspath(
            os.path.join((os.path.abspath(
                os.path.join(os.getcwd(), os.pardir))), os.pardir))
        
        self.data_path      = os.path.join(self.root_path, "data")
        self.raw_data_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_data_path) == False: os.makedirs(self.raw_data_path)
        
        self.bbg_data_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        
        self.bbg_xlsx_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\root\BBGTickers.xlsx"
        if os.path.exists(self.bbg_xlsx_path) == False: 
            self.bbg_xlsx_path = r"/Users/diegoalvarez/Desktop/BBGData/root/BBGTickers.xlsx"
        
        self.bbg_fut_path   = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\root\fut_tickers.xlsx"
        if os.path.exists(self.bbg_fut_path) == False: 
            self.bbg_fut_path = r"/Users/diegoalvarez/Desktop/BBGFuturesManager/root/fut_tickers.xlsx"
        
        self.bbg_front_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\PXFront"
        self.bbg_deliv_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\BondDeliverableRisk"
        
        self.df_fut_tickers = (pd.read_excel(
            io = self.bbg_fut_path, sheet_name = "px"))
        
        self.df_tickers    = (pd.read_excel(
            io = self.bbg_xlsx_path, sheet_name = "tickers"))
        
        self.bad_breakevens = [
            "USGGBE01 Index", "USGGBE09 Index", "USGGBE06 Index", 
            "USGGBE08 Index", "USGGBE03 Index"]
    
    def get_inflation_swap(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "InflationSwaps.parquet")
        try:
            
            if verbose == True: print("Trying to find swap data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, getting it")
            df_tickers = (self.df_tickers.query(
                "Subcategory == 'Interest Rate Swaps'").
                assign(
                    first  = lambda x: x.Description.str.split(" ").str[0],
                    second = lambda x: x.Description.str.split(" ").str[1]).
                query("first == 'USD' & second == 'Inflation'")
                [["Security", "Description"]].
                rename(columns = {"Security": "security"}))
            
            tickers = (df_tickers.assign(
                tmp = lambda x: x.security.str.split(" ").str[0]).
                tmp.
                drop_duplicates().
                sort_values().
                to_list())
            
            files = [os.path.join(
                self.bbg_data_path, ticker + ".parquet")
                for ticker in tickers]
            
            df_out = (pd.read_parquet(
                path = files, engine = "pyarrow").
                drop(columns = ["variable"]).
                merge(right = df_tickers, how = "inner", on = ["security"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_breakeven(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "BreakevenRates.parquet")
        try:
            
            if verbose == True: print("Trying to find Breakeven data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
        
        except: 
        
            if verbose == True: print("Couldn't find data, getting it")
            df_tickers = (self.df_tickers.query(
                "Subcategory == 'Miscellaneous Indices'").
                assign(
                    first  = lambda x: x.Description.str.split(" ").str[0],
                    second = lambda x: x.Description.str.split(" ").str[1]).
                query("first == 'US' & second == 'Breakeven'")
                [["Security", "Description"]].
                rename(columns = {"Security": "security"}))
            
            tickers = (df_tickers.assign(
                tmp = lambda x: x.security.str.split(" ").str[0]).
                tmp.
                drop_duplicates().
                sort_values().
                to_list())
            
            files = [os.path.join(
                self.bbg_data_path, ticker + ".parquet")
                for ticker in tickers]
            
            df_out = (pd.read_parquet(
                path = files, engine = "pyarrow").
                drop(columns = ["variable"]).
                merge(right = df_tickers, how = "inner", on = ["security"]).
                query("security != @self.bad_breakevens"))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _get_tsy_rtn(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(
                PX_rtn  = lambda x: x.PX_LAST.pct_change(),
                PX_diff = lambda x: x.PX_LAST.diff(),
                PX_bps  = lambda x: x.PX_diff / x.PX_dur).
            dropna())
        
        return df_out
    
    def get_tsy_fut(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "TreasuryFutures.parquet")
        try:
            
            if verbose == True: print("Seaching for Treasury futures")
            df_fut = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting") 
            raw_tickers = (self.df_fut_tickers.assign(
                second = lambda x: x.name.str.split(" ").str[-2]).
                query("second == 'Treasury'").
                contract.
                to_list())
            
            deliv_paths = ([
                os.path.join(self.bbg_deliv_path, path_ + ".parquet") 
                for path_ in raw_tickers])
            
            df_deliv = (pd.read_parquet(
                path = deliv_paths, engine = "pyarrow").
                pivot(
                    index   = ["date", "security"], 
                    columns = "variable",
                    values  = "value").
                reset_index().
                rename(columns = {
                    "CONVENTIONAL_CTD_FORWARD_FRSK": "PX_dur",
                    "FUT_EQV_CNVX_NOTL"            : "PX_cnvx"}))
            
            fut_paths = ([
                os.path.join(self.bbg_front_path, ticker + ".parquet") 
                for ticker in raw_tickers])
            
            df_fut = (pd.read_parquet(
                path = fut_paths, engine = "pyarrow").
                merge(right = df_deliv, how = "inner", on = ["date", "security"]).
                dropna().
                groupby("security").
                apply(self._get_tsy_rtn).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_fut.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_fut
        
def main() -> None:
        
    DataManager().get_inflation_swap(verbose = True)
    DataManager().get_breakeven(verbose = True)
    DataManager().get_tsy_fut(verbose = True)
    
if __name__ == "__main__": main()