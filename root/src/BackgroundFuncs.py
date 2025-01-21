#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:10:58 2024

@author: diegoalvarez
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from   matplotlib.ticker import FuncFormatter

from DataPreprocess import InflationPCA

class BackgroundHelperFuncs(InflationPCA):
    
    def __init__(self) -> None:
        
        super().__init__()
        
    def get_correlation(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_corr = (df.drop(
            columns = ["spread", "lag_spread"]).
            rename(columns = {
                "inflation_swap": "Inflation\nSwap",
                "tsy_breakeven" : "Treasury\nBreakeven"}).
            melt(id_vars = ["date", "variable"]).
            assign(variable = lambda x: x.curve + " " + x.variable)
            [["date", "variable", "value"]].
            pivot(index = "date", columns = "variable", values = "value").
            corr())
        
        return df_corr
    
    def plot_correlation(self, df_corr: pd.DataFrame) -> plt.Figure: 
        
        fig, axes = plt.subplots(figsize = (8,6))
        sns.heatmap(
            data   = df_corr,
            annot  = True,
            ax     = axes)
    
        axes.set_xlabel("Curve PC")
        axes.set_ylabel("Curve PC")
        axes.set_title("Correlation of PCs")
        plt.tight_layout()
        
        
    def plot_cross_correlations(self, df_corr: pd.DataFrame) -> plt.Figure: 
        
        fig, axes = plt.subplots(figsize = (12,6))
    
        (df_corr.reset_index().rename(
            columns = {"variable": "var1"}).
            melt(id_vars = "var1").
            query("variable != var1").
            assign(
                end1 = lambda x: x.var1.str.split(" ").str[-1],
                end2 = lambda x: x.variable.str.split(" ").str[-1]).
            query("end1 == end2").
            groupby("end1").
            head(1)
            [["end1", "value"]].
            rename(columns = {"end1": "PC"}).
            set_index("PC").
            plot(
                ax     = axes,
                kind   = "bar",
                legend = False,
                xlabel = "",
                ylabel = "Cross Correlation",
                title  = "Cross-Correlations of Inflation Swap Curve and Treasury Breakeven Curve"))
    
        axes.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        
    def get_backtest(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_backtest = (df.merge(
            right = InflationPCA().get_tsy_fut(), how = "inner", on = ["date"]).
            assign(signal_bps = lambda x: -np.sign(x.lag_spread) * x.PX_bps))
        
        return df_backtest
    
    def get_log_backtest(self, df: pd.DataFrame) -> pd.DataFrame: 
    
        df_backtest = (InflationPCA().get_log_pca().merge(
            right = InflationPCA().get_tsy_fut(), how = "inner", on = ["date"]).
            assign(signal_bps = lambda x: np.sign(x.lag_spread) * x.PX_bps))
        
        return df_backtest

    
    def plot_signal_correlation(self, df_backtest: pd.DataFrame) -> plt.Figure: 
        
        variables = df_backtest.variable.drop_duplicates().sort_values().to_list()
        fig, axes = plt.subplots(ncols = len(variables), figsize = (20,6))
    
        for variable, ax in zip(variables, axes.flatten()):
    
            df_corr = (df_backtest.query(
                "variable == @variable")
                [["date", "signal_bps", "security"]].
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                pivot(index = "date", columns = "security", values = "signal_bps").
                corr())
    
            sns.heatmap(
                data  = df_corr,
                annot = True,
                ax    = ax)
    
            ax.set_xlabel("Treasury Future")
            ax.set_ylabel("Treasury Future")
            ax.set_title(variable)
    
        fig.suptitle("Corrleation of Returns")
        plt.tight_layout()
        
    def plot_cum_rtn(self, df_backtest: pd.DataFrame) -> plt.Figure: 
        
        variables = df_backtest.variable.drop_duplicates().sort_values().to_list()
        fig, axes = plt.subplots(ncols = len(variables), figsize = (20,6))
    
        for variable, ax in zip(variables, axes.flatten()):
    
            (df_backtest.query(
                "variable == @variable")
                [["date", "security", "signal_bps"]].
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                rename(columns = {"security": ""}).
                pivot(index = "date", columns = "", values = "signal_bps").
                cumsum().
                plot(
                    ylabel = "Cumulative Basis Points",
                    title  = variable,
                    xlabel = "Date",
                    ax     = ax))
    
        fig.suptitle("Cumulative Return of Inflation & Treasury PC Spread From {} to {}".format(
            df_backtest.date.min().date(),
            df_backtest.date.max().date()))
        
        plt.tight_layout()
        
    def plot_sharpe(self, df_backtest: pd.DataFrame) -> plt.Figure: 
        
        (df_backtest[
            ["variable", "security", "signal_bps"]].
            groupby(["variable", "security"]).
            agg(["mean", "std"])
            ["signal_bps"].
            rename(columns = {
                "mean": "mean_rtn",
                "std" : "std_rtn"}).
            assign(sharpe = lambda x: x.mean_rtn / x.std_rtn * np.sqrt(252)).
            reset_index()
            [["variable", "security", "sharpe"]].
            assign(security = lambda x: x.security.str.split(" ").str[0]).
            rename(columns = {"variable": ""}).
            pivot(index = "security", columns = "", values = "sharpe").
            plot(
                kind    = "bar",
                ylabel  = "Annualized Sharpe",
                figsize = (12,6),
                title   = "Annualized Sharpe per each Strategy"))
        
        plt.tight_layout()
        
    def _get_vol(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(
                vol     = lambda x: x.signal_bps.rolling(window = window).std(),
                lag_vol = lambda x: x.vol.shift()))

    def _get_erc(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.sort_values(
            "date").
            assign(inv_vol = lambda x: 1 / x.lag_vol))
        
        df_out = (df_tmp[
            ["date", "inv_vol"]].
            groupby("date").
            agg("sum").
            rename(columns = {"inv_vol": "full_vol"}).
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(weight = lambda x: x.inv_vol / x.full_vol))
        
        return df_out

    def get_erc_weighting(self, df_backtest: pd.DataFrame, window: int = 30) -> pd.DataFrame: 
    
        df_weight = (df_backtest.groupby([
            "variable", "security"]).
            apply(self._get_vol, window).
            reset_index(drop = True).
            dropna().
            groupby("variable").
            apply(self._get_erc).
            reset_index(drop = True).
            assign(weight_rtn = lambda x: x.weight * x.signal_bps))
        
        return df_weight

    def get_erc_rtn(self, df_weight: pd.DataFrame) -> pd.DataFrame: 
        
        df_erc = (df_weight[
            ["date", "variable", "weight_rtn"]].
            groupby(["date", "variable"]).
            agg("sum").
            reset_index().
            rename(columns = {"variable": ""}).
            pivot(index = "date", columns = "", values = "weight_rtn"))
        
        return df_erc
        
    def plot_erc_rtn(self, df_erc: pd.DataFrame) -> pd.DataFrame:
        
        (df_erc.cumsum().plot(
            figsize = (12,6),
            ylabel  = "Cumulative Basis Points",
            title   = "Equal Risk Contribution Treasury Portfolio of Each PC From {} to {}".format(
                df_erc.index.min().date(),
                df_erc.index.max().date())))
        
        plt.tight_layout()
        
    def plot_erc_corr(self, df_erc: pd.DataFrame) -> plt.Figure: 
        
        fig, axes = plt.subplots(figsize = (8,6))
    
        sns.heatmap(
            ax    = axes,
            data  = df_erc.corr(),
            annot = True)
    
        axes.set_title("Correlation Across ERC Treasury Strategies")
        plt.tight_layout()
        
    def plot_erc_sharpe(self, df_erc: pd.DataFrame) -> plt.Figure: 
                
        df_sharpe = df_erc.mean() / df_erc.std() * np.sqrt(252)
        (df_sharpe.to_frame(
            name = "sharpe").
            plot(
                figsize = (12,6),
                kind    = "bar",
                ylabel  = "Annualized Sharpe",
                legend  = False,
                title   = "Annualized Sharpe of Each ERC Portfolio"))
            

