# Treasury Inflation PCA
This repo focuses on a trading idea that invovles trading the difference of two principal components of different term structures applied to Treasury Futures. In this case the term strucures are the breakeven curve and the inflation swap curve. Thus 3 signals are generated and all prove to generate substantial returns for trading Treasury Futures. Most of the research within this repo focuses on the quality and performance of the signal. 

After the signal quality and performance has been fully researched the focus of this repo will look into the fundamental drivers of the returns. At the time of writing this the leading theory for the fundamental driver of returns is market liquidity. Since in this case the signal is defined as the PC of the breakeven rate subtracted by the PC of the inflation swap rate (which is a less liquid) instrumnet the signals can be thought of the dislocations of liquidity.

# Repo Layout
```bash
    TreasuryInflationPCA
      └───DataQuality
      └───notebooks
          └───SignalAnalysis
          └───SignalBacktest
          └───RegressionBacktest
      └───src
```

Repo Explanation:
* ```src```: The source code directory for managing data collection and computations
* ```DataQuality```: Directory looks into the data quality and how data cleaning affects the final results 
* ```noteboks/SignalAnalysis```: Analyzes the quality of the signals using statistical measures
* ```noteboks/SignalBacktests```: Analyzes the signals' performance within the Treasury market
* ```noteboks/RegressionBacktest```: Uses an OLS model to trade the signal rather than conditioning on the raw signal. 

Below is a plot of Treasury futures conditioned on the raw signal

<img width="1328" alt="image" src="https://github.com/user-attachments/assets/8d1a340d-f8a5-4aeb-bbe3-0457bfb4efe3" />

A formal writeup of the strategy (rough draft) in this notebook called ```Treasury_Inflation_PCA_Technical_Writeup.pdf```

# Todo

1. Rolling regression comparison

