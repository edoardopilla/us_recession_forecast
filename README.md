The scripts train and use a support vector machine and a neural network to predict forecast probabilities in the US in a binary classification framework at a 12-month ahead horizon, using term spread data and macroeconomic data from FRED.

For execution, plug all the files into a single folder and run the scripts in order, noting that all of them, except "paper_repl_7744044.py" rely on "prep_df_7744044.py" as a preliminary processing step.

The file "paper_repl_7744044.py" replicates the probit model used by Engstrom and Sharpe (2018) to predict the probabilities of transitioning into a recession using quarterly US data for the near-term forward spread.
