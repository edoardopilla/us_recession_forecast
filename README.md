# us_recession_forecast

**_SVM and NN used for recession prediction at 12-month ahead horizon using FRED data, and replication of Engstrom, Sharpe (2018) paper probit model._**

**_Own paper and powerpoint presentation available on request at edo.pilla@hotmail.it_**

The scripts train and use a support vector machine and a neural network to predict forecast probabilities in the US in a binary classification framework at a 12-month ahead horizon, using term spread data and macroeconomic data from FRED.

For execution, plug all the files into a single folder and run the scripts in order, noting that all of them, except "paper_repl_7744044.py" rely on "prep_df_7744044.py" as a preliminary processing step.

The main drawbacks are an arbitrary choice of the samples for training and testing, leading to a slightly better test performance compared to the validation one, the scaling of macroeconomic features that doesn't follow any best practice from the related literature, and large values for C and gamma resulting from grid search for the support vector classifier that can potentially lead to overfitting.

The file "paper_repl_7744044.py" replicates the probit model used by Engstrom and Sharpe (2018) to predict the probabilities of transitioning into a recession using quarterly US data for the near-term forward spread.
