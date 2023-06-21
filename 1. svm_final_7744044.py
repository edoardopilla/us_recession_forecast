### SUPPORT VECTOR MACHINE
# import necessary libraries
import matplotlib
matplotlib.use("Qt5Agg", force = True)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# silence UndefinedMetricWarning when running first training due to precision scoring
from sklearn.exceptions import UndefinedMetricWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# prepare merged dataframe
exec(open('prep_df_7744044.py').read())

# draw scatterplot of spreads colored according to recession indicator
rec_hue = [0, 1]
sns.relplot(data = merged_df, x = 'spread_x', y = 'spread_y', hue = 'rec', hue_order = rec_hue, aspect = 1.61)

plt.text(-.01, .02, f"Recession:\t {(merged_df['rec'][merged_df['rec'] == 1]).count()}\n".expandtabs(1) +
                    f"Expansion:\t {(merged_df['rec'][merged_df['rec'] == 0]).count()}".expandtabs(1))

# add title
plt.title("Term spread scatterplot by category")

# add y axis title
plt.ylabel("10-year to 2-year spread")

# add x axis title
plt.xlabel("10-year to 3-month spread")

# create regressor list to set up for loop for baseline and extended model implementation
regr_lst = [merged_df.drop(columns = ["date", "index", "nonfarm", "rec"]), merged_df.drop(columns = ["date", "spread_y", "rec"])]

# split dataset into training and validation sets for prediction
for regr_arr in regr_lst:
    X_train = regr_arr[:375].values
    y_train = merged_df["rec"][:375].values

    X_val = regr_arr[275:375].values
    y_val = merged_df["rec"][275:375].values

    # set random seed for replicating results
    np.random.seed(17)

    # set up support vector classifier
    svc = svm.SVC(kernel="rbf", degree=1, C=1, gamma=.0005, class_weight="balanced", random_state=17)

    # set parameter grid for grid search
    grid_params = {"kernel": ["poly", "rbf"], "degree": [2, 3, 4], "C": np.random.randint(low=1, high=500, size=20),
                   "gamma": np.random.uniform(low=.0005, high=500, size=20)}

    # create the custom cross validation parameter for gridsearchCV to preserve the time ordering
    ps = PredefinedSplit(test_fold=np.concatenate(([-1] * 275, [0] * 100), axis=0))

    # run grid search, causes warnings due to no predictions for several combinations, hence no precision score defined
    gs = GridSearchCV(estimator=svc, param_grid=grid_params, cv=ps, scoring="precision")

    # fit classifier to train sets
    gs.fit(X_train, y_train)

    # forecast future recessions
    y_fit = gs.predict(X_val)

    # save best parameters
    best_params_val = gs.best_params_

    # save evaluation metrics for validation set
    acc_val = accuracy_score(y_true=y_val, y_pred=y_fit, normalize=True)
    prec_val = precision_score(y_true=y_val, y_pred=y_fit)
    rec_val = recall_score(y_true=y_val, y_pred=y_fit)
    fb_val = fbeta_score(y_true=y_val, y_pred=y_fit, beta=.5)

    # print evaluation metrics for validation set
    print(f"Validation metrics:\t{acc_val, prec_val, rec_val, fb_val}")

    # print best parameters from fine tuning
    print(f"Hyperparameters:\t{best_params_val}")

    # predict on training data for overfitting check with fine tuned hyperparameters
    svc = svm.SVC(kernel=best_params_val["kernel"], degree=best_params_val["degree"], C=best_params_val["C"], gamma=best_params_val["gamma"], class_weight="balanced", random_state=17)

    # fit to previous dataset
    svc.fit(X_train, y_train)

    # forecast future recessions
    y_fit = svc.predict(X_train)

    # save evaluation metrics for train set
    acc_tr = accuracy_score(y_true=y_train, y_pred=y_fit, normalize=True)
    prec_tr = precision_score(y_true=y_train, y_pred=y_fit)
    rec_tr = recall_score(y_true=y_train, y_pred=y_fit)
    fb_tr = fbeta_score(y_true=y_train, y_pred=y_fit, beta=.5)

    # print evaluation metrics for train set
    print(f"Training metrics:\t{acc_tr, prec_tr, rec_tr, fb_tr}")

    # specify test set for out of sample evaluation
    X_test = regr_arr[375:483].values
    y_test = merged_df["rec"][375:483].values

    # forecast future recessions
    y_fit = svc.predict(X_test)

    # save evaluation metrics for test set
    acc_oos = accuracy_score(y_true=y_test, y_pred=y_fit, normalize=True)
    prec_oos = precision_score(y_true=y_test, y_pred=y_fit)
    rec_oos = recall_score(y_true=y_test, y_pred=y_fit)
    fb_oos = fbeta_score(y_true=y_test, y_pred=y_fit, beta=.5)

    # compare forecast with actual series
    compare = pd.DataFrame()
    compare["date"] = merged_df["date"][375:483]
    compare["test"] = y_test
    compare["fit"] = y_fit

    fig, ax = plt.subplots(figsize=(10, 6))
    compare.plot(x="date", y=["test", "fit"], kind="line", ax=ax).legend(loc="center left")

    # add text box to show hyperparameters
    props = dict(boxstyle="round", facecolor="white", alpha=.25)
    ax.text(x=compare["date"][390], y=.8, s=f"Hyperparameters\n" +
                                            f"C:\t {best_params_val['C']}\n".expandtabs(13) +
                                            f"Gamma:\t {round(best_params_val['gamma'], 2)}\n".expandtabs(1) +
                                            f"Kernel: \t {best_params_val['kernel']}\n".expandtabs(2) +
                                            f"Degree: \t {best_params_val['degree']}".expandtabs(2),
            verticalalignment='top', bbox=props)

    # add text box to show evaluation metrics
    ax.text(x=compare["date"][390], y=.4, s=f"Metrics\n" +
                                            f"Accuracy:\t {round(acc_oos, 2)}\n".expandtabs(13) +
                                            f"Precision:\t {round(prec_oos, 2)}\n".expandtabs(5) +
                                            f"Recall: \t {round(rec_oos, 2)}\n".expandtabs(8) +
                                            f"Fb score: \t {round(fb_oos, 2)}".expandtabs(7),
            verticalalignment='top', bbox=props)

    # add title
    plt.title("Recession probability forecast from 2013 to 2022")

    # add y axis title
    plt.ylabel("Probability threshold\n(Low / High)")

    # add x axis title
    plt.xlabel("Time")

    # create array for forecast next 12 month dummies
    future_X = regr_arr[483:].values

    # generate forecast
    y_future = svc.predict(future_X)

    # generate forecast dataframe
    future_df = pd.DataFrame()
    future_df["date"] = pd.date_range(start='2022-04-01', freq="MS", periods=12).date
    future_df["rec"] = y_future

    # plot graph for out of sample prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    future_df.plot(x="date", y="rec", kind="line", ax=ax)

    # add title
    plt.title("12-month ahead recession probability forecast from 2022 to 2023")

    # add y axis title
    plt.ylabel("Probability threshold\n(Low / High)")

    # add x axis title
    plt.xlabel("Time")