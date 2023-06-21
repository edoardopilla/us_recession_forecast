### NEURAL NETWORK REPLICATION
# import necessary libraries
import matplotlib
matplotlib.use("Qt5Agg", force = True)
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# prepare merged dataframe
exec(open('prep_df_7744044.py').read())

# create regressors dictionary for model implementation, where the second element in the value list is the optimal threshold
# chosen on the validation set
regr_dct = {"vanilla_net_repl": [merged_df.drop(columns=["date", "index", "nonfarm", "rec"]), .5],
            "macro_net_repl": [merged_df.drop(columns=["date", "spread_y", "rec"]), .55]}

for k, v in regr_dct.items():
    # split dataset into validation set for hyperparameter tuning
    X_val = v[0][275:375].values
    y_val = merged_df["rec"][275:375].values

    # load model from given path
    model = tf.keras.models.load_model(k)

    # predict on validation data
    y_fit = model(X_val, training=False).numpy()

    # compare forecast with actual series
    compare = pd.DataFrame()
    compare["date"] = merged_df["date"][275:375].reset_index(drop=True)
    compare["val"] = y_val
    compare["fit"] = np.where(np.squeeze(y_fit) > v[1], 1, 0)

    # compute evaluation metrics
    acc_val = accuracy_score(y_true=y_val, y_pred=compare["fit"].values, normalize=True)
    prec_val = precision_score(y_true=y_val, y_pred=compare["fit"].values)
    rec_val = recall_score(y_true=y_val, y_pred=compare["fit"].values)
    fb_val = fbeta_score(y_true=y_val, y_pred=compare["fit"].values, beta=.5)

    # print evaluation metrics for validation set
    print(f"Validation metrics:\t{acc_val, prec_val, rec_val, fb_val}")

    # create training set for evaluation metrics for overfitting check
    X_train = v[0][:275].values
    y_train = merged_df["rec"][:275].values

    y_fit = model(X_train, training=False).numpy()

    # compare forecast with actual series
    compare = pd.DataFrame()
    compare["date"] = merged_df["date"][:275].reset_index(drop=True)
    compare["train"] = y_train
    compare["fit"] = np.where(np.squeeze(y_fit) > v[1], 1, 0)

    # compute evaluation metrics
    acc_tr = accuracy_score(y_true=y_train, y_pred=compare["fit"].values, normalize=True)
    prec_tr = precision_score(y_true=y_train, y_pred=compare["fit"].values)
    rec_tr = recall_score(y_true=y_train, y_pred=compare["fit"].values)
    fb_tr = fbeta_score(y_true=y_train, y_pred=compare["fit"].values, beta=.5)

    print(f"Training metrics:\t{acc_tr, prec_tr, rec_tr, fb_tr}")

    # predict on test data
    X_test = v[0][375:483].values
    y_test = merged_df["rec"][375:483].values

    y_fit = model(X_test, training=False).numpy()

    # compare forecast with actual series
    compare = pd.DataFrame()
    compare["date"] = merged_df["date"][375:483].reset_index(drop=True)
    compare["test"] = y_test
    compare["fit"] = np.where(np.squeeze(y_fit) > v[1], 1, 0)

    # compute evaluation metrics
    acc_oos = accuracy_score(y_true=y_test, y_pred=compare["fit"].values, normalize=True)
    prec_oos = precision_score(y_true=y_test, y_pred=compare["fit"].values)
    rec_oos = recall_score(y_true=y_test, y_pred=compare["fit"].values)
    fb_oos = fbeta_score(y_true=y_test, y_pred=compare["fit"].values, beta=.5)

    # build comparison graph
    fig, ax = plt.subplots(figsize=(10, 6))
    compare.plot(x="date", y=["test", "fit"], kind="line", ax=ax).legend(loc="right")

    # add text box to show hyperparameters
    props = dict(boxstyle="round", facecolor="white", alpha=.25)
    ax.text(x=compare["date"][30], y=.8, s=f"Hyperparameters\n" +
                                           f"Alpha:\t {0.002}\n".expandtabs(15) +
                                           f"Epochs:\t {10000}\n".expandtabs(7) +
                                           f"Threshold: \t {v[1]}\n".expandtabs(4),
            verticalalignment='top', bbox=props)

    # add text box to show evaluation metrics
    ax.text(x=compare["date"][30], y=.4, s=f"Metrics\n" +
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
    future_X = v[0][483:].values

    # generate forecast
    y_future = model(future_X, training=False).numpy()

    # generate forecast dataframe
    future_df = pd.DataFrame()
    future_df["date"] = pd.date_range(start='2022-04-01', freq="MS", periods=12).date
    future_df["rec"] = np.where(np.squeeze(y_future) > v[1], 1, 0)

    # plot graph for out of sample prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    future_df.plot(x="date", y="rec", kind="line", ax=ax)

    # add title
    plt.title("12-month ahead recession probability forecast from 2022 to 2023")

    # add y axis title
    plt.ylabel("Probability threshold\n(Low / High)")

    # add x axis title
    plt.xlabel("Time")