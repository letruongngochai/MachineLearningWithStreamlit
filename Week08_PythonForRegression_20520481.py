import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title("LOGISTIC REGRESSION WITH STREAMLIT")
"""
# Dataset
"""
file_upload = st.file_uploader("")
"""
# Input feature:
"""


def default():
    list = ["A", "B", "C", "D"]
    col = st.columns(4)
    with col[0]:
        st.checkbox('A', value=False)
    with col[1]:
        st.checkbox('B', value=False)
    with col[2]:
        st.checkbox('C', value=False)
    with col[3]:
        st.checkbox('D', value=False)


output = "E"
dataset = pd.DataFrame()
train_feature = []
non_use_feature = []
if (file_upload):
    dataset = pd.read_csv(file_upload)
    num_col = len(dataset.columns)

    train_feature = st.multiselect(
        "Select which feature for training: ", dataset.columns[:-1])
    temp = dataset.columns[:-1].to_list()
    for i in temp[:]:
        if i in train_feature:
            temp.remove(i)
    non_use_feature = temp
    if len(non_use_feature) > 0:
        st.write(
            "So these features won't be used for training process: ", non_use_feature[:])
    X = dataset.iloc[:, :-1].drop(columns=non_use_feature)
    y = dataset.iloc[:, -1].drop(columns=non_use_feature)

    output = dataset.columns[-1]
else:
    default()


def preprocessing(data):
    for (columnName, columnData) in data.iteritems():
        if (columnData.dtype == "object"):
            data = pd.get_dummies(data, columns=[columnName])
    return data


def train_test():
    """
    # EXECUTE
    """

    run = st.button("Click here to run!!")
    if run:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=float(
                                                                train_size),
                                                            random_state=0)
        X_train_pre = preprocessing(X_train)
        X_test_pre = preprocessing(X_test)

        missing_cols = set(X_train_pre.columns) - set(X_test_pre.columns)
        for c in missing_cols:
            X_test_pre[c] = 0
        X_test_pre = X_test_pre[X_train_pre.columns]

        model = LinearRegression()
        model.fit(X_train_pre, y_train)
        y_pred = model.predict(X_test_pre)

        mse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)

        visualization = y_test.to_frame()
        visualization.insert(1, "Prediction", y_pred, True)
        st.write(visualization)

        col = st.columns(2)
        with col[0]:
            st.write(f'Mean Squared Error: {mse:.2f}')
        with col[1]:
            st.write(f'Mean Absolute Error: {mae:.2f}')

        fig, _ = plt.subplots(figsize=(3, 3))
        plt.bar(["Mean Squared Error", "Mean Absolute Error"],
                [mse, mae], color='maroon', width=0.3)
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        st.pyplot(fig)


def CrossValidation(X, y, cv):
    """
    # EXECUTE
    """

    run = st.button("Click here to run!!")
    if run:
        if int(num_K) > 1:
            model = LinearRegression()
            X_pre = preprocessing(X)
            scores = cross_val_score(
                model, X_pre, y,
                cv=cv,
                n_jobs=-1)
            kfold_result = pd.DataFrame()
            kfold_result.insert(0, "Score", scores)
            kfold_result['Fold'] = kfold_result.index
            kfold_result = swap_columns(kfold_result, "Score", "Fold")
            st.write(kfold_result)
            st.write('Cross-validation result: %.3f (%.3f)' %
                     (mean(scores), std(scores)))


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


"""
# Output: 
"""
st.write(output)
"""
# Which method you want to use:
"""
method = st.radio(
    "What's your choice",
    ('train_test_split', 'KFold'))

if method == 'train_test_split':
    """
    # Train/Test split:
    """
    train_size = st.text_input(label='Train size: ', value=0.7)
    train_test()
else:
    """
    # KFold:
    """
    num_K = st.text_input(label='Number of K folds (>= 2): ', value=2)
    cross_validation = KFold(n_splits=int(num_K), shuffle=True)
    CrossValidation(X, y, cross_validation)
