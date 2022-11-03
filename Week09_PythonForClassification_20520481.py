import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

st.title("LOGISTIC REGRESSION WITH STREAMLIT")
"""
# Dataset
"""
file_upload = st.file_uploader("Upload", label_visibility="collapsed")
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
if (file_upload):
    dataset = pd.read_csv(file_upload, sep=";")
    all_features = dataset.columns.to_numpy()
    input_features = []
    st.header("Select which feature for training: ")
    for i in range(len(all_features) - 1):
        checkbox = st.checkbox(all_features[i])
        if checkbox:
            input_features.append(all_features[i])
    output = all_features[-1]
    X = dataset[input_features]
    y = dataset[output]

    output = all_features[-1]
    # st.write(X)
    # st.write(y)
else:
    default()


def train_test():
    run = st.button("Click here to run!!")

    if (is_scaler_applied == "Yes"):
        scaler = StandardScaler()
        global X
        X = scaler.fit_transform(X)
    if run:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=train_size,
                                                            random_state=0)
        model = LogisticRegression(random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        visualization = y_test.to_frame()
        visualization.insert(1, "Prediction", y_pred, True)
        st.write(visualization)

        f1 = f1_score(y_test, y_pred)
        log = log_loss(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        col = st.columns(2)
        with col[0]:
            st.write(f'Precision: {precision:.2f}')
            st.write(f'Recall: {recall:.2f}')
        with col[1]:
            st.write(f'F1 Score: {f1:.2f}')
            st.write(f'Log loss: {log:.2f}')
        fig, _ = plt.subplots(figsize=(8, 4))
        plt.bar(["Precision", "Recall", "F1 Score"],
                [precision, recall, f1], color='maroon', width=0.3)
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        st.pyplot(fig)


def CrossValidation(X, y, cv):
    run = st.button("Click here to run!!")
    if (is_scaler_applied == "Yes"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    if run:
        if int(num_K) > 1:
            model = LogisticRegression()
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                n_jobs=-1)
            kfold_result = pd.DataFrame()
            kfold_result.insert(0, "Score", scores)
            kfold_result['Fold'] = kfold_result.index
            kfold_result = swap_columns(kfold_result, "Score", "Fold")
            st.write(kfold_result)
            st.write('Cross-validation result: %.3f (%.3f)' %
                     (mean(scores), std(scores)))

            fig, _ = plt.subplots(figsize=(8, 4))
            plt.bar(kfold_result["Fold"],
                    kfold_result["Score"], color='maroon', width=0.3)
            plt.xlabel("Fold")
            plt.ylabel("Result")
            st.pyplot(fig)


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


"""
# Output: 
"""
st.header("So we are going to predict: " + output)
"""
# Standard Scaler:
"""
st.header("Do you want to apply Standard Scaler?")
is_scaler_applied = st.radio("scaler", ("Yes", "No"),
                             label_visibility="collapsed", index=1)

"""
# Which method you want to use:
"""
method = st.selectbox(
    "Please select: ", ("Train/Test Split", "KFold Cross Validation"))

if method == 'Train/Test Split':
    """
    # Train/Test split:
    """
    train_size = float(st.slider('Train size: ', 0.1, 0.9, 0.7, 0.1))
    train_test()
else:
    """
    # KFold:
    """
    num_K = st.slider('Number of K folds: ', 2, 12, 7)
    cross_validation = KFold(n_splits=int(num_K), shuffle=True)
    CrossValidation(X, y, cross_validation)
