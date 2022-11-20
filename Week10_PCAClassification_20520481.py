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
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

st.title("PCA + LOGISTIC REGRESSION WITH STREAMLIT")
"""
# Dataset:
"""
dataset = load_wine()
file_upload = pd.DataFrame(dataset.data)
df = file_upload
df.columns = dataset.feature_names
df["target"] = dataset.target
st.dataframe(df)

"""
# PCA application:
"""
st.header("Do you want to apply PCA?")
is_PCA_applied = st.radio("PCA", ("Yes", "No"),
                          label_visibility="collapsed", index=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


def apply_PCA(X, n):
    pca = PCA(n_components=n)
    return pca.fit_transform(X)


if (is_PCA_applied == "Yes"):
    n_components = st.slider("Your data's dimension will be decreased down to: ", 1, len(
        df.columns) - 1, step=1, value=int((len(df.columns)-2)/2))
    X = apply_PCA(X, n_components)
    st.header("Dataset after applying PCA:")
    st.dataframe(X)


def train_test():
    run = st.button("Click here to run!!")
    if run:

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=train_size,
                                                            random_state=0)
        model = LogisticRegression(random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)

        visualization = y_test.to_frame()
        visualization.insert(1, "Prediction", y_pred, True)
        st.write(visualization)

        f1 = f1_score(y_test, y_pred, average='macro')
        log = log_loss(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

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


st.header("Which method you want to use:")

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
