import math
from collections import Counter

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

MAX_DEPTH = 5
MIN_SAMPLES = 2
RANDOM_STATE = 42
NUMERIC_BINS = 4


def load_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        st.error(f"Could not read this file as a CSV: {exc}")
        st.stop()
    if df.empty or df.shape[1] < 2:
        st.error("The file needs at least two columns and one row of data.")
        st.stop()
    return df.loc[:, ~df.columns.duplicated()]


def preprocess(df, target):
    data = df.dropna(subset=[target]).dropna(axis=1, how="all").copy()
    X = data.drop(columns=[target]).copy()
    y = data[target].astype(str)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
            if X[col].nunique() > NUMERIC_BINS:
                X[col] = pd.qcut(X[col], q=NUMERIC_BINS, duplicates="drop").astype(str)
            else:
                X[col] = X[col].astype(str)
        else:
            mode = X[col].mode()
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            X[col] = X[col].fillna(fill).astype(str)

    id_like = [c for c in X.columns if X[c].nunique() > 0.5 * len(X)]
    X = X.drop(columns=id_like)
    return X, y


def entropy(labels):
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def split_score(y, column, method):
    n = len(y)
    groups = [g for _, g in y.groupby(column)]
    info_gain = entropy(y) - sum(len(g) / n * entropy(g) for g in groups)
    if method == "ID3":
        return info_gain
    split_info = -sum((len(g) / n) * math.log2(len(g) / n) for g in groups)
    return info_gain / split_info if split_info > 0 else 0.0


def build_tree(X, y, features, method, depth=0):
    majority = y.value_counts().idxmax()
    if len(y.unique()) == 1 or not features or depth >= MAX_DEPTH or len(y) < MIN_SAMPLES:
        return {"leaf": True, "label": majority}

    best_feature, best_score = None, -1.0
    for f in features:
        score = split_score(y, X[f], method)
        if score > best_score:
            best_feature, best_score = f, score

    if best_score <= 1e-10:
        return {"leaf": True, "label": majority}

    node = {"leaf": False, "feature": best_feature, "default": majority, "branches": {}}
    remaining = [f for f in features if f != best_feature]
    for value in X[best_feature].unique():
        mask = X[best_feature] == value
        node["branches"][value] = build_tree(
            X[mask], y[mask], remaining, method, depth + 1
        )
    return node


def predict_one(node, row):
    while not node["leaf"]:
        child = node["branches"].get(row[node["feature"]])
        if child is None:
            return node["default"]
        node = child
    return node["label"]


def tree_to_graphviz(tree):
    dot = graphviz.Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#eef6ff")
    counter = [0]

    def add(node):
        nid = str(counter[0])
        counter[0] += 1
        if node["leaf"]:
            dot.node(nid, f"Class: {node['label']}", shape="ellipse", fillcolor="#d5f5e3")
        else:
            dot.node(nid, str(node["feature"]))
            for value, child in node["branches"].items():
                cid = add(child)
                dot.edge(nid, cid, label=str(value))
        return nid

    add(tree)
    return dot


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Decision Tree Classifier", layout="wide")
st.title("Decision Tree Classification System")
st.caption("ID3 and C4.5 implemented from scratch")

uploaded = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

df = load_csv(uploaded)
st.subheader("Dataset preview")
st.dataframe(df.head())

target = st.sidebar.selectbox("Target attribute", df.columns)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

if df[target].dropna().nunique() < 2:
    st.error("The target column must have at least two classes.")
    st.stop()

X, y = preprocess(df, target)
if y.nunique() < 2:
    st.error("After removing missing targets, fewer than two classes remain.")
    st.stop()
if X.shape[1] == 0:
    st.error("No usable feature columns remain after preprocessing.")
    st.stop()

stratify = y if y.value_counts().min() >= 2 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )
except ValueError as exc:
    st.error(f"Could not split the data: {exc}")
    st.stop()

st.write(f"Training rows: **{len(X_train)}** | Testing rows: **{len(X_test)}**")

features = list(X.columns)
labels = sorted(y.unique())
tabs = st.tabs(["ID3", "C4.5", "Comparison"])
all_metrics = []

for tab, method in zip(tabs[:2], ["ID3", "C4.5"]):
    with tab:
        tree = build_tree(X_train, y_train, features, method)
        y_pred = np.array([predict_one(tree, row) for _, row in X_test.iterrows()])
        metrics = evaluate(y_test, y_pred)
        all_metrics.append({"Algorithm": method, **metrics})

        st.subheader(f"{method} Decision Tree")
        st.graphviz_chart(tree_to_graphviz(tree))

        st.subheader("Evaluation")
        cols = st.columns(4)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_confusion(y_test, y_pred, labels, f"{method} Confusion Matrix"))
        with right:
            st.write("Predictions (test set)")
            result = X_test.copy()
            result["Actual"] = y_test.values
            result["Predicted"] = y_pred
            st.dataframe(result)

with tabs[2]:
    comparison = pd.DataFrame(all_metrics).set_index("Algorithm").round(4)
    st.subheader("Performance comparison")
    st.dataframe(comparison)
    st.bar_chart(comparison)
    best = comparison["Accuracy"].idxmax()
    st.success(f"Best accuracy: **{best}** ({comparison.loc[best, 'Accuracy']:.4f})")
