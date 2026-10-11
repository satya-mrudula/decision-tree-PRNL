# Decision Tree Classification System

An interactive Streamlit app that builds ID3 and C4.5 decision trees from an uploaded CSV dataset, displays the trees, generates predictions, and compares the algorithms using accuracy, precision, recall, F1-score, and confusion matrices.

## Run locally

pip install -r requirements.txt
streamlit run app.py

## Usage

1. Upload a CSV file from the sidebar.
2. Select the target attribute.
3. Adjust the test set size if needed.
4. Review the ID3 and C4.5 trees, metrics, and predictions in the tabs, then check the comparison tab.
