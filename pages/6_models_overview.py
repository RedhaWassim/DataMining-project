import streamlit as st
import pandas as pd
import plotly.express as px
from soil_fertility.utils import retreive_base_path, load_object
import json

st.title("📊 Models Overview")

base_path = retreive_base_path()

# Select Classifier
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Decision Tree", "Random Forest", "Apriori")
)
classifier_type = st.sidebar.radio("Classifier Type", ("Normal", "Grid Search"))


def load_metrics(classifier_name, classifier_type):
    classifier_type_formatted = (
        "_gridsearch" if classifier_type == "Grid Search" else ""
    )
    model_name = f"metrics{classifier_type_formatted}.pkl"
    metrics = load_object(str(base_path) + f"/artifacts/models/{model_name}")
    return metrics


# Load Metrics
metrics = load_metrics(classifier_name, classifier_type)
from pprint import pprint


# Plotting Function
def plot_metrics(metric, classifier_name, metric_type):
    metrics = json.loads(metric)

    print(metrics)
    # Access the correct level in the metrics dictionary
    if classifier_name == "Decision Tree":
        classifier_name = "decision_tree"

    if classifier_name == "Random Forest":
        classifier_name = "random_forest"
    classifier_metrics = metrics["metrics"].get(classifier_name, {})
    print(classifier_metrics)

    # Check if 'metrics_per_class' exists for the selected classifier
    if "metrics_per_class" in classifier_metrics:
        metrics_per_class = classifier_metrics["metrics_per_class"]

        # Prepare data for plotting
        data = []
        for class_label, class_metrics in metrics_per_class.items():
            data.append(
                {
                    "Class": class_label,
                    "Precision": class_metrics["precision"],
                    "Recall": class_metrics["recall"],
                    "F1-Score": class_metrics["f1_score"],
                    "Specificity": class_metrics["specificity"],
                }
            )

        df = pd.DataFrame(data)

        # Create Plot
        fig = px.bar(df, x="Class", y=[metric_type], title=f"{metric_type} per Class")
        st.plotly_chart(fig)
    else:
        st.write(f"No 'metrics_per_class' data available for {classifier_name}")


# Usage
plot_metrics(metrics, classifier_name, "Precision")
plot_metrics(metrics, classifier_name, "Recall")
plot_metrics(metrics, classifier_name, "F1-Score")
