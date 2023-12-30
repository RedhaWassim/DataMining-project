import streamlit as st
import pandas as pd
import plotly.express as px
from soil_fertility.utils import load_object, retreive_base_path
from soil_fertility.pipeline.predict_pipeline import InputData, PredictPipeline
from soil_fertility.logger import logging
from soil_fertility.components.data_pipeline.data_ingestion import DataIngestion
from soil_fertility.components.data_pipeline.data_transformation import (
    DataTransformationThree,
)
from soil_fertility.components.model_pipeline.models.apriori import MyApriori
from typing import List
import os


st.title(" :bar_chart: Model GUI")


# ... (rest
base_path = retreive_base_path()

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Decision Tree", "Random Forest", "apriori")
)


if classifier_name == "apriori":
    strategie = st.sidebar.selectbox("Select strategies", ("width", "frequency"))
    k = st.sidebar.number_input("Enter a value for K", value=5, step=1)
    apriory_min_support = st.sidebar.slider("Select min support", 0.0, 1.0, 0.5)
    apriory_min_confidence = st.sidebar.slider("Select min confidence", 0.0, 1.0, 0.5)

    data = st.file_uploader("upload file", type={"xlsx"})
    if data is not None:
        df = pd.read_excel(data)

    def execute_apriori(
        df: pd.DataFrame,
        strategie: str,
        apriory_min_support: float,
        apriory_min_confidence: float,
        transactions_group: List[str],
        items_groups: List[str],
    ):
        numerical_features = [
            element
            for element in transactions_group + items_groups
            if element not in ["Crop", "Soil", "Fertilizer"]
        ]

        obj = DataIngestion()
        train, test = obj.init_ingestion(
            data=df, option="xlsx", save=False, return_df=True
        )

        tranformation_obj = DataTransformationThree()
        data, preprocessor = tranformation_obj.transform(
            train=train,
            test=test,
            numerical_features=numerical_features,
            k=5,
            strategie=strategie,
            save=False,
            return_df=True,
        )

        aprio = MyApriori(apriory_min_support, apriory_min_confidence)

        aprio.fit(
            data, transaction_columns=transactions_group, items_groups=items_groups
        )
        rules = aprio.get_rules()

        return rules

    list_elem = ["Crop", "Soil", "Fertilizer", "Temperature", "Humidity", "Rainfall"]

    transactions_group = st.multiselect("Select transaction features ", list_elem)
    items = [x for x in list_elem if x not in transactions_group]
    items_groups = st.multiselect("Select items groups ", items)

    if st.button("Execute"):
        results = execute_apriori(
            df,
            strategie,
            apriory_min_support,
            apriory_min_confidence,
            transactions_group,
            items_groups,
        )
        st.write(results)

else:
    classifier_type = st.sidebar.radio("Select Classifier", ("normal", "grid search"))

    classifier_name = classifier_name.lower().replace(" ", "_")
    if classifier_type == "grid search":
        classifier_type = "_" + classifier_type.lower().replace(" ", "")
    else:
        classifier_type = ""
    model_name = f"{classifier_name}{classifier_type}.pkl"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        N = st.number_input("Enter a value for N")
        P = st.number_input("Enter a value for P")
        K = st.number_input("Enter a value for K")

    with col2:
        pH = st.number_input("Enter a value for pH")
        EC = st.number_input("Enter a value for EC")
        OC = st.number_input("Enter a value for OC")

    with col3:
        S = st.number_input("Enter a value for S")
        Zn = st.number_input("Enter a value for Zn")
        Fe = st.number_input("Enter a value for Fe")
    with col4:
        Cu = st.number_input("Enter a value for Cu")
        Mn = st.number_input("Enter a value for Mn")
        B = st.number_input("Enter a value for B")

    OM = st.number_input("Enter a value for OM")

    reference_data = pd.read_csv(
        os.path.join(base_path, "artifacts/intermediate/1/", "train.csv")
    )
    selected_feature = st.sidebar.selectbox(
        "Select Feature for Analysis",
        reference_data.columns,
    )

    def predict_one(model_name):
        data = InputData(
            N=N,
            P=P,
            K=K,
            pH=pH,
            EC=EC,
            OC=OC,
            S=S,
            Zn=Zn,
            Fe=Fe,
            Cu=Cu,
            Mn=Mn,
            B=B,
            OM=OM,
        )
        df = data.get_data_as_df()
        pipeline = PredictPipeline()
        prediction, drift_value, drift_results = pipeline.predict(
            df, model_name, drift=True
        )

        st.write(f"Prediction : {prediction}")

        if st.button("drift"):
            if drift_value:
                st.write("Drift detected, model retraining required")
                st.write("drift_result is :", drift_results)

            else:
                st.write("No drift detected")

        st.subheader("train data")
        fig_hist = px.histogram(
            reference_data, x=selected_feature, marginal="box", nbins=30
        )
        fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("test data")
        fig_hist = px.histogram(df, x=selected_feature, marginal="box", nbins=30)
        fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.button("Predict", on_click=predict_one(model_name))
