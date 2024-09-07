import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import json

models_names = ["score_3m", "score_6m", "score_9m", "score_12m"]

from catboost import CatBoostRegressor, Pool

PATH_TO_TEST_PREDICTIONS = 'artifacts/test_predictions.csv'

COLS_TO_DROP = [
    "target",
    "№ п/п",
    'ДатаНачалаЗадачи',
    'ДатаОкончанияЗадачи',
    'ДатаначалаБП0',
    'ДатаокончанияБП0',
    'date_report'
]
OBJECT_KEY = 'Кодзадачи'
TARGET = 'target'
IMPORTANT_TASK_CODES = ()


@st.cache_resource
def load_model():
    model_3m = pd.read_pickle("model_3m.pkl")
    model_6m = pd.read_pickle("model_6m.pkl")
    model_9m = pd.read_pickle("model_9m.pkl")
    model_12m = pd.read_pickle("model_12m.pkl")
    return model_3m, model_6m, model_9m, model_12m


@st.cache_resource
def load_features():
    with open("config.json") as f:
        config = json.load(f)
    return config


@st.cache_resource
def preprocess(uploaded_file, config):
    # Preprocessing code here
    features = pd.read_csv(uploaded_file)
    print(features[config['model_3m_feat']].shape)
    pool_3m = Pool(features[config['model_3m_feat']], cat_features=config['model_3m_cat_feat'])
    pool_6m = Pool(features[config['model_6m_feat']], cat_features=config['model_6m_cat_feat'])
    pool_9m = Pool(features[config['model_9m_feat']], cat_features=config['model_9m_cat_feat'])
    pool_12m = Pool(features[config['model_12m_feat']], cat_features=config['model_12m_cat_feat'])
    return pool_3m, pool_6m, pool_9m, pool_12m


# @st.cache_resource
def get_preds(models, pools, model_names):
    scored_df = pd.DataFrame()
    for model, pool, name in zip(models, pools, model_names):
        pred = model.predict_proba(pool)[:, 1]

        scored_df[name] = pred

    return scored_df


# @st.cache_data
# def get_shap_values(_model, features):
#    explainer = shap.Explainer(_model)
#    shap_values = explainer(features)
#    return explainer, shap_values

@st.cache_data
def get_shap_values():
    shap_values = pd.read_pickle('artifacts/shap_values.pkl')
    return shap_values


def main():
    uploaded_file = st.file_uploader("Выберите файл в формате .csv")
    if uploaded_file is not None:
        models = load_model()
        config = load_features()
        pools = preprocess(uploaded_file, config)
        print(models[0])
        df = get_preds(models, pools, models_names)
        st.dataframe(df)


if __name__ == '__main__':
    main()
