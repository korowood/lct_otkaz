import pandas as pd
import json
import warnings
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

uploaded_file = "sample.csv"
models_names = ["score_3m", "score_6m", "score_9m", "score_12m"]


def load_model():
    model_3m = pd.read_pickle("model_3m.pkl")
    model_6m = pd.read_pickle("model_6m.pkl")
    model_9m = pd.read_pickle("model_9m.pkl")
    model_12m = pd.read_pickle("model_12m.pkl")
    return model_3m, model_6m, model_9m, model_12m


def load_features():
    with open("config.json") as f:
        config = json.load(f)
    return config


def preprocess(uploaded_file, config):
    # Preprocessing code here
    features = pd.read_csv(uploaded_file)
    print(features[config['model_3m_feat']].shape)
    pool_3m = Pool(features[config['model_3m_feat']], cat_features=config['model_3m_cat_feat'])
    pool_6m = Pool(features[config['model_6m_feat']], cat_features=config['model_6m_cat_feat'])
    pool_9m = Pool(features[config['model_9m_feat']], cat_features=config['model_9m_cat_feat'])
    pool_12m = Pool(features[config['model_12m_feat']], cat_features=config['model_12m_cat_feat'])
    return pool_3m, pool_6m, pool_9m, pool_12m


def get_preds(models, pools, model_names):
    scored_df = pd.DataFrame()
    for model, pool, name in zip(models, pools, model_names):
        pred = model.predict_proba(pool)[:, 1]

        scored_df[name] = pred

    return scored_df


if __name__ == '__main__':
    models = load_model()
    config = load_features()
    pools = preprocess(uploaded_file, config)

    df = get_preds(models, pools, models_names)
    df.to_csv("scored_sample.csv", index=False)
