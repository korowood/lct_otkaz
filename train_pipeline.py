import pandas as pd
import gc
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import json
import warnings

warnings.filterwarnings('ignore')

mapper_shift = {-90: 'target_3m',
                -180: 'target_6m',
                -270: 'target_9m',
                -360: 'target_12m'}

mapper_split = {'target_3m': 4000,
                'target_6m': 3000,
                'target_9m': 2000,
                'target_12m': 1300, }

drop_cols = ['failure', 'serial_number', 'model']  #


def load_data(data_path, shift_value=-90):
    data = pd.read_parquet(data_path)
    data.sort_values(by=['serial_number', 'date'], inplace=True)
    print(data.shape)
    target_name = mapper_shift[shift_value]
    data[target_name] = data.groupby('serial_number')['failure'].shift(shift_value)
    print(data[target_name].value_counts())
    return_data = data[~data[target_name].isna()].copy()
    return_data[target_name] = return_data[target_name].astype(int)

    del data
    gc.collect()  # free memory

    return return_data, target_name


def split_data(data, target_name):
    ser_num = data['serial_number'].unique().tolist()  # list of serial numbers
    print(target_name)
    value_split = mapper_split[target_name]
    train_ser_num = ser_num[:value_split]
    val_ser_num = ser_num[value_split:]
    train_data = data[data['serial_number'].isin(train_ser_num)]
    val_data = data[data['serial_number'].isin(val_ser_num)]

    X_train, y_train = train_data.drop(drop_cols+[target_name], axis=1), train_data[target_name]
    X_val, y_val = val_data.drop(drop_cols+[target_name], axis=1), val_data[target_name]

    del data, train_data, val_data
    gc.collect()  # free memory

    return X_train, y_train, X_val, y_val


def model_train(X_train, y_train, X_val, y_val, model_name):
    cat_feat = X_train.select_dtypes(include=['object', 'int64']).columns.tolist()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                         y=y_train)

    train_pool = Pool(X_train, y_train, cat_features=cat_feat)
    val_pool = Pool(X_val, y_val, cat_features=cat_feat)

    model = CatBoostClassifier(iterations=200,
                               depth=4,
                               learning_rate=0.05,
                               loss_function='Logloss',
                               eval_metric='AUC',
                               od_type='Iter',
                               od_wait=20,
                               random_seed=42,
                               class_weights=class_weights,
                               use_best_model=True,
                               verbose=50)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    fe = model.get_feature_importance(prettified=True)
    new_f = fe[fe['Importances'] > 0]['Feature Id'].tolist()

    cat_feat = X_train[new_f].select_dtypes(include=['object', 'int64']).columns.tolist()
    train_pool = Pool(X_train[new_f], y_train, cat_features=cat_feat)
    val_pool = Pool(X_val[new_f], y_val, cat_features=cat_feat)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    with open("config.json") as f:
        config = json.load(f)

    config[model_name + '_feat'] = new_f
    config[model_name + '_cat_feat'] = cat_feat

    with open("config.json", "w") as f:
        json.dump(config, f)

    pd.to_pickle(model, model_name + '.pkl')

    return model, val_pool


def score(model, val_pool, y_val, model_name):
    y_proba = model.predict_proba(val_pool)[:, 1]
    pred = (y_proba > 0.5).astype(int)
    print(classification_report(y_val, pred))
    print(roc_auc_score(y_val, y_proba))
    print(balanced_accuracy_score(y_val, pred))

    with open("config.json") as f:
        config = json.load(f)

    config[model_name + '_roc_auc'] = roc_auc_score(y_val, y_proba)
    config[model_name + '_f1_score'] = f1_score(y_val, pred)

    with open("config.json", "w") as f:
        json.dump(config, f)


if __name__ == '__main__':
    data_path = 'data_train.pq'

    # train 3 month
    data, target_name = load_data(data_path, shift_value=-90)
    X_train, y_train, X_val, y_val = split_data(data, target_name)
    model_3m, val_pool = model_train(X_train, y_train, X_val, y_val, 'model_3m')
    score(model_3m, val_pool, y_val, 'model_3m')

    # train 6 month
    data, target_name = load_data(data_path, shift_value=-180)
    X_train, y_train, X_val, y_val = split_data(data, target_name)
    model_6m, val_pool = model_train(X_train, y_train, X_val, y_val, 'model_6m')
    score(model_6m, val_pool, y_val, 'model_6m')

    # train 9 month
    data, target_name = load_data(data_path, shift_value=-270)
    X_train, y_train, X_val, y_val = split_data(data, target_name)
    model_9m, val_pool = model_train(X_train, y_train, X_val, y_val, 'model_9m')
    score(model_9m, val_pool, y_val, 'model_9m')

    # train 12 month
    data, target_name = load_data(data_path, shift_value=-360)
    X_train, y_train, X_val, y_val = split_data(data, target_name)
    model_12m, val_pool = model_train(X_train, y_train, X_val, y_val, 'model_12m')
    score(model_12m, val_pool, y_val, 'model_12m')
