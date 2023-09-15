from datetime import datetime
import os
import json
import pandas as pd
import dill

path = os.environ.get('PROJECT_PATH', '..')


def predict():
    all_models = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{all_models[-1]}', 'rb') as file:
        model = dill.load(file)

    df = pd.DataFrame(columns=['car_id', 'pred'])

    for elem in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{elem}') as file:
            json_data = json.load(file)
        data = pd.DataFrame.from_dict([json_data])
        predict = model.predict(data)
        predict_dict = {'car_id': data.id, 'pred': predict}
        predict_df = pd.DataFrame(predict_dict)
        df = pd.concat([predict_df, df])

    df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
