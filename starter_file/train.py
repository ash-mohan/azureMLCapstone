from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd

from azureml.core.run import Run
from azureml.core.dataset import Dataset


def clean_data(data):
    enc = OrdinalEncoder()
    scaler = MinMaxScaler()

    numeric_columns = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    enc = OrdinalEncoder()
    scaler = MinMaxScaler()
    
    x_df = data.to_pandas_dataframe().drop_duplicates()
    x_df = x_df[x_df['year'].between(2000, 2019)]
    y_df = x_df.pop("popularity")

    x_df["explicit"] = x_df.explicit.apply(lambda s: 1 if s == "True" else 0)
    artists = pd.get_dummies(x_df.artist, prefix="artist")
    genres = pd.get_dummies(x_df.genre, prefix="genre")
    year = enc.fit_transform(x_df[["year"]])
    scaled = scaler.fit_transform(x_df[numeric_columns])
    x_df.drop(["song", "artist", "genre", "year"], inplace=True, axis=1)
    x_df.drop(numeric_columns, inplace=True, axis=1)
    x_df = x_df.join(artists)
    x_df = x_df.join(genres)

    x_arr = np.concatenate((x_df.to_numpy(), year, scaled), axis=1)



    return x_arr, y_df.to_numpy()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help = "The maximum depth of the tree.")
    parser.add_argument('--max_features', default=1.0, help = "The number of features to consider when looking for the best split.")

    args = parser.parse_args()
    run = Run.get_context()

    run.log("Number of Estimators:", np.int(args.n_estimators))
    run.log("Max Depth:", np.int(args.max_depth))
    run.log("Max Features: ", str(args.max_features))

    ds = Dataset.Tabular.from_delimited_files('https://raw.githubusercontent.com/ash-mohan/azureMLCapstone/main/starter_file/data/songs_normalize.csv') 

    X, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, max_features=args.max_features).fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5

    run.log("MSE", np.float(mse))
    run.log("RMSE", np.float(rmse))

if __name__ == '__main__':
    main()

    
    

