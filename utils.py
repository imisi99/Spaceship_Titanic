import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier


def modify_data(file_path):
    df = pd.read_csv(file_path)
    print('Data before preprocessing...')

    print(df.head())

    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'VIP', 'Age'])

    # filling nan values
    # df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].mean())
    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].mean())
    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean())
    df['Spa'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean())
    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].mean())

    df['CryoSleep'] = df['CryoSleep'].fillna(0)
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')

    # Converting categorical data to one hot encoding
    df = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination'], drop_first=True)

    # Converting boolean data to numerical data
    bool_columns = df.select_dtypes(include='bool').columns
    df[bool_columns] = df[bool_columns].astype(int)

    print('Data after preprocessing...')
    print(df.head())

    return df


def build_nn():
    tf.random.set_seed(42)
    model1 = Sequential([
        Dense(120, activation='relu'),
        Dense(80, activation='relu'),
        Dense(40, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='linear')
    ],
        name='model1'
    )

    model2 = Sequential(
        [
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(6, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model2'
    )

    model3 = Sequential([
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(12, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='linear')
    ])

    return [model1, model2, model3]


def build_rf():
    rf1 = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=4, max_features='sqrt',
                                 random_state='42')
    rf2 = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=2, max_features='log2',
                                 random_state='42')
    rf3 = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=3, max_features='sqrt',
                                 random_state='42')

    rf1.name = 'rf1'
    rf2.name = 'rf2'
    rf3.name = 'rf3'

    print(rf3.name)

    return [rf1, rf2, rf3]
