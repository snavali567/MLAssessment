from data_loader import data_preprocessing
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle


def train(model, data_train, target_train, model_name):
    print(f"Training the {model_name}")
    model.fit(data_train, target_train)
    return model


def prediction(row_json, model_file):
    # row_data comes in the JSON format

    with open(model_file, 'rb') as handle:
        model_data = pickle.load(handle)

    row_list = []

    for i in model_data['cols']:
        if i in row_json:
            row_list.append(row_json[i])
        else:
            row_list.append(model_data[i])

    assert len(row_list) == len(model_data['cols'])

    temp_df = pd.DataFrame([row_list], columns=model_data['cols'])

    predictions = model_data['model'].predict_proba(temp_df)

    return {'Prediction': predictions[0][1]}


def validate(trained_model, data_valid, target_valid, model_name):
    print(f"Validation Accuracy for {model_name}:", trained_model.score(data_valid, target_valid))


def save_model(trained_model, data_train, model_name):
    model_dict = dict()

    model_dict['cols'] = ['Issue Date', 'Issue time', 'RP State Plate', 'Plate Expiry Date', 'Body Style', 'Color',
                          'Location', 'Route', 'Agency', 'Violation code', 'Violation Description', 'Fine amount']

    dataTypes = dict(data_train.dtypes)

    # store the mean and mode values to accomodate for missing values during test time
    for column_name, column_type in dict(data_train.dtypes).items():
        if column_type == 'object':
            model_dict[column_name] = data_train[column_name].mode()[0]
        else:
            model_dict[column_name] = data_train[column_name].mean()

    model_dict['model'] = trained_model

    pickle.dump(model_dict, open(model_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def trainer(data_url):
    max_iter = 2000

    data_train, data_valid, target_train, target_valid = data_preprocessing(data_url)

    # obtain list of categorical features and numerical features
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(data_train)
    categorical_columns = categorical_columns_selector(data_train)

    # Use for logistic regression
    # categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    # Use for Gradient Boost Algo
    categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                                             unknown_value=-1)

    # preprocessor = ColumnTransformer([
    #     ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    #     ('standard_scaler', numerical_preprocessor, numerical_columns)])

    preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

    # define the model pipeline and validate the model on the validation set

    # Logistic Regression Model
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=max_iter))
    trained_model_lr = train(model, data_train, target_train, "Logistic Regression Model")
    validate(trained_model_lr, data_valid, target_valid, "Logistic Regression Model")

    # Gradient Boosting Model
    model_gb = make_pipeline(preprocessor, AdaBoostClassifier(n_estimators=100))
    trained_model_gb = train(model_gb, data_train, target_train, "Gradient Boosting Model")
    validate(trained_model_gb, data_valid, target_valid, "Gradient Boosting Model")

    # save the model along with mean/mode values
    save_model(trained_model_lr, data_train, "logistic_model.pkl")
    save_model(trained_model_gb, data_train, "gb_model.pkl")


if __name__ == '__main__':
    trainer('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')
