"""
Implements a model training pipeline
from preprocessing to model selection 
to model evaluation.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

class ModelingPipeline:

    def __init__(self):
        pass


    def get_splitted_data(self, target, test_size, input_object):
        """
        Returns a dictionary containing the splitted data in train, test
        and a list of numerical and categorical columns.

        Arguments:
            target: (string) the dependent variable we want to predict the vlaues
            input_object (string | pandas.DataFrame) path to the input data of pandas data frame
            test_size (float) the percentage of test data relative to training data

        Return:
            a dictionary of the form 
            data_dict = {
                    'numerical_columns': num_cols,'categorical_columns': cat_cols,  
                    'X_train': X_train, 'y_train': y_train,
                    'X_test': X_test, 'y_test': y_test
                }
        """
        df = input_object
        if isinstance(input_object, str):
            df = pd.read_csv(input_object)
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        non_numeric_types = ['object']
        num_cols = list(df.select_dtypes(include=numeric_types).columns)
        cat_cols = list(df.select_dtypes(include=non_numeric_types).columns)
        
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=0)
        y_train = df_train[target]
        X_train = df_train.drop(target, axis=1, inplace=False)
        y_test = df_test[target]
        X_test = df_test.drop(target, axis=1, inplace=False)
        data_dict = {
                'numerical_columns': num_cols,'categorical_columns': cat_cols,  
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }
        return data_dict


    def fit_model(self, model, preprocessor, data, target):
        """
        Fit the model.
        Arguments:
            model: the model to be fitted
            preprocessor: a pipeline using sklearn 
                         ColumnTransformer class
            data: a dictionary containing the data.
                  this is implemeted in the function 'get_splitted_data'
        Return:
            the fitted model
        """
        data['numerical_columns'].remove(target)
        pipeline = Pipeline([
            ('data_preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(data['X_train'], data['y_train'])
        return pipeline


    def test_fitted_model(self, fitted_model, data, metric):
        """
        Test the fitted model
        """
        y_preds = fitted_model.predict(data['X_test'])
        score = metric(y_preds, data['y_test'])
        return score


    def run_experiment(self, data, models, preprocessor, metric, target):
        """run an experiment to evatuate different models accuracy"""
        scores = {}
        for name, model in models.items():
            fitted_model = self.fit_model(model, preprocessor, data, target)
            score = self.test_fitted_model(fitted_model, data, metric)
            scores[name] = score
        return scores


    def cross_validation(self):
        """Perform cross validation for what?"""
        pass


    def save_new_feature(self, df, exp_numb):
        if not os.path.exists:
            path = f"features/{exp_numb}"
            os.makedirs(path)
        df.to_csv(f"{path}/experiment{exp_numb}_data.csv")


    def generate_experiment_report(self):
        pass


    def select_features(self, X, y, columns):
        model = SelectKBest(score_func=f_regression, k='all')
        model.fit(X, y)
        plt.bar([columns[:-1][i] for i in range(len(model.scores_))], model.scores_)
        from pylab import rcParams
        rcParams['figure.figsize'] = 10, 10
        plt.xlabel('Features')
        plt.ylabel('Score')
        plt.title('Feature importance score')
        plt.show()










    