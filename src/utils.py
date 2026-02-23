import os,sys
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import dill # used for pickling an object


def save_object(filepath: str, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info(f"Proprocessor Objected Saved at {filepath}")

    except Exception as e:
        try:
            raise CustomException(e,sys)
        except CustomException as ce:
            print(ce)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        logging.info("Generating Report by evaluate_model method for each model based on its Training and Testing")
        logging.info("HyperParameters uses GridSearchCV (Cross Validation) Technique for finding best parameters")

        report = dict()
        trained_model = dict()
        
        for key in models:
            model = models[key]
            param = params.get(key,{})

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[key] = {
                "Training Score" : train_score,
                "Testing Score" : test_score,
            }
            trained_model[key] = best_model

        logging.info("Generating Model Report")

        return report,trained_model
    except Exception as e:
        try:
            raise CustomException(e,sys)
        except CustomException as ce:
            print(ce)




