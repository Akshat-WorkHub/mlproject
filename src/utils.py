import os,sys

from src.logger import logging
from src.exception import CustomException

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
