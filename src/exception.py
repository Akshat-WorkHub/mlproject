import sys
from src.logger import logging

def error_msg_details(error,error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()

    error_file_name = exc_tb.tb_frame.f_code.co_filename
    error_line = exc_tb.tb_lineno


    error_msg = f"""Error Occured !!
Error File : {error_file_name},
Error Line No. : {error_line},
Error Message : {str(error)}"""
    
    return error_msg


class CustomException(Exception):
    def __init__(self,error_msg,error_detail: sys):
        super().__init__(str(error_msg))

        self.error_msg = error_msg_details(error_msg,error_detail)

    def __str__(self):
        return self.error_msg
    

if __name__ == "__main__":
    try:
        a = 10/0
    except Exception as e:
        logging.info("Division by zero")
        try:
            raise CustomException(e, sys)
        except CustomException as ce:
            print(ce)

