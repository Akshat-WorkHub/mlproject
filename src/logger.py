import logging,os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('Date_%m_%d_%Y_Time_%H_%M_%S')}.log"

logpath = os.path.join(os.getcwd(),"logs",LOG_FILE[:-4])
os.makedirs(logpath,exist_ok=True)

LOG_FILE_PATH = os.path.join(logpath,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s %(levelname)s %(message)s"
)

if __name__ == "__main__":
    logging.info("Logging has started !!")