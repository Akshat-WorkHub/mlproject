from setuptools import find_packages, setup
from typing import List

HYPHEN="-e ."
def get_req(filepath: str) -> List[str]:
    '''
        This Function will take requirement.txt as filepath and returns a List of str content.
    '''
    req = []
    with open(filepath) as file:
        req = file.readlines()
        req = [r.replace("\n","") for r in req]

        if HYPHEN in req:
            req.remove(HYPHEN)
        return req
    
setup(
    name='mlproject',
    version='0.0.1',
    author='Akshat',
    author_email='akshatsoftware9829@gmail.com',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)