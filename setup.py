from setuptools import setup,find_packages
from typing import List
def get_requirments(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("\n","") for req in requirments]
    return requirments
setup(
    name='Diamond_price_prediction',
    version='0.0.1',
    auhtor='aravind',
    author_email='aravindkavati83@gmail.com',
    install_requires=['pandas','numpy'],
    packages=find_packages()

)