from setuptools import find_packages, setup 
from typing import List 

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements
    and removes '-e .' if present
    """
    requirements = []
    with open(file_path) as file:
        for line in file:
            req = line.strip()
            if req and req != HYPEN_E_DOT:
                requirements.append(req)
    return requirements
        

setup(
    name="Employee_Burnout_Score_Predition",
    version="0.0.1",
    author="Ashira Maharjan" "Dipak",
    email="ashiramaharjan13@gmail.com",
    packages=find_packages(),
    #install_requires=['pandas','numpy','seaborn','matplotlib','sklearn']
    install_requires=get_requirements('requirements.txt')

)