import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
#README = (HERE / "README.md").read_text()
with open("README.md", "r", encoding='utf8') as fh:
  README = fh.read()
  
    
setup(name='miniCHAMP',
      version='0.0.0',
      description='miniCHAMP',
      long_description=README,
      long_description_content_type="text/markdown",
      url='https://github.com/philip928lin/miniCHAMP',
      author='Chung-Yi Lin',
      author_email='philip928lin@gmail.com',
      license='--',
      packages=find_packages(), #['miniCHAMP'],
      install_requires = ["numpy", "pandas", "dotmap", "gurobipy"],
      classifiers=[
        #"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Framework :: IPython",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
      zip_safe=False,
      include_package_data = True,
      python_requires='>=3.10')



