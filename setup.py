from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Weighted ERM'
LONG_DESCRIPTION = 'Weighted ERM'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="WERM", 
        version=VERSION,
        author="Gabriel Arpino",
        author_email="ga442@cam.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)