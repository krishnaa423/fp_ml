#region modules
from setuptools import setup, find_packages
from Cython.Build import cythonize
#endregions

#region variables
#endregions

#region functions
setup(
    name='fp_ml',
    version='1.0.0',
    description='ML for First priciples calculations',
    author='Krishnaa Vadivel',
    author_email='krishnaa.vadivel@yale.edu',
    requires=[
        'numpy',
        'torch',
        'torch_geometric',
        'cython',
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={'fpml': ['pkg_data/**/*']},
)
#endregions

#region classes
#endregions
