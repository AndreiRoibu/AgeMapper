from setuptools import setup, find_packages

setup(
    name='AgeMapper',
    version='0.0.1',
    description='AgeMapper: A Deep Learning Framework for Brain Age Estimation',
    license='BSD 3-clause license',
    maintainer='Andrei-Claudiu Roibu',
    maintainer_email='andrei-claudiu.roibu@dtc.ox.ac.uk',
    install_requires=[
        'numpy',
        'pandas',
        'torch==1.13.1',
        'fslpy',
        'tensorboardX',
        'sklearn',
        'ipython',
        'nibabel',
        'h5py',
        'hdf5',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'scipy',
        ],
)

# NOTE: THIS NEEDS to be updated, as it is an old, out-of-date version. Need to update this by the end of the project!