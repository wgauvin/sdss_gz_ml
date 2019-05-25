from setuptools import setup, find_packages

setup(
    name='redshift_nn',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    description='Redshfit NN using a keras model on Cloud ML Engine',
    author='Will Gauvin',
    author_email='wgauvin@gmail.com',
    license='MIT',
    install_requires=[
        'keras',
        'h5py'
    ],
    zip_safe=False
)
