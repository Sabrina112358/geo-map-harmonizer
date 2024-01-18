from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(
    name='geo-map-harmonizer',
    version='0.0.0.2',
    license='MIT License',
    author='Sabrina G Marques',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='sabrina.marques@unesp.br',
    keywords='gdal geoespacial LULC',
    description=u'Algorithm for harmonizing geotiff map legends',
    packages=['geomapharmonizer', 'geomapharmonizer.src'],
    python_requires='>=3.9',
    install_requires=[
        'pandas>=2.1.3',
        'numpy>=1.26.2',
        'scikit-learn>=1.3.2',
        'setuptools'
    ],
)
