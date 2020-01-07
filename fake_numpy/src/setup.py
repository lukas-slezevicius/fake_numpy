from distutils.core import setup, Extension

module1 = Extension('ndarray', sources = ['ndarray.c'])

setup(
    name = 'ndarray',
    version = '1.0',
    description = 'This is ndarray',
    ext_modules = [module1]
)
