import unittest
import numpy as np
import fake_numpy as fnp

#This test module is a temporary hack
class TestNdarrayNdarray(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def test_emptyShape(self):
        shape = ()
        self.assertEqual(np.ndarray(shape).shape, fnp.ndarray(shape).shape)

# Test overflows

class TestArray(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.scalar = 25
        super().__init__(*args, **kwargs)
    
    def arrayCalls(self, a, dtype, copy):
        return (np.array(a, dtype=dtype, copy=copy), fnp.array(a, dtype=dtype, copy=copy))
    
    def arrayShapes(self, a, dtype = None, copy = True):
        return (arr.shape for arr in self.arrayCalls(a, dtype, copy))
    
    def arrayStrides(self, a, dtype = None, copy = True):
        return (arr.strides for arr in self.arrayCalls(a, dtype, copy))
    
    def arrayTobytes(self, a, dtype = None, copy = True):
        return (arr.tobytes() for arr in self.arrayCalls(a, dtype, copy))
    
    def arraySize(self, a, dtype = None, copy = True):
        return (arr.size for arr in self.arrayCalls(a, dtype, copy))
    
    def arrayNdim(self, a, dtype=None, copy = True):
        return (arr.ndim for arr in self.arrayCalls(a, dtype, copy))
    
    def arrayItemsize(self, a, dtype=None, copy = True):
        return (arr.itemsize for arr in self.arrayCalls(a, dtype, copy))

    def test_scalarShape(self):
        self.assertEqual(*self.arrayShapes(self.scalar))
    
    def test_scalarCopy(self):
        pass

def generateArrayTests():
    dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128"
    ]
    functions = {
        #"Shapes" : TestArray.arrayShapes,
        "Strides": TestArray.arrayStrides,
        # "Tobytes",
        # "Size",
        # "Ndim",
        # "Itemsize"
    }
    data = {
        "scalar": 25
    }
    for dtype in dtypes:
        for func_name, func in functions.items():
            for name, obj in data.items():
                def test_func(self):
                    self.assertEqual(*func(self, obj, dtype=dtype))
                setattr(TestArray,
                    f"test_{name}{func_name}{dtype[0].upper() + dtype[1:]}",
                    test_func
                    )

if __name__ == "__main__":
    generateArrayTests()
    unittest.main()