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

    def test_scalarStridesInt8(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="int8"))

    def test_scalarStridesInt16(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="int16"))

    def test_scalarStridesInt32(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="int32"))

    def test_scalarStridesInt64(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="int64"))

    def test_scalarStridesUint8(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="uint8"))

    def test_scalarStridesUint16(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="uint16"))

    def test_scalarStridesUint32(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="uint32"))

    def test_scalarStridesUint64(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="uint64"))

    def test_scalarStridesFloat16(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="float16"))

    def test_scalarStridesFloat32(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="float32"))

    def test_scalarStridesFloat64(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="float64"))

    def test_scalarStridesComplex64(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="complex64"))

    def test_scalarStridesComplex128(self):
        self.assertEqual(*self.arrayStrides(self.scalar, dtype="complex128"))

if __name__ == "__main__":
    unittest.main()
