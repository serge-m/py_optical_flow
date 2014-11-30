__author__ = 's'

import unittest
import warper
import data.image_generator
import numpy
from StringIO import StringIO
import train_function
class MyTestCase(unittest.TestCase):

    def test_warper_is_working(self):

        I0 = data.image_generator.get_image("artificial/5x5/left_right", 0)
        I1 = data.image_generator.get_image("artificial/5x5/left_right", 1)


        wrpr = warper.Warper(I0.shape, numpy.zeros_like(I0), numpy.zeros_like(I0), I0, I1, display=False)
        warps = 10
        for i in range(warps):
            wrpr.warp()


        ref_u = numpy.loadtxt(StringIO("""
             0.            0.            0.            0.            0.          
             0.            0.           -0.7724831597  0.            0.          
             0.            0.           -1.2524064595  0.            0.          
             0.            0.           -0.310634815   0.            0.          
             0.            2.9115355661  0.            0.            0.          
              0.            0.7719629276  0.            0.            0.          
              0.            0.6794089036  1.9026790588  0.9600049797  0.          
              0.            0.            0.            0.            0."""))

        ref_v = numpy.loadtxt(StringIO("""
             0.            0.            0.            0.            0.          
              0.           -0.9048892247  0.            0.            0.          
              0.            0.           -0.1565508074  0.            0.          
              0.            0.9048892247  0.5824402607  0.            0.          
              0.           -1.0918258047 -0.9407026796  0.            0.          
              0.            0.0964953659  1.3313447097  0.            0.          
              0.            0.           -0.7135046258  0.            0.          
              0.            0.            0.            0.            0.           """))

        self.assertTrue(numpy.allclose(wrpr.u, ref_u, atol=1e-8))
        self.assertTrue(numpy.allclose(wrpr.v, ref_v, atol=1e-8))

        self.assertFalse(numpy.allclose(wrpr.v, ref_v+0.1, atol=1e-8))
        #self.assertTrue(numpy.allclose(wrpr.v, ref_v+0.1, atol=1e-8))

    def test_warper_40_steps(self):

        I0 = data.image_generator.get_image("artificial/5x5/left_right", 0)
        I1 = data.image_generator.get_image("artificial/5x5/left_right", 1)


        wrpr = warper.Warper( I0.shape, numpy.zeros_like(I0), numpy.zeros_like(I0), I0, I1, 
              train_function = train_function.TrainFunctionSimple(numpy.zeros_like(I0), numpy.zeros_like(I0), rate=0.1, num_steps = 40), display=False)
        warps = 10
        for i in range(warps):
            wrpr.warp()


        ref_u = numpy.loadtxt(StringIO("""
             0.            0.            0.            0.            0.          
             0.            0.           -0.7724831597  0.            0.          
             0.            0.           -1.2524064595  0.            0.          
             0.            0.           -0.310634815   0.            0.          
             0.            2.9115355661  0.            0.            0.          
              0.            0.7719629276  0.            0.            0.          
              0.            0.6794089036  1.9026790588  0.9600049797  0.          
              0.            0.            0.            0.            0."""))

        ref_v = numpy.loadtxt(StringIO("""
             0.            0.            0.            0.            0.          
              0.           -0.9048892247  0.            0.            0.          
              0.            0.           -0.1565508074  0.            0.          
              0.            0.9048892247  0.5824402607  0.            0.          
              0.           -1.0918258047 -0.9407026796  0.            0.          
              0.            0.0964953659  1.3313447097  0.            0.          
              0.            0.           -0.7135046258  0.            0.          
              0.            0.            0.            0.            0.           """))

        self.assertTrue(numpy.allclose(wrpr.u, ref_u, atol=1e-4))
        self.assertTrue(numpy.allclose(wrpr.v, ref_v, atol=1e-4))

        self.assertFalse(numpy.allclose(wrpr.v, ref_v+0.1, atol=1e-4))
        #self.assertTrue(numpy.allclose(wrpr.v, ref_v+0.1, atol=1e-8))


if __name__ == '__main__':
    unittest.main()
