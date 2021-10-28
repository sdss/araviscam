#!/usr/bin/env python3

"""
Python3 class for fiber bundle names turned into field angles 
.. module:: fieldrotation
.. moduleauthor:: Richard J. Mathar <mathar@mpia.de>
"""

import sys
import math
import unittest

__all__ = ['Fiber', 'TestFiber']

class Fiber():
    """ A specific fiber on one of the 4 telescopes
    """
    # cos(30 deg)=sqrt(3)/2 relevant in hexagonal lattices
    # For sin(30 deg)=1/2 we don't need a special value
    SQRT3_2 = 0.866025403784438646763723

    # pi/3 equivalent to 60 degrees relevant in hexagonal lattices
    PI_3 = 1.04719755119659774615421446

    def __init__(self, name):
        """ 
        :param name
            The name is one of S1-1 to S1-600, S2-1 to S2-600 and S3-1 to S3-601
            on the science IFU. One of A1-1 to A1-19, A2-1 to A2-20 and A3-1 to A3-21
            (and A replaced by B) on the sky background telescopes, And P1-1 to P1-12 and P2-1 to P2-12
            on the spectro-photometric channel. See LVMi-0086 for details
        : type name string
        """
        self.name = name

        # 330 micron core to core distance on the hexagonal lattice
        # supposed to be the same on all 4 fiber bundles.
        self.pitch = 330.0

    def xyFocalPlaneSAB(self, pixperrow):
        """
        :return east and up location in units of mm for Sciene IFU or background
        """
        idx = int(self.name[3:])
        # Note that there is no need to check the upper index
        # because that's done below.
        if idx < 1 :
            raise NameError("invalid fiber name " + self.name) 

        # Start counting idx 0-based from here on (!)
        # so idx = 0-599 in S1, 0-600 in S3, 0-599 in S2
        # 0-19 in A1 or B1, 0-19 in A2 or B2, 0-20 in A3 or B3
        idx -= 1
 
        nameStrt = self.name[1:3]
        # the part of the third of the bundle that contains
        # fiber number 1 is treated specially here such
        # that afterwards the indices in the 3 sectors can be treated alike.
        if nameStrt == '1-' :
            # row 0 = idx 0 to 23
            # row 1 = idx 24 to 47
            # row 24 = idx 24*24 to 599
            pass
        elif nameStrt == '2-' :
            # row 0 = idx 0 to 23
            # row 1 = idx 24 to 47
            # row 24 = idx 24*24 to 599
            pass
        else :
            # row 0 = idx 0 to 24 includes the central hexagon S3-1, A3-1, B3-1
            # row 1 = idx 25 to 48
            # row 24 = idx 24*24+1 to 600
            if idx == 0 :
                return 0,0
            else :
                # align index scheme to the other 2 bundles
                # so the intermediate coordinate system is at S3-2, A3-2, B3-2
                idx -= 1

        # split the index in a row number (row =consecutive indices)
        # and an index within the meandring rows.
        row = idx // pixperrow
        # if the index in the name was too large, we're not allowing it...
        if row > pixperrow :
            raise NameError("invalid fiber name " + self.name) 
        idx %= pixperrow

        # even row = index up = outwarrds, odd row = index up = inwards
        # undo the meander by countin all columns inwards out, also for odd rows
        if (row % 2) != 0 :
            idx = pixperrow - idx - 1

        # use 3 different unit vectors along row directions and
        # at angles of 120 degrees in the hex lattice
        idx += 1
        if nameStrt == '1-' :
            # for each increase of row++ (x,y) += ( sqrt(3)/2,-1/2), downwards, cos(30 deg) and sin(30deg)
            # for each increase of idx++ (x,y) += (0,1) (and offset one up)
            x = self.SQRT3_2 * row 
            y = -0.5* row + idx
        elif nameStrt == '2-' :
            # for each increase of row++ (x,y) += ( -sqrt(3)/2,-1/2), downwards, cos(30 deg) and sin(30deg)
            # for each increase of idx++ (x,y) += ( sqrt(3)/2, -1/2) (and offset one up)
            x = self.SQRT3_2 * ( idx - row )
            y = -0.5* (row + idx)
        else :
            # for each increase of row++ (x,y) += ( 0,1), upwards
            # for each increase of idx++ (x,y) += ( -sqrt(3)/2, -1/2) (and offset one up)
            x = -self.SQRT3_2 * idx
            y = -0.5* idx + row

        # So far the x and y coordinates are in units where
        # the distance between hexagon neigbours is (center to center) is 1.
        # Multiply by pitch to get actual physical dimensions in the focal plane
        return x*self.pitch, y*self.pitch

    def xyFocalPlaneP(self):
        """
        :return east and up location in units of mm for specphot fiber
        """
        fidx = int(self.name[3:])
        if fidx < 1 or fidx > 12 :
            raise NameError("invalid fiber name " + self.name) 

        # Start counting idx 0-based from here on (!) from 0 to 11
        fidx -= 1

        # compute position of idx=0 and idx=1 and rotate the other by 
        # multiples of 60 deg.
        # angle for idx=1 is the 60 deg complement for angle of idx=0
        # P2-1 is 23 hexagons up and 6 down (at angle of 30 deg)
        # in the 1- scheme above this is row=6, idx=23
        # P1-1 is 17 hexagons up and 5 down (at angle of 30 deg)
        # in the 1- scheme above this is row=5, idx=18
        if self.name[1] == '1':
            # inner ring P1
            row, idx = 5, 18
        else :
            # outer ring P2
            row, idx = 6, 23

        # location of x=east, y=up of P2-1 or P1-1
        x = self.SQRT3_2 * row
        y = -0.5* row + idx
        # angle relative to 0=up, east=90
        baseAng = math.atan2(x,y)
        # for odd 0-based indices use 60 deg = pi/3 rad complement
        # print("base Ang " + str(math.degrees(baseAng)) + " fidx " + str(idx) + " " + self.name)
        if ( fidx % 2 ) != 0 :
            baseAng = self.PI_3 - baseAng

        # fidx=0,1 ->0 fidx=2,3 -> 1, each time the index increases by 2,
        # the rotation angle increases by 60 deg clockwise.
        rot60 = fidx // 2
        baseAng += self.PI_3*rot60
        return self.pitch*math.sin(baseAng), self.pitch*math.cos(baseAng)

    def xyFocalPlane(self):
        """ 
          Compute the position in the focal plane, where first coordinate
          is horizontally E and second coordinate up (orthogonally) .
        : return pair x,y in microns
        """
        nameStrt = self.name[0:3]
        if nameStrt == 'S1-' or nameStrt == 'S2-' or nameStrt == 'S3-' :
            return self.xyFocalPlaneSAB(24)
        elif nameStrt == 'A1-' or nameStrt == 'A2-' or nameStrt == 'A3-' :
            return self.xyFocalPlaneSAB(4)
        elif nameStrt == 'B1-' or nameStrt == 'B2-' or nameStrt == 'B3-' :
            return self.xyFocalPlaneSAB(4)
        elif nameStrt == 'P1-' or nameStrt == 'P2-' :
            return self.xyFocalPlaneP()
        else :
            raise NameError("invalid fiber name " + self.name) 

    def labAngle(self):
        """ 
        :return the laboratory angle (direction in the local FP) in radians
            The angle 0 is up, and +90 deg (in radians) is horizontally to the E,
            i.e. to the right in the front view of the fiber bundle.
        """
        e, up = self.xyFocalPlane()
        # return (math.pi/2.0 -math.atan2(up,e)) # worse: might be outside +-180deg
        return math.atan2(e,up)

class TestFiber(unittest.TestCase):
    """ Test numerics of the Fiber class
    """
    def test_S(self):
        """ Test locations of fibers in the range S1-1 to S3-600
        """
        fib = Fiber("S4-1")
        self.assertRaises(NameError,fib.labAngle)

        fib = Fiber("S1-1")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,0.0)

        fib = Fiber("S1-2")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,0.0)

        fib = Fiber("S1-24")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,0.0)

        fib = Fiber("S1-25")
        degs = math.degrees(fib.labAngle())
        self.assertGreater(degs,0.0)

        fib = Fiber("S1-48")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,60.0)

        fib = Fiber("S1-49")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,90.0)

        fib = Fiber("S1-600")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,60.0)

        fib = Fiber("S1-601")
        self.assertRaises(NameError,fib.labAngle)

        fib = Fiber("S2-1")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,120.0)

        fib = Fiber("S2-24")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,120.0)

        fib = Fiber("S2-25")
        degs = math.degrees(fib.labAngle())
        self.assertGreater(degs,120.0)

        fib = Fiber("S2-48")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,180.0)

        fib = Fiber("S2-49")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-150.0)

        fib = Fiber("S2-50")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,180.0)

        fib = Fiber("S2-600")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,180.0)

        fib = Fiber("S3-1")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-120.0)

        fib = Fiber("S3-25")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-120.0)

        fib = Fiber("S3-601")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-60.0)

    def test_P(self):
        """ Test locations of fibers in the range P1-1 to P2-12
        """
        fib = Fiber("P3-1")
        self.assertRaises(NameError,fib.labAngle)

        fib = Fiber("P2-0")
        self.assertRaises(NameError,fib.labAngle)

        fib = Fiber("P1-13")
        self.assertRaises(NameError,fib.labAngle)

        # Note there are rounded angles in LVMi-0086...
        fib = Fiber("P1-1")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,15.61,2)

        fib = Fiber("P1-4")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,104.39,2)

        # Note angles in LVMi-0086... are 0..360 and here -180..180
        fib = Fiber("P1-10")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-75.61,2)

        fib = Fiber("P2-1")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,14.56,2)

    def test_A(self):
        """ Test locations of fibers in the range A1-1 to B2-20
        """
        fib = Fiber("A1-18")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,90.0)

        fib = Fiber("B1-18")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,90.0)

        fib = Fiber("A2-20")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,180.0)

        fib = Fiber("B3-21")
        degs = math.degrees(fib.labAngle())
        self.assertAlmostEqual(degs,-60.0)

if __name__ == "__main__":
    """ Example application demonstrating the interface.
    """
    if False:
        unittest.main()
    
    else :
        import argparse
        parser = argparse.ArgumentParser()
    
        # there is one mandatory command line argument, the fiber name (like P1-13 or S3-50)
        parser.add_argument("fibname")
    
        args = parser.parse_args()
    
        # check that there was a command line argument
        if args.fibname is not None :
            fib = Fiber(args.fibname)
            rads = fib.labAngle()
            print("lab angle " + str(math.degrees(rads)) + " deg")
        else :
            raise NotImplementedError("fiber name missing")
