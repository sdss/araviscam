#!/usr/bin/env python3

"""
Python3 class for siderostat field angles using homogeneous coordinates
.. module:: fieldrotation
.. moduleauthor:: Richard J. Mathar <mathar@mpia.de>
"""

import sys
import math
import numpy
import astropy.coordinates
import astropy.time
import astropy.units

__all__ = ['HomTrans', 'Mirr', 'Site', 'Ambi', 'Target', 'Sider']


class HomTrans():
    """ A single coordinate transformation.
    """

    def __init__(self, entries):

        if isinstance(entries, numpy.ndarray) :
            self.matr = entries
        else :
            self.matr = numpy.array(entries, numpy.double)


    def multiply(self, rhs):
        """ 
        """
        if isinstance(rhs, HomTrans) :
            prod = numpy.matmul(self.matr,rhs.matr)
            return HomTrans(prod)
        elif isinstance(rhs, numpy.ndarray) :
            prod = numpy.matmul(self.matr,rhs)
            return HomTrans(prod)
        else :
            raise TypeError("invalid data types")

    def apply(self, rhs):
        """ 
        """
        if isinstance(rhs, numpy.ndarray) :
            if rhs.ndim == 1 :
                if rhs.shape[0] == 4 :
                    prod = numpy.dot(self.matr,rhs)
                elif rhs.shape[0] == 3 :
                    w = numpy.append(rhs,[1])
                    prod = numpy.dot(self.matr,w)

                prod /= prod[3]
                return numpy.array([prod[0],prod[1],prod[2]],numpy.double)
            else:
                raise TypeError("rhs not  a vector")

class Mirr():
    """ A flat mirror
    """

    def __init__(self, normal, disttoorg):

        if isinstance(normal, numpy.ndarray) and isinstance(disttoorg, (int,float)) :
            self.d = float(disttoorg)
            if normal.ndim == 1 and normal.shape[0] == 3:
                len = numpy.linalg.norm(normal)
                normal /= len
                self.n = normal
            else :
                raise TypeError("invalid data types")
        else :
            raise TypeError("invalid data types")

    def toHomTrans(self) :
        matr = numpy.zeros((4,4))
        for r in range(4):
            for c in range(4):
                if r == c :
                    matr[r,c] += 1.0
                if r < 3 :
                    if c < 3 :
                        matr[r,c] -= 2.0*self.n[r]*self.n[c]
                    else :
                        matr[r,c] = 2.0*self.d*self.n[r]
        return HomTrans(matr)

class Site():
    """ Geolocation of observatory
    """
    def __init__(self, long= -70.70056, lat= -29.01091, alt=2280, name=None) :
        """ Geolocation of observatory
        :param long geodetic longitude in degrees, E=positive
        :type long float
        :param lat geodetic latitude in degrees, N=positive
        :type lat float
        :param alt altitude above sea level
        :type alt float
        :param name one of the LVM site acronyms, {LCO|APO|MPIA|KHU}
        :type name string
        """

        if name is not None:
            if name == 'LCO' :
                self.long = -70.70056
                self.lat = -29.01091
                self.alt = 2280.
            elif name == 'APO' :
                self.long = -105.8202778 # -105d49m13s
                self.lat =  32.78028 # 32d46m49s
                self.alt = 2788.
            elif name == 'MPIA' :
                self.long = 8.724
                self.lat =  49.3965
                self.alt = 560.
            elif name == 'KHU' :
                self.long = 127.0533
                self.lat =  37.5970
                self.alt = 80.
        elif isinstance(long, (int,float) ) and isinstance(lat, (int,float)) and isinstance(alt,(int,float)) :
            self.long = long
            self.lat = lat
            self.alt = alt
        else :
            raise TypeError("invalid data types")

        # print("site" +str(self.long)+ "deg " + str(self.lat) + "deg")

    def toEarthLocation(self) :
        return astropy.coordinates.EarthLocation.from_geodetic(self.long, self.lat, height=self.alt)

class Ambi():
    """ ambient parameters relevant to atmospheric refraction
    :param press pressure in hPa. Can be None if the site parameter
                 is not-None so we can get an estimate from sea level altitude.
    :type press float
    :param temp Temperature in deg Celsius.
    :type temp float
    :param rhum relative humidity in the range 0..1.
    :type rhum float
    :param wlen Observing wavelength in microns.
    :type wlen float
    :param site location of observatory
    :type site fieldrotation.Site
    """
    def __init__(self, press = None, temp = 15, rhum = 0.2, wlen=0.5,  site=None) :
        """ 
        """
        self.temp = temp
        self.rhum = rhum
        self.wlen = wlen
        if press is None :
            if site is None :
                self.press = 1013.0
            elif isinstance(site, Site) :
                self.press = 1013.0*math.exp(-site.alt/8135.0)
            else :
                self.press = 1013.0
        else:
            self.press = press

class Target():
    """ sidereal astronomical target
    """
    def __init__(self, targ) :
        """ target coordinates
        :param targ Position in ecliptic coordinates
        :type targ astropy.coordinates.Skycoord
        """

        if isinstance(targ, astropy.coordinates.SkyCoord) :
            self.targ = targ
        else :
            raise TypeError("invalid data types")
        # print(self.targ)

    def toHoriz(self, site, ambi = None, time = None) :
        """ convert from elliptical to horizontal coordinates
        :param site Observatory location
        :type site fieldrotation.site
        :param ambi Ambient parameters characterizing refraction
        :type ambi fieldrotation.Ambi
        :param time time of the observation
        :type time astropy.time.Time
        :return alt-az coordinates
        :return type astropy.coordinates.AltAz
        """
        if isinstance(time, astropy.time.Time) :
            now = time 
        elif isinstance(time, str):
            now = astropy.time.Time(time, format='isot', scale='utc')
        elif time is None:
            now = astropy.time.Time.now()

        if isinstance(site, Site) :
            if ambi is None:
               refr = Ambi(site = site)
            elif isinstance(site, Ambi) :
               refr = ambi
            else :
                raise TypeError("invalid ambi data type")
            earthloc = site.toEarthLocation()
        else :
            raise TypeError("invalid site  data type")

        # print(earthloc)
        # print(astropy.units.Quantity(100.*refr.press,unit=astropy.units.Pa))
        # print(astropy.units.Quantity(refr.wlen,unit= astropy.units.um))
        # todo: render also promer motions (all 3 coords)
        # This is a blank form of Alt/aZ because the two angles are yet unknown
        # altaz = astropy.coordinates.builtin_frames.AltAz
        altaz = astropy.coordinates.AltAz(
                 location = earthloc,
                 obstime=now,
                 pressure= astropy.units.Quantity(100.*refr.press,unit=astropy.units.Pa),
                 temperature = astropy.units.Quantity(refr.temp, unit = astropy.units.deg_C),
                 relative_humidity = refr.rhum,
                 obswl = astropy.units.Quantity(refr.wlen,unit= astropy.units.um))

        horiz = self.targ.transform_to(altaz)
        return horiz

      

class Sider():
    """ A siderostat of 2 mirrors
    """
    def __init__(self, zenang=90.0, azang=0.0, medSign=1) :
        """ A siderostat of 2 mirrors
        :param zenang Zenith angle of the direction of the exit beam (degrees). Default
                      is the nominal value of the LCO LVMT.
        :type zenang float
        :param azang Azimuth angle of the direction of the exit beam (degrees), N=0, E=90.
                     Default is the nominal value of the LCO LVMT.
        :type azang float
        :param medSign Sign of the meridian flip design of the mechanics.
                       Must be either +1 or -1. Default is the LCO LVMT design (in most
                       but not all of the documentation).
        :type medSign int
        """

        if isinstance(zenang, (int,float) ) and isinstance(azang, (int,float)) :
            self.b = numpy.zeros((3))
            self.b[0] = math.sin( math.radians(azang)) * math.sin( math.radians(zenang))
            self.b[1] = math.cos( math.radians(azang)) * math.sin( math.radians(zenang))
            self.b[2] = math.cos( math.radians(zenang))
        else :
            raise TypeError("invalid data types")

        if isinstance(medSign,int) :
            if medSign == 1 or medSign == -1:
                self.sign = medSign
            else:
                raise ValueError("invalid medSign value")
        else :
            raise TypeError("invalid medSign data type")

        # axes orthogonal to beam
        self.box = numpy.zeros((3))
        self.box[0] = 0.0
        self.box[1] = -self.b[2]
        self.box[2] = self.b[1]
        self.boy = numpy.cross(self.b,self.box)
        

    def fieldAngle(self, site, target, ambi, wlen=0.5, time=None) :
        """ 
        :param site location of the observatory
        :type site fieldrotation.Site
        :param target sidereal target in ra/dec
        :type target astropy.coordinates
        :param ambi Ambient data relevant for refractive index
        :type ambi
        :param wlen wavelenght of observation in microns
        :type wlen float
        :param time time of the observation /UTC
        :type time
        :return field angle (direction to NCP) in radians
        """
        if isinstance(time, astropy.time.Time) :
            now = time 
        elif isinstance(time, str):
            now = astropy.time.Time(time, format='isot', scale='utc')
        elif time is None:
            now = astropy.time.Time.now()

        # copute mirror positions 
        horiz = target.toHoriz(site=site, ambi=ambi, time = now) 
        print(horiz)

        star = numpy.zeros((3))
        # same procedure as in the construction of b in the Sider ctor, but with 90-zenang=alt
        star[0] = math.sin( horiz.az.radian) * math.cos( horiz.alt.radian )
        star[1] = math.cos( horiz.az.radian ) * math.cos( horiz.alt.radian)
        star[2] = math.sin( horiz.alt.radian)
        # print("star",star)

        m2tom1 = numpy.cross(star,self.b)
        len = numpy.linalg.norm(m2tom1)
        m2tom1 /= self.sign*len

        m1norm = star - m2tom1
        len = numpy.linalg.norm(m1norm)
        m1norm /= len
        m1 = Mirr(m1norm,1.0)

        m2norm = self.b + m2tom1
        len = numpy.linalg.norm(m2norm)
        m2norm /= len
        m2 = Mirr(m2norm,0.0)

        m1trans = m1.toHomTrans()
        m2trans = m2.toHomTrans()
        trans = m2trans.multiply(m1trans)

        # for the field angle need a target that is just a little bit
        # more north (but not too little to avoid loss of precision)
        # print('target',target.targ)
	# 10 arcmin = 0.16 deg further N
        targNcp = Target(target.targ.spherical_offsets_by(
                       astropy.coordinates.Angle("0deg"), astropy.coordinates.Angle("0.16deg")))
        # print('targetN',targNcp.targ)
        horizNcp = targNcp.toHoriz(site=site, ambi=ambi, time = now) 
        # print('targetN atlatz',horizNcp)

        starNcp = numpy.zeros((3))
        # same procedure as in the construction of b in the Sider ctor, but with 90-zenang=alt
        starNcp[0] = math.sin( horizNcp.az.radian) * math.cos( horizNcp.alt.radian )
        starNcp[1] = math.cos( horizNcp.az.radian ) * math.cos( horizNcp.alt.radian)
        starNcp[2] = math.sin( horizNcp.alt.radian)
        # print("starNcp",starNcp)

        # image of targNcp while hitting M1
        m1img = trans.apply(m2tom1)
        # print("m1img",m1img)
        # image of targNcp before arriving at M1
        starOffm1 = m2tom1 + starNcp
        starimg = trans.apply(starOffm1)
        # print("starimg",starimg)

        # virtual direction of ray as seen from point after M2
        # no need to normalize this because the atan2 will do...
        starvirt = m1img - starimg 
        # print("diff img",starvirt)

        # project in a plane orthogonal to  self.b
        cosFang = numpy.dot(starvirt,self.box)
        sinFang = numpy.dot(starvirt,self.boy)
        return math.atan2(sinFang, cosFang)


if __name__ == "__main__":
    """ Example application demonstrating the interface.
    
    """
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("-v", '--verbose', action='store_true',
    #                     help="print some notes to stdout")

    # right ascension in degrees
    parser.add_argument("-r", '--ra', help="RA J2000 in degrees or in xxhxxmxxs format")

    # declination in degrees
    parser.add_argument("-d", '--dec', help="DEC J2000 in degrees or in +-xxdxxmxxs format")

    # shortcut for site coordinates: observatory
    parser.add_argument("-s", '--site', default="LCO", help="LCO or MPIA or APO or KHU")

    args = parser.parse_args()

    # check ranges and combine ra/dec into a single SkyCoord
    if args.ra is not None and args.dec is not None :
        if args.ra.find("h") < 0 :
            # apparently simple floating point representation
            targ = astropy.coordinates.SkyCoord(ra=float(args.ra), dec=float(args.dec),unit="deg")
        else :
            targ = astropy.coordinates.SkyCoord(args.ra + " " + args.dec)
    else :
        targ = None

    # step 1: define where the observatory is (on Earth)
    geoloc = Site(name = args.site)
    # print(geoloc)

    # step 2: define where the output beam of the siderostat points to
    sid = Sider()
    # print(sid)

    # step 3: define where the sidereostat is pointing on the sky
    point = Target(targ)
    print("target is ",targ)

    # calculate the field angle (in radians)
    rads = sid.fieldAngle(geoloc, point, None)
    print("field angle " + str(math.degrees(rads)) + " deg")
