#!/usr/bin/env python3

"""
Python3 class to work with Aravis/GenICam cameras, subclass of sdss-basecam.
.. module:: araviscam
.. moduleauthor:: Richard J. Mathar <mathar@mpia.de>

It is not clear at the moment whether this will be used at all.
There are competing implementations:
https://github.com/sdss/lvmcam/tree/main/python/lvmcam
"""

import sys
import abc
import math
import asyncio
import numpy as np
import astropy
from collections import namedtuple

from logging import DEBUG, WARNING

from sdsstools.logger import StreamFormatter  

from basecam.mixins import ImageAreaMixIn, CoolerMixIn, ExposureTypeMixIn
from basecam import CameraSystem, BaseCamera, CameraEvent, CameraConnectionError
from basecam.models import FITSModel, Card, WCSCards
from astropy import wcs

# Since the aravis wrapper for GenICam cameras (such as the Blackfly)
# is using glib2 GObjects to represent cameras and streams, the
# PyGObject module allows to call the C functions of aravis in python.
# https://pygobject.readthedocs.io/en/latest/
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis


# https://pypi.org/project/sdss-basecam/
# https://githum.com/sdss/basecam/

__all__ = ['BlackflyCameraSystem', 'BlackflyCamera']

class GainMixIn(object, metaclass=abc.ABCMeta):
    """A mixin that provides manual control over the camera gain."""

    @abc.abstractmethod
    async def _set_gain_internal(self, gain):
        """Internal method to set the gain."""

        raise NotImplementedError

    @abc.abstractmethod
    async def _get_gain_internal(self):
        """Internal method to get the gain."""

        raise NotImplementedError

    async def set_gain(self, gain):
        """Seta the  gain of the camera."""

        return await self._set_gain_internal(gain)

    async def get_gain(self):
        """Gets the gain of the camera."""

        return await self._get_gain_internal()


class BlackflyCameraSystem(CameraSystem):
    """ A collection of GenICam cameras, possibly online
    :param camera_class : `.BaseCamera` subclass
        The subclass of `.BaseCamera` to use with this camera system.
    :param camera_config : 
        A dictionary with the configuration parameters for the multiple
        cameras that can be present in the system, or the path to a YAML file.
        Refer to the documentation for details on the accepted format.
    :type camera_config : dict or path
    :param include : List of camera UIDs that can be connected.
    :type include : list
    :param exclude : list
        List of camera UIDs that will be ignored.
    :param logger : ~logging.Logger
        The logger instance to use. If `None`, a new logger will be created.
    :param log_header : A string to be prefixed to each message logged.
    :type log_header : str
    :param log_file : The path to which to log.
    :type log_file : str
    :param verbose : Whether to log to stdout.
    :type verbose : bool
    :param ip_list: A list of IP-Adresses to be checked/pinged.
    :type ip_list: List of strings.
    """

    from araviscam import __version__

    ## A list of ip addresses in the usual "xxx.yyy.zzz.ttt" or "name.subnet.net"
    ## format that have been added manually/explicitly and may not be found by the
    ## usual broadcase auto-detection (i.e., possibly on some other global network).
    #ips_nonlocal = []

    def __init__(self, camera_class=None, camera_config=None,
                 include=None, exclude=None, logger=None,
                 log_header=None, log_file=None, verbose=False, ip_list=None):
        super().__init__(camera_class=camera_class, camera_config=camera_config,
                         include=include, exclude=exclude, logger=logger, log_header=log_header,
                         log_file=log_file, verbose=verbose)

    def list_available_cameras(self):

        return self.cameras

Rect = namedtuple('Rect', ['x0', 'y0', 'wd', 'ht'])

class BlackflyCamera(BaseCamera, ExposureTypeMixIn, ImageAreaMixIn, CoolerMixIn, GainMixIn):
    """ A FLIR (formerly Point Grey Research) Blackfly camera.
    Given the pixel scale on the benches of LVMi and the assumption
    of 9 um pixel sizes of the LVMi cameras, we assume that the
    cameras have roughly 1 arsec per pixel, so they are used without binning.

    In addition we let the camera flip the standard image orientation of the data
    values assuming that values are stored into a FITS interface (where
    the first values in the sequential data are the bottom row).
    So this is not done in this python code but by the camera.

    ..todo: is there anything here to be adopted from https://github.com/sdss/LVM_FLIR_Software ?
    """

    #TEST_CARDS: Dict[str, DefaultCard] = {
        #"EXPTIME": DefaultCard(
            #"EXPTIME",
            #value="{__exposure__.exptime}",
            #comment="Exposure time of single integration [s]",
            #type=float,
        #),
    #}
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.logger.sh.setLevel(DEBUG)
        self.logger.sh.formatter = StreamFormatter(fmt='%(asctime)s %(name)s %(levelname)s %(filename)s:%(lineno)d: \033[1m%(message)s\033[21m') 


        self.scraper_data = self.camera_params.get('scraper_data', {})
        self.logger.debug(f"{self.scraper_data}")

        self.gain = -1
        self.hbin, vbin = -1, -1
        self.cam_type = "unknown"
        self.cam_temp = -1
        self.roi = Rect(0, 0, -1, -1) # x0, y0, width, height 
        self.pixsize = self.camera_params.get("pixsize", None)
        self.pixscale = self.camera_params.get("pixscale", None)
        self.flen = self.camera_params.get("flen", 1839.8)
        self.wcs = WCSCards()

        model = self.fits_model[0].header_model
        model.append(Card("GAIN", value="{__exposure__.camera.gain}", comment="[ct] Camera gain"))
        model.append(Card("CamType", value="{__exposure__.camera.cam_type}", comment="Camera model"))
        model.append(Card("CamTemp", value="{__exposure__.camera.cam_temp}", comment="[C] Camera Temperature"))
        model.append(Card("BinX", value="{__exposure__.camera.hbin}", comment="[ct] Horizontal Bin Factor 1, 2 or 4"))
        model.append(Card("BinY", value="{__exposure__.camera.vbin}", comment="[ct] Vertical Bin Factor 1, 2 or 4"))
        model.append(Card("RoiX0", value="{__exposure__.camera.roi.x0}", comment="[Pixel] Roi x0"))
        model.append(Card("RoiY0", value="{__exposure__.camera.roi.y0}", comment="[Pixel] Roi y0"))
        model.append(Card("RoiWd", value="{__exposure__.camera.roi.wd}", comment="[Pixel] Roi Width"))
        model.append(Card("RoiHt", value="{__exposure__.camera.roi.ht}", comment="[Pixel] Roi Height"))
        model.append(self.wcs)

        self.logger.debug(f"{self.fits_model[0].header_model}")
#        self.logger.debug(f"{self.camera_params}")
        

    async def _connect_internal(self, **kwargs):
        """Connect to a camera and upload basic binning and ROI parameters.
        :param kwargs:  not used
        """
        self.logger.debug(f"connect {self.name} {self.uid}")
        ip = self.camera_params.get("ip")

        self.logger.debug(f"{ip}")
#        self.logger.debug(self.camera_system.list_available_cameras())
        
        self.cam = Aravis.Camera.new(ip)
        self.cam_type = self.cam.get_model_name()

        # search for an optional gain key in the arguments
        # todo: one could interpret gain=0 here as to call set_gain_auto(ARV_AUTO_ON)
        await self.set_gain(self.camera_params.get('gain', 0))

        # search for an optional x and y binning factor, fullframe image area will be set automatically with the binning.
        await self.set_binning(*self.camera_params.get('binning', [1,1]))
        
        # see arvenums.h for the list of pixel formats. This is MONO_16 here, always
        self.cam.set_pixel_format(0x01100007)

        # scan the general list of genicam featured values
        # of the four native types
        for typp, arvLst in self.camera_params.get("genicam_params", {}).items():
            if arvLst is not None:
                if typp == 'bool':
                    for genkey, genval in arvLst.items():
                        try:
                            if self.cam.get_boolean(genkey) != genval:
                               self.cam.set_boolean(genkey, int(genval))
                            self.logger.debug(f"genicam param : {genkey}={genval}")
                        except Exception as ex:
                            self.logger.error(f"failed setting: {genkey}={genval} {ex}")
                elif typp == 'int':
                    for genkey, genval in arvLst.items():
                        try:
                            self.cam.set_integer(genkey, genval)
                            self.logger.debug(f"genicam param : {genkey}={genval}")
                        except Exception as ex:
                            self.logger.error(f"failed setting: {genkey}={genval} {ex}")
                elif typp == 'float':
                    for genkey, genval in arvLst.items():
                        try:
                            self.cam.set_float(genkey, genval)
                            self.logger.debug(f"genicam param : {genkey}={genval}")
                        except Exception as ex:
                            self.logger.error(f"failed setting: {genkey}={genval} {ex}")
                elif typp == 'string':
                    for genkey, genval in arvLst.items():
                        try:
                            self.logger.debug(f"genicam param : {genkey}={genval}")
                            self.cam.set_string(genkey, genval)
                        except Exception as ex:
                            self.logger.error(f"failed setting: {genkey}={genval} {ex}")


    async def _disconnect_internal(self):
        """Close connection to camera.
        """
        self.cam = None

    async def _expose_grabFrame(self, exposure):
        """ Read a single unbinned full frame.
        The class splits the parent class' exposure into this function and
        the part which generates the FITS file, because applications in guiders
        are usually only interested in the frame's data, and would not
        take the detour of generating a FITS file and reading it back from
        disk.

        :param exposure:  On entry, exposure.exptim is the intended exposure time in [sec]
                          On exit, exposure.data is the numpy array of the 16bit data
                          arranged in FITS order (i.e., the data of the bottom row appear first...)
        :return: The dictionary with the window location and size (x=,y=,width=,height=)
        """

        # To avoid being left over by other programs with no change
        # to set the exposure time, we switch the auto=0=off first
        self.cam.set_exposure_time_auto(0)
        # Aravis assumes exptime in micro second integers
        exptime_ms = int(0.5 + exposure.exptime * 1e6)
        self.cam.set_exposure_time(exptime_ms)

        # timeout (factor 2: assuming there may be two frames in auto mode taken
        #   internally)
        #   And 5 seconds margin for any sort of transmission overhead over PoE
        tout_ms = int(1.0e6 * (2.*exposure.exptime+5))
        self.notify(CameraEvent.EXPOSURE_INTEGRATING)

        # the buffer allocated/created within the acquisition()
        buf = await self.loop.run_in_executor(None, self.cam.acquisition, tout_ms)
        if buf is None:
            raise ExposureError("Exposing for " + str(exposure.exptime) +
                                " sec failed. Timout " + str(tout_ms/1.0e6))

        # Decipher which methods this aravis buffer has...
        # print(dir(buf))

        # reg becomes a x=, y=, width= height= dictionary
        # these are in standard X11 coordinates where upper left =(0,0)
        reg = buf.get_image_region()
        # print('region',reg)

        data = buf.get_data()

        exposure.data = np.ndarray(buffer=data, dtype=np.uint16,
                                      shape=(1, reg.height, reg.width))
        # print("exposure data shape", exposure.data.shape)

        return reg

    async def _expose_internal(self, exposure, **kwargs):
        """ Read a single unbinned full frame and store in a FITS file.
        :param exposure:  On entry exposure.exptim is the intended exposure time in [sec]
                  On exit, exposure.data contains the 16bit data of a single frame
        :return: There is no return value
        """

        # fill exposure.data with the frame's 16bit data
        # reg becomes a x=, y=, width= height= dictionary
        # these are in standard X11 coordinates where upper left =(0,0)
        reg = await self._expose_grabFrame(exposure)

        params = {k: v.val for k,v in self.scraper_data.items()}
        
        if kmirror_angle := kwargs.get("km_d", 0.0):
            params["km_d"] = kmirror_angle

        if ra_h := kwargs.get("ra_h", None):
            if dec_d := kwargs.get("dec_d", None):
                params["ra_h"] = ra_h
                params["dec_d"] = dec_d


        self.logger.debug(f"{params}")

        self._expose_wcs(exposure, reg)
        
        self.cam_temp = await self.get_temperature()
 
    def _expose_wcs(self, exposure, reg):
        """ Gather information for the WCS FITS keywords
        :param exposure:  On entry exposure.exptim is the intended exposure time in [sec]
                  On exit, exposure.data contains the 16bit data of a single frame
        :param reg The binning and region information 
        """
        ## the section/dictionary of the yaml file for this camera
        #yamlconfig = self.camera_system._config[self.name]
#        wcsHeaders = []

        self.wcs = wcs.WCS()
        self.wcs.cdelt = np.array([-0.066667, 0.066667])
        self.wcs.crval = [0, -90]
        self.wcs.cunit = ["deg", "deg"]
        self.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # The distance from the long edge of the FLIR camera to the center
        # of the focus (fiber) is 7.144+4.0 mm according to SDSS-V_0110 figure 6
        # and 11.14471 according to figure 3-1 of LVMi-0081
        # For the *w or *e cameras the pixel row 1 (in FITS) is that far
        # away in the y-coordinate and in the middle of the x-coordinate.
        # For the *c cameras at the fiber bundle we assume them to be in the beam center.
        crpix1 = reg.width / 2
        crpix2 = 11.14471 * 1000.0 / self.pixsize
        self.wcs.crpix = [crpix1, crpix2]

    def _status_internal(self):
        return {"temperature": self.cam.get_float("DeviceTemperature"), 
                "cooler": math.nan}

    async def _get_binning_internal(self):
        return self.cam.get_binning()

    async def _set_binning_internal(self, hbin, vbin):
        # search for an optional x and y binning factor
        try:
            self.logger.debug(f"set binning: {hbin} {vbin}")
            self.cam.set_binning(hbin, vbin)
            self.hbin = hbin
            self.vbin = vbin

        except Exception as ex:
            # horizontal and vertical binning set to 1
            self.logger.error(f"failed to set binning: {self.camera_params.get('binning', None)}  {ex}")
            
        await self._set_image_area_internal()

    async def _get_image_area_internal(self):
        self.regionBounds = [self.cam.get_width_bounds().max, self.cam.get_height_bounds().max]
        return self.regionBounds

    async def _set_image_area_internal(self, area=None):
        if area:
            self.logger.warning(f"image area only with fullframe")
            return # not supported
        await self._get_image_area_internal()
        self.cam.set_region(0, 0, *self.regionBounds)
        self.roi = Rect(0, 0, *self.regionBounds) # x0, y0, width, height 

    async def _get_temperature_internal(self):
        return self.cam.get_float("DeviceTemperature")

    async def _set_temperature_internal(self, temperature):
        self.logger.warning(f"temperature setting not possible")

    async def _set_gain_internal(self, gain):
        """Internal method to set the gain."""
        try:
            self.logger.debug(f"set gain: {self.camera_params.get('gain', None)}")
            if gain == 0.0:
                self.cam.set_gain_auto(1)
            else:
                self.cam.set_gain_auto(0)
                mn, mx = self.cam.get_gain_bounds( )
                self.cam.set_gain(max(min(mx, gain), mn))
                self.gain = gain

        except Exception as ex:
            self.logger.error(f"failed to set gain: {gain} {ex}")
        

    async def _get_gain_internal(self):
        """Internal method to get the gain."""
        return self.camera_params.get('gain', None)


async def singleFrame(exptim, name, verb=False, ip_add=None, config="cameras.yaml", targ=None,
                      kmirr=0.0, flen =None):
    """ Expose once and write the image to a FITS file.
    :param exptim: The exposure time in seconds. Non-negative.
    :type exptim: float
    :param verb: Verbosity on or off
    :type verb: boolean
    :param ip_add: list of explicit IP's (like 192.168.70.51 or lvmt.irws2.mpia.de)
    :type ip_add: list of strings
    :param config: Name of the YAML file with the cameras configuration
    :type config: string of the file name
    :param targ: alpha/delta ra/dec of the sidereal target
    :type targ: astropy.coordinates.SkyCoord
    :param kmirr: Kmirr angle in degrees (0 if up, positive with right hand rule along North on bench)
    :type kmirr: float
    :param flen: focal length of telescope/siderostat in mm
                 If not provided it will be taken from the configuration file
    :type flen: float
    """

    cs = BlackflyCameraSystem(
        BlackflyCamera, camera_config=config, verbose=verb, ip_list=ip_add)
    cam = await cs.add_camera(name=name)
    # print("cameras", cs.cameras)
    # print("config" ,config)
 

    exp = await cam.expose(exptim, "LAB TEST")

    if targ is not None and kmirr is not None:
        # if there is already a (partial) header information, keep it,
        # otherwise create one ab ovo.
        if exp.wcs is None :
            wcshdr = astropy.io.fits.Header()
        else :
            wcshdr = exp.wcs.to_header()

        key = astropy.io.fits.Card("CUNIT1","deg","WCS units along axis 1")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CUNIT2","deg","WCS units along axis 2")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CTYPE1","RA---TAN","WCS type axis 1")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CTYPE2","DEC--TAN","WCS type axis 2")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CRVAL1",targ.ra.deg,"[deg] RA at reference pixel")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CRVAL2",targ.dec.deg,"[deg] DEC at reference pixel")
        wcshdr.append(key)

        # field angle: degrees, then radians
        # direction of NCP on the detectors (where we have already flipped pixels
        # on all detectors so fieldrot=kmirr=0 implies North is up and East is left)
        # With right-handed-rule: zero if N=up (y-axis), 90 deg if N=right (x-axis)
        # so the direction is the vector ( sin(f), cos(f)) before the K-mirror.
        # Action of K-mirror is ( cos(2*m), sin(2*m); sin(2*m), -cos(2*m))
        # and action of prism is (-1 0 ; 0 1), i.e. to flip the horizontal coordinate.
        # todo: get starting  value from a siderostat field rotation tracking model
        fieldrot = 0.0

        
        if name[-1] == 'c' :
            # without prism, assuming center camera placed horizontally
            if name[:4] == 'spec' :
                # without K-mirror
                pass
            else :
                # with K-mirror
                # in the configuration the y-axis of the image has been flipped,
                # the combined action of (1, 0; 0, -1) and the K-mirror is (cos(2m), sin(2m); -sin(2m), cos(2m))
                # and applied to the input vector this is (sin(2m+f), cos(2m+f))
                fieldrot += 2.*kmirr
        else :
            # with prism
            if name[:4] == 'spec' :
                # without K-mirror
                # Applied to input beam this gives (-sin(f), cos(f)) but prism effect
                # had been undone by vertical flip in the FLIR image. 
                pass
            else :
                # with K-mirror
                # Combined action of K-mirror and prism is (-cos(2*m), -sin(2*m);sin(2*m), -cos(2*m)).
                # Applied to input beam this gives (-sin(2*m+f), -cos(2*m+f)) = (sin(2*m+f+pi), cos(2*m+f+pi)).
                fieldrot += 2.*kmirr+180.0

            if name[-1] == 'w' :
                # Camera is vertically,
                # so up in the lab is right in the image
                fieldrot += 90
            else :
                # Camera is vertically,
                # so up in the lab is left in the image
                fieldrot -= 90

        fieldrot = math.radians(fieldrot)

        # the section/dictionary of the yaml file for this camera
        yamlconfig = cs._config[name]

        if flen is None:
            flen = yamlconfig["flen"]

        # pixel scale per arcseconds is focal length *pi/180 /3600
        # = flen * mm *pi/180 /3600
        # = flen * um *pi/180 /3.6, so in microns per arcsec...
        pixscal = math.radians(flen)/3.6

        # degrees per pixel is arcseconds per pixel/3600 = (mu/pix)/(mu/arcsec)/3600
        degperpix =  yamlconfig["pixsize"]/pixscal/3600.0

        # for the right handed coordinates
        # (pixx,pixy) = (cos f', -sin f'; sin f', cos f')*(DEC,RA) where f' =90deg -fieldrot
        # (pixx,pixy) = (sin f, -cos f; cos f , sin f)*(DEC,RA)
        # (sin f, cos f; -cos f, sin f)*(pixx,pixy) = (DEC,RA)
        # (-cos f, sin f; sin f, cos f)*(pixx,pixy) = (RA,DEC)
        # Note that the det of the WCS matrix is negativ (because RA/DEC is left-handed...)
        cosperpix = degperpix*math.cos(fieldrot) 
        sinperpix = degperpix*math.sin(fieldrot) 
        key = astropy.io.fits.Card("CD1_1",-cosperpix,"[deg/px] WCS matrix diagonal")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CD2_2",cosperpix,"[deg/px] WCS matrix diagonal")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CD1_2",sinperpix,"[deg/px] WCS matrix outer diagonal")
        wcshdr.append(key)
        key = astropy.io.fits.Card("CD2_1",sinperpix,"[deg/px] WCS matrix outer diagonal")
        wcshdr.append(key)

        exp.wcs = astropy.wcs.WCS(wcshdr)
        # print(exp.wcs.to_header_string()) 
        for headr in wcshdr.cards :
            exp.fits_model[0].header_model.append(Card(headr))

    await exp.write()
    if verb:
        print("wrote ", exp.filename)
