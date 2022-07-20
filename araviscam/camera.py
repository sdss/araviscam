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
from typing import Any, Dict

from logging import DEBUG, WARNING

from sdsstools.logger import StreamFormatter  

from basecam import Exposure, CameraSystem, BaseCamera, CameraEvent, CameraConnectionError
from basecam.mixins import ImageAreaMixIn, CoolerMixIn, ExposureTypeMixIn
from basecam.models import FITSModel, Card, MacroCard, WCSCards

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

Point = namedtuple('Point', ['x0', 'y0'])
Size = namedtuple('Size', ['wd', 'ht'])
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
    """

    class WCSCards(MacroCard):
       name = "WCS information"

       def __init__(self, camera):
            super().__init__()
            self.camera = camera
            
       def macro(self, exposure: Exposure, context: Dict[str, Any] = {}):
            print(exposure.reg)            
            print(context)            
            if exposure.wcs is None:
                wcs = astropy.wcs.WCS()
            else:
                wcs = exposure.wcs
            return list(wcs.to_header().cards)

    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.logger.sh.setLevel(DEBUG)
        self.logger.sh.formatter = StreamFormatter(fmt='%(asctime)s %(name)s %(levelname)s %(filename)s:%(lineno)d: \033[1m%(message)s\033[21m') 

        self.scraper_store = self.camera_params.get('scraper_store', {})

        self.gain = -1
        self.binning = [-1, -1]
        self.cam_type = "unknown"
        self.temperature = -1

        self.site = self.camera_params.get('site', "LCO")

        self.detector_size = Size(-1, -1)
        self.region_bounds=Size(-1, -1)
        self.image_area=Rect(-1, -1, -1, -1)

        self.pixsize = self.camera_params.get('pixsize', 0.0)
        self.flen = self.camera_params.get('flen', 0.0)
        # pixel scale per arcseconds is focal length *pi/180 /3600
        # = flen * mm *pi/180 /3600
        # = flen * um *pi/180 /3.6, so in microns per arcsec...
        self.pixscale = math.radians(self.flen)/3.6
        self.arcsec_per_pix = self.pixsize / self.pixscale
        self.log(f"arcsec_per_pix {self.arcsec_per_pix}")
        # degrees per pixel is arcseconds per pixel/3600 = (mu/pix)/(mu/arcsec)/3600
        self.degperpix =  self.pixsize/self.pixscale/3600.0


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

#        self.logger.debug(f"{self.cam.get_binning()}")
        # self.detector_size = list(self.cam.get_sensor_size()) # returns [+8, +4]
        try:
            await self.set_binning(1,1)
        except Exception as ex:
            self.logger.warning(f"{ex}")
            await asyncio.wait(5.0)
            self.logger.warning(f"{ex}")
            await self.set_binning(1,1)

        self.detector_size = Size(self.cam.get_width_bounds().max, self.cam.get_height_bounds().max)
        self.logger.debug(f"{self.detector_size}")

        # search for an optional gain key in the arguments
        # todo: one could interpret gain=0 here as to call set_gain_auto(ARV_AUTO_ON)
        await self.set_gain(self.camera_params.get('gain', 0))

        # search for an optional x and y binning factor, fullframe image area will be set automatically with the binning.
        await self.set_binning(*self.camera_params.get('binning', [1,1]))
        self.logger.debug(f"{self.image_area}")
        
        
        
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

        return buf.get_data(), buf.get_image_region()


    async def _expose_internal(self, exposure, **kwargs):
        """ Read a single full frame and store in a FITS file.
        :param exposure:  On entry exposure.exptim is the intended exposure time in [sec]
                  On exit, exposure.data contains the 16bit data of a single frame
        :return: There is no return value
        """

        exposure.scraper_store = self.scraper_store.copy()

        if kmirror_angle := kwargs.get("km_d", None):
            exposure.scraper_store.set("km_d", kmirror_angle)

        if ra_h := kwargs.get("ra_h", None):
            if dec_d := kwargs.get("dec_d", None):
                exposure.scraper_store.set("ra_h", ra_h)
                exposure.scraper_store.set("dec_d", dec_d)


        # fill exposure.data with the frame's 16bit data
        # reg becomes a x=, y=, width= height= dictionary
        # these are in standard X11 coordinates where upper left =(0,0)
        data, roi = await self._expose_grabFrame(exposure)

        self.logger.debug(f"{roi} {self.image_area}")

        exposure.data = np.ndarray(
            buffer=data,
            dtype=np.uint16,
            shape=(1, self.image_area.ht, self.image_area.wd)
        )
        self.temperature = await self.get_temperature()
 

    def _status_internal(self):
        return {"temperature": self.cam.get_float("DeviceTemperature"), 
                "cooler": math.nan}

    async def _get_binning_internal(self):
        return list(self.cam.get_binning())

    async def _set_binning_internal(self, hbin, vbin):
        # search for an optional x and y binning factor
        try:
            self.logger.debug(f"set binning: {hbin} {vbin}")
            self.cam.set_binning(hbin, vbin)
            self.binning = [hbin, vbin]

        except Exception as ex:
            # horizontal and vertical binning set to 1
            self.logger.error(f"failed to set binning: {[hbin, vbin]}  {ex}")
            
        await self._set_image_area_internal()

    async def _get_image_area_internal(self):
        self.region_bounds = Size(self.cam.get_width_bounds().max, self.cam.get_height_bounds().max)
        return self.region_bounds

    async def _set_image_area_internal(self, area=None):
        if area:
            self.logger.warning(f"image area only with fullframe")
            return # not supported
        await self._get_image_area_internal()
        self.cam.set_region(0, 0, *self.region_bounds)
        self.image_area = Rect(0, 0, *self.region_bounds) # x0, y0, width, height 

    async def _get_temperature_internal(self):
        self.temperature = self.cam.get_float("DeviceTemperature")
        return self.temperature

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


 
