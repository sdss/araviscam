#!/usr/bin/env python3

"""
Python3 class to work with Aravis/GenICam cameras, subclass of sdss-basecam.
.. module:: araviscam
.. moduleauthor:: Richard J. Mathar <mathar@mpia.de>
"""

import asyncio
import numpy

# Since the aravis wrapper for GenICam cameras (such as the Blackfly)
# is using glib2 GObjects to represent cameras and streams, the
# PyGObject module allows to call the C functions of aravis in python.
# https://pygobject.readthedocs.io/en/latest/
import gi

gi.require_version('Aravis', '0.8')
from gi.repository import Aravis

# https://pypi.org/project/sdss-basecam/
# https://githum.com/sdss/basecam/
from basecam import CameraSystem, BaseCamera, CameraEvent, CameraConnectionError

__all__ = ['BlackflyCameraSystem','BlackflyCamera']


class BlackflyCameraSystem(CameraSystem):
    """ A collection of GenICam cameras, possibly online
    """

    __version__ = "0.0.3"

    def list_available_cameras(self):
        """ Gather serial numbers of online Aravis/Genicam devices.
        :return: a list of serial numbers (as strings). This list may be
                 empty if no cameras are online/switched on.
        :rtype: list

        .. todo:: optionally implement a specific filter for Blackfly's if Basler 
                  cameras should not be listed.
        """

        # Start with (pessimistic) initially empty set of online devices
        serialNums = []

        # Scan the ethernet/bus for recognized cameras.
        # Warning/todo: this gathers also cameras that are not of the Blackfly class,
        # and in conjunction with the SDSS may also recognize the Basler cameras..
        Aravis.update_device_list()
        Ndev = Aravis.get_n_devices()
        # print(str(Ndev) + " cameras online")

        # get_device_id returns a string of type, SN, MAC etc
        for i in range(Ndev) :
            cam = Aravis.Camera.new(Aravis.get_device_id(i))
            uid = cam.get_string("DeviceSerialNumber")
            # print('online SN ' + str(i) + " : " + uid)
            serialNums.append(uid)

        return serialNums


class BlackflyCamera(BaseCamera):
    """ A FLIR (formerly Point Grey Research) Blackfly camera.
    Given the pixel scale on the benches of LVMi and the assumption
    of 9 um pixel sizes of the LVMi cameras, we assume that the
    cameras have roughly 1 arsec per pixel, so they are used without binning.

    In addition we let the camera flip the standard image orientation of the data
    values assuming that values are stored into a FITS interface (where
    the first values in the sequential data are the bottom row).
    So this is not done in this python code but by the camera.
    """

    async def _connect_internal(self, **kwargs):
        """Connect to a camera and upload basic binning and ROI parameters.
        :param kwargs:  recognizes the key uid with integer value, the serial number
                        If the key uid is absent, tries to attach to the first camera.
                        This is a subdictionary of 'cameras' in practise.
        """

        # search for an optional uid key in the arguments
        try :
            uid = kwargs['uid']
        except :
            uid = None

        if uid is None :
            # uid was not specified: grab the first device that is found
            print("no uid provided, attaching to first camera")
            cam = Aravis.Camera.new(Aravis.get_device_id(0))
        else :
            # reverse lookup of the uid in the list of known cameras
            cs = BlackflyCameraSystem(BlackflyCamera)
            slist = cs.list_available_cameras()
            # print("searching " + uid + " in " + str(slist) )
            try :
                idx = slist.index(uid)
            except ValueError :
                raise CameraConnectionError("SN " + uid + " not connected")
 
            cam = Aravis.Camera.new(Aravis.get_device_id(idx))

        # search for an optional gain key in the arguments
        # todo: one could interpret gain=0 here as to call set_gain_auto(ARV_AUTO_ON)
        try :
            gain = kwargs['gain']
            if gain > 0.0 :
                # todo: inprincple one may need a set_gain_auto(ARV_AUTO_OFF) here
                # to protect against cases where that had been set before in the camera
                cam.set_gain(gain)
        except Exception as ex :
            # print("failed to set gain " + str(ex))
            pass

        # flip vertically (reverse Y) so image in numpy is ordered according to 
        # FITS standards, where the bottom line is sequentialized first, and 
        # the lower left coordinate is at (1,1).
        cam.set_boolean("ReverseY",1)
        cam.set_boolean("ReverseX",0)

        # see arvenums.h for the list of pixel formats. This is MONO_16 here 
        cam.set_pixel_format(0x01100007)

        try :
            imgrev[0] = self.device.get_boolean("ReverseX")
            imgrev[1] = self.device.get_boolean("ReverseY")
            # print("reversed" +  str(imgrev[0]) + str(imgrev[1]) )
        except Exception as ex:
            # print("failed to read ReversXY" + str(ex))
            imgrev = None

        # should we also use a variable gain or keep it as fixed?
        #cam.Gain = cam.Gain.Min

        # horizontal and vertical binning set to 1
        cam.set_binning(1,1)

        dev = cam.get_device()

        # the binning modes are enumerations: 0 = mean, 1=sum
        # The sum is what most astronomers would expect...
        dev.set_integer_feature_value("BinningHorizontalMode",1)
        dev.set_integer_feature_value("BinningVerticalMode",1)

        # Take full frames by default (maximizing probability of LVM guide camera
        # to find guide stars in the field)
        roiBounds = [-1,-1]
        try :
            roiBounds[0] = dev.get_integer_feature_value("WidthMax")
            roiBounds[1] = dev.get_integer_feature_value("HeightMax")
            # print(" ROI " + str(roiBounds[0]) + " x " + str(roiBounds[1]) )
            cam.set_region(0,0,roiBounds[0],roiBounds[1])
        except Exception as ex:
            # print("failed to set ROI " + str(ex))
            pass

        self.device = cam
        self.regionBounds = roiBounds

    async def _disconnect_internal(self):
        """Close connection to camera.
        """
        pass

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

        # Aravis assumes exptime in micro second integers
        exptime_ms = int(0.5 + exposure.exptime * 1e6)
        self.device.set_exposure_time(exptime_ms)

        # timeout (factor 2: assuming there may be two frames in auto mode taken 
        #   internally)
        #   And 5 seconds margin for any sort of transmission overhead over PoE
        tout_ms = int( 1.0e6* (2.*exposure.exptime+5) )
        self._notify(CameraEvent.EXPOSURE_INTEGRATING)

        # the buffer allocated/created within the acquisition()
        buf = await self.loop.run_in_executor(None, self.device.acquisition, tout_ms)

        # Decipher which methods this aravis buffer has...
        # print(dir(buf))

        # reg becomes a x=, y=, width= height= dictionary
        # these are in standard X11 coordinates where upper left =(0,0)
        reg = buf.get_image_region()
        # print('region',reg) 

        data = buf.get_data()

        exposure.data = numpy.ndarray(buffer=data, dtype=numpy.uint16, 
                shape=(1,reg.height, reg.width))
        # print("exposure data shape", exposure.data.shape)

        return reg

    async def _expose_internal(self, exposure):
        """ Read a single unbinned full frame and store in a FITS file.
        :param exposure:  On entry exposure.exptim is the intended exposure time in [sec]
                  On exit, exposure.data contains the 16bit data of a single frame
        :return: There is no return value
        """

        # fill exposure.data with the frame's 16bit data
        # reg becomes a x=, y=, width= height= dictionary
        # these are in standard X11 coordinates where upper left =(0,0)
        reg = await self._expose_grabFrame(exposure)
        # print('region',reg) 

        binxy = {}
        try :
            # becomes a dictionary with dx=... dy=... for the 2 horiz/vert binn fact
            binxy = self.device.get_binning()
        except Exception as ex:
            binxy = None

        # append FITS header cards
        # For the x/y coordinates transform from X11 to FITS coordinates
        # Todo: reports the camera y-flipped reg.y if ReversY=true above??
        addHeaders = [
            ("BinX", binxy.dx, "[ct] Horizontal Bin Factor 1, 2 or 4"),
            ("BinY", binxy.dy, "[ct] Vertical Bin Factor 1, 2 or 4"),
            ("Width", reg.width, "[ct] Pixel Columns"),
            ("Height", reg.height, "[ct] Pixel Rows"),
            ("RegX", 1+reg.x, "[ct] Pixel Region Horiz start"),
            # The lower left FITS corner is the upper left X11 corner...
            ("RegY", self.regionBounds[1]-(reg.y+reg.height-1), 
                    "[ct] Pixel Region Vert start")
        ]

        dev = self.device.get_device()
        # print(dir(dev))

        try :
            gain = dev.get_float_feature_value("Gain")
            addHeaders.append(("Gain",gain,"Gain"))
        except Exception as ex:
            # print("failed to read gain" + str(ex))
            pass


        imgrev = [False, False]
        try :
            imgrev[0] = self.device.get_boolean("ReverseX")
            addHeaders.append(("ReverseX",imgrev[0] != 0," Flipped left-right"))
            imgrev[1] = self.device.get_boolean("ReverseY")
            addHeaders.append(("ReverseY",imgrev[1] != 0," Flipped up-down"))
            # print("reversed" +  str(imgrev[0]) + str(imgrev[1]) )
        except Exception as ex:
            # print("failed to read ReversXY" + str(ex))
            pass

        # This is an enumeration in the GenICam. See features list of
        #  `arv-tool-0.8 --address=192.168.70.50 features`

        binMod = [-1,-1]
        try :
            binMod[0] = dev.get_integer_feature_value("BinningHorizontalMode")
            if binMod[0] == 0 :
                addHeaders.append(("BinModeX","Averag","Horiz Bin Mode Sum or Averag"))
            else :
                addHeaders.append(("BinModeX","Sum","Horiz Bin Mode Sum or Averag"))
            binMod[1] = dev.get_integer_feature_value("BinningVerticalMode")
            if binMod[1] == 0 :
                addHeaders.append(("BinModeY","Averag","Vert Bin Mode Sum or Averag"))
            else :
                addHeaders.append(("BinModeY","Sum","Vert Bin Mode Sum or Averag"))
        except Exception as ex:
            # print("failed to read binmode" + str(ex))
            pass

        try :
            camtyp = self.device.get_model_name()
            addHeaders.append(("CAMTYP",camtyp,"Camera model"))
        except :
            pass


        for header in addHeaders:
            exposure.fits_model[0].header_model.append(header)

        # unref() is currently usupported in this GObject library.
        # Hope that this does not lead to any memory leak....
        # buf.unref()
        return


async def singleFrame(exptim, gain=-1.0, verb=False):
    """ Expose once and write the image to a FITS file.
    :param exptim: The exposure time in seconds. Non-negative.
    :type exptim: float
    :param gain: Gain in the range 0.01 to roughly 50.
                Negative values are interpreted as not to set the gain
                but to keep it as it was configured in the camera by the
                previous exposure.
    :type exptim: float
    :param verb: Verbosity on or off
    :type verb: boolean
    """
    # this is the MPIA only camera ID as of 2020-12-08
    # Note that the base camera system uses strings, not integers, here.
    serialNum = "19283186"
    # use this invalid SN to test whether the exposure is indeed refused...
    # serialNum = "1928318"
    config = {
        'cameras': {
            'lvmt1': {
                'uid': serialNum,
                'connection': {
                    'uid': serialNum,
                    'gain': gain
                }
            }
        }
    }

    cs = BlackflyCameraSystem(BlackflyCamera, camera_config=config, verbose=verb)
    cam = await cs.add_camera(uid=serialNum, autoconnect=True)
    # print("cameras", cs.cameras)
    # print("config" ,config)

    exp = await cam.expose(exptim,"LAB TEST")
    await exp.write()
    if verb :
        print("wrote ", exp.filename)

# A debugging aid and single test run
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", '--exptime', type=float, default=5.0,
                        help="Expose for for exptime seconds")
    
    parser.add_argument("-g", '--gain', type=float, default=15.0,
                        help="Set gain, range 0.01 to 50. Negative argument to keep previous value")
    
    parser.add_argument("-v", '--verbose', action='store_true',
                        help="print some notes to stdout")
    
    args = parser.parse_args()

    # The following 2 lines test that listing the connected cameras works...
    # bsys = BlackflyCameraSystem(camera_class=BlackflyCamera)
    # bsys.list_available_cameras()

    asyncio.run(singleFrame(args.exptime, gain=args.gain, verb=args.verbose))

