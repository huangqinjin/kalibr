#!/usr/bin/env python

CAMERA_FILE_HEADER="""
/**
 * @file   CameraGeometryBaseSerialization.hpp
 * @author Paul Furgale <paul.furgale@gmail.com>
 * @date   Tue Apr 16 09:43:33 2013
 * 
 * @brief  This file is needed to enable serialization using CameraGeometryBase pointers
 *
 * THIS FILE HAS BEEN AUTOGENERATED BY gen_files.py
 * 
 */


#include <boost/serialization/export.hpp>
#include <aslam/cameras.hpp>

BOOST_CLASS_EXPORT_KEY( aslam::cameras::CameraGeometryBase );
"""

FRAME_FILE_HEADER="""
#ifndef ASLAM_FRAME_BASE_SERIALIZATION_HPP
#define ASLAM_FRAME_BASE_SERIALIZATION_HPP

/**
 * @file   FrameBaseSerialization.hpp
 * @author Paul Furgale <paul.furgale@gmail.com>
 * @date   Tue Apr 16 09:43:33 2013
 * 
 * @brief  This file is needed to enable serialization using FrameBase pointers
 *
 * THIS FILE HAS BEEN AUTOGENERATED BY gen_files.py
 * 
 */


#include <boost/serialization/export.hpp>
#include <aslam/cameras.hpp>
#include <aslam/Frame.hpp>
#include \"LinkCvSerialization.hpp\"

"""


BOOST_SERIALIZATION_HEADERS="""// Standard serialization headers
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
// These ones are in sm_boost
#include <boost/portable_binary_iarchive.hpp>
#include <boost/portable_binary_oarchive.hpp>

"""

FRAME_EXPORT="""
namespace aslam {{
    
template void Frame<aslam::cameras::{0} >::save<>(boost::archive::text_oarchive & ar, const unsigned int version) const;
template void Frame<aslam::cameras::{0} >::load<>(boost::archive::text_iarchive & ar, const unsigned int version);
template void Frame<aslam::cameras::{0} >::save<>(boost::archive::xml_oarchive & ar, const unsigned int version) const;
template void Frame<aslam::cameras::{0} >::load<>(boost::archive::xml_iarchive & ar, const unsigned int version);
template void Frame<aslam::cameras::{0} >::save<>(boost::archive::binary_oarchive & ar, const unsigned int version) const;
template void Frame< aslam::cameras::{0} >::load<>(boost::archive::binary_iarchive & ar, const unsigned int version);
template void Frame<aslam::cameras::{0} >::save<>(boost::archive::portable_binary_oarchive & ar, const unsigned int version) const;
template void Frame<aslam::cameras::{0} >::load<>(boost::archive::portable_binary_iarchive & ar, const unsigned int version);

}} // namespace aslam

"""

cameras = ["PinholeCameraGeometry",
           "DistortedPinholeCameraGeometry",
           "PerspectiveDistortedPinholeCameraGeometry",
           "EquidistantDistortedPinholeCameraGeometry",
           "FovDistortedPinholeCameraGeometry",
           "OmniCameraGeometry",
           "DistortedOmniCameraGeometry",
           "EquidistantDistortedOmniCameraGeometry",
           "PinholeRsCameraGeometry",
           "DistortedPinholeRsCameraGeometry",
           "PerspectiveDistortedPinholeRsCameraGeometry",
           "EquidistantDistortedPinholeRsCameraGeometry",
           "OmniRsCameraGeometry",
           "DistortedOmniRsCameraGeometry",
           "EquidistantDistortedOmniRsCameraGeometry",
           "MaskedPinholeCameraGeometry",
           "MaskedDistortedPinholeCameraGeometry",
           "MaskedPerspectiveDistortedPinholeCameraGeometry",
           "MaskedEquidistantDistortedPinholeCameraGeometry",
           "MaskedOmniCameraGeometry",
           "MaskedDistortedOmniCameraGeometry",
           "MaskedEquidistantDistortedOmniCameraGeometry",
           "MaskedPinholeRsCameraGeometry",
           "MaskedDistortedPinholeRsCameraGeometry",
           "MaskedPerspectiveDistortedPinholeRsCameraGeometry",
           "MaskedEquidistantDistortedPinholeRsCameraGeometry",
           "MaskedOmniRsCameraGeometry",
           "MaskedDistortedOmniRsCameraGeometry",
           "MaskedEquidistantDistortedOmniRsCameraGeometry",
           "DepthCameraGeometry",
           "DistortedDepthCameraGeometry",
           "EquidistantDistortedDepthCameraGeometry"]


# write the camera header
with open('include/aslam/cameras/CameraBaseSerialization.hpp','w') as outf:
    outf.write(CAMERA_FILE_HEADER)
    for cam in cameras:
        outf.write( 'BOOST_CLASS_EXPORT_KEY(aslam::cameras::%s);\n' % cam )
    outf.write( '\n\n' )

# write the individual camera class files
cam_fnames=[]
for cam in cameras:
    fname = 'src/autogen/Camera-%s.cpp' % (cam)
    cam_fnames.append(fname)
    with open(fname,'w') as outf:
        outf.write('#include <aslam/cameras/CameraBaseSerialization.hpp>\n\n')
        outf.write(BOOST_SERIALIZATION_HEADERS)
        outf.write('BOOST_CLASS_EXPORT_IMPLEMENT(aslam::cameras::%s);\n\n' % cam)

# write the CMake File
with open('autogen_cameras.cmake','w') as outf:
    outf.write("# THIS FILE HAS BEEN AUTOGENERATED BY gen_files.py\n")
    outf.write("SET( AUTOGEN_CAMERA_CPP_FILES\n")
    for cam in cam_fnames:
        outf.write('\t%s\n' % cam)
    outf.write(')\n\n')


# write the frame header
with open('include/aslam/FrameBaseSerialization.hpp','w') as outf:
    outf.write(FRAME_FILE_HEADER)
    for cam in cameras:
        outf.write( 'BOOST_CLASS_EXPORT_KEY(aslam::Frame<aslam::cameras::%s >);\n' % cam )
    outf.write( '\n\n' )
    #outf.write( 'inline void exportFrameBase()\n' )
    #outf.write( '{\n' )
    #for cam in cameras:
    #    frame = "aslam::Frame< aslam::cameras::{0} >".format(cam)
    #    outf.write( '\tboost::serialization::void_cast_register< {0}, aslam::FrameBase>(static_cast<{0} *>(NULL), static_cast< aslam::FrameBase * >(NULL) );\n'.format(frame) )
    #outf.write( '}\n' )
    outf.write( '#endif // ASLAM_FRAME_BASE_SERIALIZATION_HPP\n\n' )
# write the individual frame class files
cam_fnames=[]
for cam in cameras:
    fname = 'src/autogen/Frame-%s.cpp' % (cam)
    cam_fnames.append(fname)
    with open(fname,'w') as outf:
        outf.write("// THIS FILE HAS BEEN AUTOGENERATED BY gen_files.py\n")
        outf.write('#include <aslam/cameras.hpp>\n')
        outf.write('#include <aslam/Frame.hpp>\n')
        outf.write('#include <aslam/FrameBaseSerialization.hpp>\n\n')
        outf.write(BOOST_SERIALIZATION_HEADERS)
        outf.write('BOOST_CLASS_EXPORT_IMPLEMENT(aslam::Frame<aslam::cameras::%s >);\n\n' % cam)
        outf.write(FRAME_EXPORT.format( cam ) )
        outf.write("\n\n")

# write the "linkCvSerialization" cpp file
with open('src/LinkCvSerialization.cpp', 'w') as outf:
    outf.write('#include <aslam/LinkCvSerialization.hpp>\n')
                  
# write the CMake File
with open('autogen_frames.cmake','w') as outf:
    outf.write("# THIS FILE HAS BEEN AUTOGENERATED BY gen_files.py\n")
    outf.write("SET( AUTOGEN_FRAME_CPP_FILES\n")
    for cam in cam_fnames:
        outf.write('\t%s\n' % cam)
    outf.write(')\n\n')
