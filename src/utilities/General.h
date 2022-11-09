/*
 * General.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef GENERAL_H_
#define GENERAL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <string>

#define PATH_SEP "/"
#define IMPROVEDBGMODEL false
#define VOXELCLUSTERING true
#define VOXELCOLORING true
#define VISUALIZEHISTOGRAMS false
#define ISOLATESHIRTS false
#define SHIRTABSOLUTELOWERZ 500
#define SHIRTLOWERZ 0.51
#define SHIRTUPPERZ 0.845
#define INITCAMERA 1 // which view's histograms are used for initialization
#define BINCOUNT0 4 // saturation
#define RANGEMIN0 10
#define RANGEMAX0 180
#define BINCOUNT1 4 // value
#define RANGEMIN1 16
#define RANGEMAX1 212
#define BLURDIAMETER 4 // bilateral blur
#define BLURSIGMACOLOR 60
#define BLURSIGMASPACE 60
#define HSLIDER 10  // 8
#define SSLIDER 30 // 30
#define VSLIDER 50 // 40
#define CENTROIDTHRESHOLD 400.0
#define MINCLUSTERSIZE 20
#define MAINCLUSTERCOUNT 4
#define SECONDARYCLUSTERCOUNT 3
#define PATHDISTANCETHRESHOLD 500
#define MISWAPDIFFDISTANCE 100
#define PATHIMAGEWIDTH 2000
#define PATHIMAGEHEIGHT 2000
#define PATHESTIMATIONPAST 4
#define WRONGMATCHDETECTION true
#define MATCHSWAPPING true
#define POSITIONESTIMATION true
#define MISMATCHTHRESHOLD 5
#define GRACEPERIOD 5

namespace nl_uu_science_gmt
{

// Version and Main OpenCV window name
const static std::string VERSION = "2.5";
const static std::string VIDEO_WINDOW = "Video";
const static std::string HISTOGRAM_WINDOW = "Histogram";
const static std::string SCENE_WINDOW = "OpenGL 3D scene";

// Some OpenCV colors
const static cv::Scalar Color_BLUE = cv::Scalar(255, 0, 0);
const static cv::Scalar Color_GREEN = cv::Scalar(0, 200, 0);
const static cv::Scalar Color_RED = cv::Scalar(0, 0, 255);
const static cv::Scalar Color_YELLOW = cv::Scalar(0, 255, 255);
const static cv::Scalar Color_MAGENTA = cv::Scalar(255, 0, 255);
const static cv::Scalar Color_CYAN = cv::Scalar(255, 255, 0);
const static cv::Scalar Color_WHITE = cv::Scalar(255, 255, 255);
const static cv::Scalar Color_BLACK = cv::Scalar(0, 0, 0);

class General
{
public:
	static const std::string CBConfigFile;
	static const std::string IntrinsicsFile;
	static const std::string ExtrinsicsVideo;
	static const std::string ExtrinsicsImageFile;
	static const std::string CalibrationVideo;
	static const std::string CheckerboadVideo;
	static const std::string CheckerboadCorners;
	static const std::string VideoFile;
	static const std::string VideoImageFile;
	static const std::string BackgroundVideoFile;
	static const std::string BackgroundImageFile;
	static const std::string ConfigFile;

	static bool fexists(const std::string &);
};

} /* namespace nl_uu_science_gmt */

#endif /* GENERAL_H_ */
