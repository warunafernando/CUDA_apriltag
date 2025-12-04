#pragma once

#include <opencv2/highgui.hpp>
#include "config.h"

// Apply camera control settings from config to OpenCV VideoCapture
void applyCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls);

