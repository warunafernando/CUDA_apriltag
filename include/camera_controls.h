#pragma once

#include <opencv2/highgui.hpp>
#include "config.h"
#include <iostream>

// Apply camera control settings from config to OpenCV VideoCapture
void applyCameraControls(cv::VideoCapture& cap, const AppConfig::Camera::Controls& controls);

// Read all camera control values from camera
AppConfig::Camera::Controls readCameraControls(cv::VideoCapture& cap);

// Print all camera control values to console
void printCameraControls(cv::VideoCapture& cap);

// Save current camera settings to config
bool saveCameraControlsToConfig(cv::VideoCapture& cap, const std::string& config_file);

