#pragma once
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//refactor for LICENSE PLATE	

class PanoramicImage {

	// Methods

public:

	// constructor 
	PanoramicImage(cv::String namefolder);

	// project images
	void projectImages(double angle);

	// extract features
	void extractFeatures();

	//match features
	void matchFeatures1(float ratio);

	void matchFeatures2();

	//get the Window Size
	void showPanoramicImage();
	// Data



protected:

	// input image
	std::vector<cv::Mat> input_images;

	std::vector<cv::Mat> projected_images;

	// output image (filter result)
	cv::Mat panoramic_result_image;

	std::vector<std::vector<cv::KeyPoint>> keypoints;

	std::vector<cv::Mat> descriptors;

	std::vector<std::vector<cv::DMatch>> matches;

	std::vector<float> mins;

	std::vector<float> maxs;

	int size;

	std::vector<cv::Mat> homographies;

	std::vector<double> distances;

};
