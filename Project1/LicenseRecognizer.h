#pragma once
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class LicenseRecognizer {

	// Methods

public:

	// constructor do nothing
	LicenseRecognizer();

	//real constructor
	vector<Mat> licensePlate(String namefolder);

	//equalizehist
	Mat equalizeHista(Mat in);

	//verifysizes1
	bool verifySizesContours(RotatedRect candidate);

	//verifysizes2
	bool verifySizesFloodFill(RotatedRect candidate);

	//initial processing
	void initialProcessing(Mat input, int foldername);



protected:

	// input image
	std::vector<cv::Mat> input_images;



	

};