
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/ml.hpp>


using namespace std;
using namespace cv;
using namespace ml;

//variables to be declared local
int numCharacters = 26 + 10;
ANN_MLP ann;
bool trained;



void cPlusMachineLearning(Mat input) {
	Mat img_threshold;
	threshold(input, img_threshold, 60, 255, THRESH_BINARY_INV);
	imshow("Threshold plate", img_threshold);
	Mat img_contours;
	img_threshold.copyTo(img_contours);
	//Find contours of possibles characters
	vector< vector< Point> > contours;
	findContours(img_contours,
		contours, // a vector of contours
		RETR_EXTERNAL, // retrieve the external contours
		CHAIN_APPROX_NONE); // all pixels of each contour

}


bool verifySizes(Mat r)
{
	//Char sizes 45x77
	float aspect = 45.0f / 77.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;
	//We have a different aspect ratio for number 1, and it can be
	//~0.2
	float minAspect = 0.2;
	float maxAspect = aspect + aspect * error;
	//area of pixels
	float area = countNonZero(r);
	//bb area
	float bbArea = r.cols*r.rows;
	//% of pixel in area
	float percPixels = area / bbArea;
	if (percPixels < 0.8 && charAspect > minAspect && charAspect <
		maxAspect && r.rows >= minHeight && r.rows < maxHeight)
		return true;
	else
		return false;
}

void train(Mat TrainData, Mat classes, int nlayers)
{	
	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = TrainData.cols;
	layerSizes.at<int>(1) = nlayers;
	layerSizes.at<int>(2) = numCharacters;
	ann.create(layerSizes, ANN_MLP::SIGMOID_SYM, 1, 1); //ann is global class variable
		//Prepare trainClasses
		//Create a mat with n trained data by m classes
	Mat trainClasses;
	trainClasses.create(TrainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++)
	{
		for (int k = 0; k < trainClasses.cols; k++)
		{
	
				//If class of data i is same than a k class
				if (k == classes.at<int>(i))
					trainClasses.at<float>(i, k) = 1;
				else
					trainClasses.at<float>(i, k) = 0;
		}
	}
	Mat weights(1, TrainData.rows, CV_32FC1, Scalar::all(1));
	//Learn classifier
	ann.train(TrainData, trainClasses, weights);
	trained = true;
}

int classify(Mat f)
{
	int result = -1;
	Mat output(1, numCharacters, CV_32FC1);
	ann.predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
	//We need to know where in output is the max val, the x (cols) is
	//the class.
	return maxLoc.x;
}
