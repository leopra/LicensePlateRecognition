#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <filesystem>

using namespace std;
using namespace cv;

double LetterMatching(Mat img) {
	
	
	vector<Mat> templates;
	vector<String> fn;
	glob("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\charactersfortemplate", fn, false);
	for (int i = 0; i < fn.size(); i++) {
		templates.push_back(imread(fn[i]));
	}
	int match_method = 0;
	Mat result;
	Mat templ = img;
	imshow("image_window", templ);
	waitKey(0);
	/// Source image to display
	Mat img_display;
	img.copyTo(img_display);

	/// Create the result matrix
	int result_cols = img.cols - templ.cols + 1; // this equals as one ( the images have the same dimension
	int result_rows = img.rows - templ.rows + 1; // this equals as one ( the images have the same dimension

	result.create(1, 1, CV_32F );

	/// Do the Matching and Normalize
	double temp = -1;
	String lettername;
	for (int i = 0; i < fn.size(); i++) {
		matchTemplate(img, templates[i], result, match_method);
		//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;
		//cout << result << endl;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
		if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
		{
			matchLoc = minLoc;
		}
		else
		{
			matchLoc = maxLoc;
		}

		if (temp < result.at<int>(0,0)) {
			temp = result.at<int>(0,0);
			lettername = fn[i];
			
		}
		/// Show me what you got
		rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
		rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);

		/*imshow("image_window", img_display);
		waitKey(0);*/
/*
		imshow("result_window", result);
		waitKey(0);*/

	}

	cout << "FINAL PREDICTION: " << lettername << endl;
	return temp;
}


//main

//String filename = "E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\char-found\\100\\";
//vector<String> fn;
//glob(filename, fn, false);
//for (int i = 0; i < fn.size(); i++) {
//	cout << "//////////////////////////////" << endl;
//
//	cout << fn[i] << endl;
//	cout << "//////////////////////////////" << endl;
//
//	Mat imagess = imread(fn[i]);
//	double s = LetterMatching(imagess);
//}
//
//waitKey(0);