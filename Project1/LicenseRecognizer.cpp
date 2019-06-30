#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <filesystem>


#include "LicenseRecognizer.h"


using namespace std::experimental::filesystem::v1;
using namespace std;
using namespace cv;



LicenseRecognizer::LicenseRecognizer()
{
}

//LOAD IMAGES
vector<Mat> LicenseRecognizer::licensePlate(String namefolder)
{
	vector<String> fn;
	vector<Mat> LicenseVector;
	glob(namefolder + "/*.jpg", fn, false);

	for (int i = 0; i < fn.size(); i++) {
		LicenseVector.push_back(imread(fn[i]));
		cout << "done" << i << endl;
	}
	return LicenseVector;


}

//license plate image equalization
Mat LicenseRecognizer::equalizeHista(Mat in)
{
	Mat out(in.size(), in.type());
	if (in.channels() == 3) {
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(in, hsv, COLOR_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, COLOR_HSV2BGR);
	}
	else if (in.channels() == 1) {
		equalizeHist(in, out);
	}

	return out;

}

//CHECK IF THE RECTANGLE FOUND IS OF THE RIGHT DIMENSIONS
bool LicenseRecognizer::verifySizesContours(RotatedRect candidate) {
	float error = 0.5;
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 15 * aspect * 15; // minimum area
	int max = 125 * aspect * 125; // maximum area
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;
	int area = candidate.size.height * candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	}
	else {
		return true;
	}
}

//CHECK IF THE RECTANGLE FOUND IS OF THE RIGHT DIMENSIONS
bool LicenseRecognizer::verifySizesFloodFill(RotatedRect candidate) {
	float error = 0.6;
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 10 * aspect * 10; // minimum area
	int max = 125 * aspect * 125; // maximum area
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;
	int area = candidate.size.height * candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	}
	else {
		return true;
	}
}

void LicenseRecognizer::initialProcessing(Mat input, int foldername) {

	//create a folder for che carachters to be saved
	String namepath = string("char-found/" + to_string(foldername));
	cout << namepath << endl;
	path myRoot(namepath);
	create_directory(myRoot);
	int imagenumber = foldername;
	cout << "what" << endl;
	// conversion to gray and blur
	Mat img_gray;
	cvtColor(input, img_gray, COLOR_BGR2GRAY);
	blur(img_gray, img_gray, Size(5, 5));

	//Find vertical lines. Car plates have high density of vertical lines
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);

	//After a Sobel filter, we apply a threshold filter to obtain a binary image with a threshold value obtained

	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU + THRESH_BINARY);

	//imshow("threshold", img_threshold);
	//waitKey(0);

	//By applying a close morphological operation, we can remove blank spaces between each vertical edge

	Mat element = getStructuringElement(MORPH_RECT, Size(22, 3)); ///added some pixel here 17->22
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
	//cout << LetterSplitting::type2str(img_threshold.type()) << endl;
	//Find contours of possibles plates
	vector< vector< Point> > contours;
	findContours(img_threshold,
		contours, // a vector of contours
		RETR_EXTERNAL, // retrieve the external contours
		CHAIN_APPROX_NONE); // all pixels of each contour

	///for debugging
	cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(255, 0, 0), // in blue
		1); // with a thickness of 1
	imshow("CONTOURS INITIAL", result);
	waitKey(0);

	//Start to iterate to each contour found
	vector<vector<Point> >::iterator itc = contours.begin();
	vector<RotatedRect> rects;
	//Remove patch that has no inside limits of aspect ratio and area.
	while (itc != contours.end()) {
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizesContours(mr)) {
			itc = contours.erase(itc);
		}
		else {
			++itc;
			rects.push_back(mr);
		}
	}

	// Draw blue contours on a white image
	//cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(255, 0, 0), // in blue
		1); // with a thickness of 1
	imshow("CONTOURS", result);
	waitKey(0);


	//USING FLOODFILL TO SAVE LESS WRONG LICENSE PATCHES
	int patchnumber = 0;

	for (int i = 0; i < rects.size(); i++) {


		//circle(result, rects[i].center, 10, Scalar(0, 255, 0), -1);

		float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5;
		//initialize rand and get 5 points around center for floodfill algorithm
		//Initialize floodfill parameters and variables
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 5;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++) {
			Point seed;
			seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
			//	circle(result, seed, 1, Scalar(0, 255, 255), -1);
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
		}

		imshow("MASK", mask);
		waitKey(0);

		//Check again if the rectangle has the right dimensions
		vector<Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itMask != end; ++itMask)
			if (*itMask == 255) //takes the points that the mask rendered white
				pointsInterest.push_back(itMask.pos());

		RotatedRect minRect = minAreaRect(pointsInterest);

		//RotatedRect minRect = minAreaRect(contours[i]);


	//CHECK IF RECTANGLE AFTER FLOODFILL IS RIGHT AND THEN CUT THE LICENSE PLATE AND STORE IT IN THE LICENSE PALTES FOLDER

		if (verifySizesFloodFill(minRect)) {
			// rotated rectangle drawing 
			Point2f rect_points[4]; minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);

			//Get rotation matrix
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			if (r < 1)
				angle = 90 + angle;
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

			//Create and rotate image
			Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(), INTER_CUBIC);

			//Crop image
			Size rect_size = minRect.size;
			if (r < 1)
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
			//Equalize croped image
			Mat grayResult;
			cvtColor(resultResized, grayResult, COLOR_BGR2GRAY);
			blur(grayResult, grayResult, Size(3, 3));
			grayResult = equalizeHista(grayResult);
			imshow("result", grayResult);
			waitKey(0);
			int v1 = rand() % 10000;
			string filename = string("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\license-plates\\") + std::to_string(imagenumber) + "-" + std::to_string(patchnumber) + "-possiblelicense.jpg";
			cout << filename << endl;
			patchnumber++;
			try {
				imwrite(filename, grayResult);
				cout << "done" << endl;

			}
			catch (runtime_error& ex) {
				fprintf(stderr, "exception saving image: %s\n", ex.what());
				return;
			}
		}
	}
}