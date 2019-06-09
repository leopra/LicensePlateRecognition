#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

void splitLetters(int x, int y) {

}

vector<Mat> licensePlateLoad()
{
	vector<String> fn;
	vector<Mat> LicenseVector;
	glob("license-plates/*.jpg", fn, false);

	for (int i = 0; i < fn.size(); i++) {
		LicenseVector.push_back(imread(fn[i]));
		cout << "done" << i << endl;
	}
	return LicenseVector;


}


//see type of matrix for debugging
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//verify the correct size of the char found
bool verifySizesChar(Rect candidate) {
	float error = 0.4;
	//character size = 27/14 
	const float aspect = 1.92;
	//Set a min and max area. All other patches are discarded
	int min = 200; // minimum area
	int max = 500; // maximum area
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;
	int area = candidate.height * candidate.width;
	cout << area << endl;
	float r = (float)candidate.width / (float)candidate.height;
	cout << r << endl;
	if (r < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	}
	else {
		return true;
	}
}

void characterProcessing(Mat input) {

	Mat img_threshold;
	cvtColor(input, img_threshold, COLOR_BGR2GRAY);

	threshold(img_threshold, img_threshold, 60, 255, THRESH_BINARY_INV);
	imshow("Threshold plate", img_threshold);
	cout << type2str(img_threshold.type()) << endl;

	Mat img_contours;
	img_threshold.copyTo(img_contours);
	Mat dst;
	vector< vector< Point> > contours;
	findContours(img_threshold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(255, 0, 0), // in blue
		1); // with a thickness of 1
	cout << "diocan" << endl;
	/*imshow("Segmented Chars", result);
	waitKey(0);*/

	//iterate between each contour
	vector<vector<Point> >::iterator itc = contours.begin();
	while (itc != contours.end()) {

		//Create bounding rect of object
		Rect mr = boundingRect(Mat(*itc));
		rectangle(result, mr, Scalar(0, 255, 0));
		//Crop image
		//Rect outputcut = Rect(mr.x, mr.y, mr.width + 10, 33);
		Rect outputcut = mr;

		Mat outputchar = input(outputcut);
		threshold(outputchar, outputchar, 60, 255, THRESH_BINARY_INV);

		///Resize che char to a 28x28 pixel image to match EMNIST DIMENSIONS
		Mat resizedchar;
		resize(outputchar, resizedchar, Size(28, 28));
		/*imshow("output cut", outputchar);
		waitKey(0);*/
		Mat auxRoi(img_threshold, mr);
		int imagereference = 0;
		if (verifySizesChar(mr)) {
			//auxRoi = preprocessChar(auxRoi);
			//output.push_back(CharSegment(auxRoi, mr));
			rectangle(result, mr, Scalar(0, 125, 255));
			/*imshow("SEgmented Chars", result);
			waitKey(0);*/
			int v1 = rand() % 1000000;
			string filename = string("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\char-found\\") + std::to_string(v1) + "char.jpg";
			cout << filename << endl;
			try {
				imwrite(filename, resizedchar);
				cout << "done" << endl;

			}
			catch (runtime_error& ex) {
				fprintf(stderr, "exception saving image: %s\n", ex.what());
				return;
			}
		}
		imagereference++;
		++itc;
	}
}
//useless method 
void imgProcessing(Mat input) {
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);
	const int MEDIAN_BLUR_FILTER_SIZE = 7;
	medianBlur(gray, gray, MEDIAN_BLUR_FILTER_SIZE);
	Mat edges;
	const int LAPLACIAN_FILTER_SIZE = 5;
	Laplacian(gray, edges, CV_8U, LAPLACIAN_FILTER_SIZE);

	Mat mask;
	const int EDGES_THRESHOLD = 80;
	threshold(edges, mask, EDGES_THRESHOLD, 255, THRESH_BINARY_INV);

	imshow("test", mask);
	waitKey(0);
}



