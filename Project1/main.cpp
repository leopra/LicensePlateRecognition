#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>



using namespace std;
using namespace cv;

vector<Mat> licensePlate(String namefolder)
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

Mat histeq(Mat in)
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
//We make basic validations about the regions detected based on its area and aspect ratio.We only
//consider that a region can be a plate if the aspect ratio is approximately 520 / 110 = 4.727272 (plate
//	width divided by plate height) with an error margin of 40 percent and an area based on a minimum of
//	15 pixels and maximum of 125 pixels for the height of the plate.These values are calculated depending
//	on the image sizes and camera position :
bool verifySizes(RotatedRect candidate) {
	float error = 0.4;
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

void initialProcessing(Mat input, String name) {

	// conversion to gray and blur
	Mat img_gray;
	cvtColor(input, img_gray, COLOR_BGR2GRAY);
	blur(img_gray, img_gray, Size(5, 5));

	//Find vertical lines. Car plates have high density of vertical lines
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);

	/*imshow("test", img_sobel);
	waitKey(0)*/;
	/*After a Sobel filter, we apply a threshold filter to obtain a binary image with a threshold value obtained
		through Otsu's method. Otsu's algorithm needs an 8 - bit input image and Otsu's method automatically
		determines the optimal threshold value :*/
		//threshold image
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 230, 255, THRESH_OTSU + THRESH_BINARY);

	/*imshow("test", img_threshold);
	waitKey(0);*/
	/*By applying a close morphological operation, we can remove blank spaces between each vertical edge
		line, and connect all regions that have a high number of edges.In this step we have the possible regions
		that can contain plates.*/
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);

	/*imshow("dilate", img_threshold);
	waitKey(0);*/
	//Find contours of possibles plates
	vector< vector< Point> > contours;
	findContours(img_threshold,
		contours, // a vector of contours
		RETR_EXTERNAL, // retrieve the external contours
		CHAIN_APPROX_NONE); // all pixels of each contour

	//Start to iterate to each contour found
	vector<vector<Point> >::iterator itc = contours.begin();
	vector<RotatedRect> rects;

	cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(255, 0, 0), // in blue
		1); // with a thickness of 1

	imshow("all contours", result);
	waitKey(0);

	int color = 0;
	//Remove patch that has no inside limits of aspect ratio and area.
	while (itc != contours.end()) {
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr)) {
			itc = contours.erase(itc);
			//cv::drawContours(result, contours,
			//	-1, // draw all contours
			//	cv::Scalar(color + 5 , color + 2, 0), // in blue
			//	1); // with a thickness of 1
			//imshow("all contours", result);
			//waitKey(0);
		}
		else {
			++itc;
			rects.push_back(mr);
		}
		//result = input;
	}

	// Draw blue contours on a white image

	input.copyTo(result);
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(255, 0, 0), // in blue
		1); // with a thickness of 1
	/*imshow("drawcontour", result);
	waitKey(0);*/

	for (int i = 0; i < rects.size(); i++) {

		//For better rect cropping for each posible box
		//Make floodfill algorithm because the plate has white background
		//And then we can retrieve more clearly the contour box
		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		//get the min size between width and height
		float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5;
		//initialize rand and get 5 points around center for floodfill algorithm
		srand(time(NULL));
		//Initialize floodfill parameters and variables
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++) {
			Point seed;
			seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
			circle(result, seed, 1, Scalar(0, 255, 255), -1);
			/*imshow("test", result);
			waitKey(0);*/
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
			 
		}
/*
		
		imshow("MASK", mask);
		waitKey(0);*/

		//Check new floodfill mask match for a correct patch.
		//Get all points detected for get Minimal rotated Rect
		vector<Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itMask != end; ++itMask)
			if (*itMask == 255)
				pointsInterest.push_back(itMask.pos());

		RotatedRect minRect = minAreaRect(pointsInterest);

		cv::Scalar color = cv::Scalar(122.0, 200.0, 255.0); // white

		// We take the edges that OpenCV calculated for us
		cv::Point2f vertices2f[4];
		minRect.points(vertices2f);

		// Convert them so we can use them in a fillConvexPoly
		cv::Point vertices[4];
		for (int i = 0; i < 4; ++i) {
			vertices[i] = vertices2f[i];
		}

		// Now we can fill the rotated rectangle with our specified color
		cv::fillConvexPoly(input,
			vertices,
			4,
			color);
		/*imshow("rectangle", input);
		waitKey(0);*/


		if (verifySizes(minRect)) {
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
			grayResult = histeq(grayResult);
			imshow("result", grayResult);
			waitKey(0);

			String filename = name;

			stringstream ss(stringstream::in | stringstream::out);
			ss << "tmp/" << filename << "_" << i << ".jpg";
			imwrite(ss.str(), grayResult);
			/*if (saveRegions) {
				stringstream ss(stringstream::in | stringstream::out);
				ss << "tmp/" << filename << "_" << i << ".jpg";
				imwrite(ss.str(), grayResult);
			}*/
			//output.push_back(Plate(grayResult, minRect.boundingRect()));
		}
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

	/*imshow("test", edges);
	waitKey(0);*/
}

int main()
{	//this is the name of the folder with the photos
	String pathdir = "data";
	vector<Mat> input_images = licensePlate(pathdir);
	for (int i = 0; i < input_images.size(); i++) {
		String name = "gray" + i;
		initialProcessing(input_images[5], name);
	}
	cout << "dioacn" << endl;
	//imgProcessing(input_images[0]);
	waitKey(0);
	return 0;
}