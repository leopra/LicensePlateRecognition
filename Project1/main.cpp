#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <filesystem>
#include "PossibleChar.h"

using namespace std::experimental::filesystem::v1;
using namespace std;
using namespace cv;

//FORWARD DECLARATION (METHODS THAT ARE IN THE OTHER CPP)
vector<Mat> licensePlateLoad();

void imgProcessing(Mat input);

string type2str(int type);

int characterProcessing(Mat input, int foldername, int DEBUG);

double LetterMatching(Mat img);
////////////////////////////////////////////////////////
const int MIN_PIXEL_WIDTH = 2;
const int MIN_PIXEL_HEIGHT = 8;

const double MIN_ASPECT_RATIO = 0.25;
const double MAX_ASPECT_RATIO = 1.0;

const int MIN_PIXEL_AREA = 80;

// constants for comparing two chars
const double MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;
const double MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;

const double MAX_CHANGE_IN_AREA = 0.5;

const double MAX_CHANGE_IN_WIDTH = 0.8;
const double MAX_CHANGE_IN_HEIGHT = 0.2;

const double MAX_ANGLE_BETWEEN_CHARS = 12.0;

// other constants
const int MIN_NUMBER_OF_MATCHING_CHARS = 3;

const int RESIZED_CHAR_IMAGE_WIDTH = 20;
const int RESIZED_CHAR_IMAGE_HEIGHT = 30;

const int MIN_CONTOUR_AREA = 100;
//////////////////////////////////////////////////////
//FIND MAX METHOD
int findMax(vector<int> vec) {
	int temp = 0;
	int index = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (temp < vec[i]) {
			temp = vec[i];
			index = i;
		}
		cout << temp << endl;
	}

	if (temp < 3)
		cout << "TARGA NON TROVATA" << endl;
	else
		cout << "TARGA TROVATA - CARATTERI SALVATI" << endl;

	return index;

}
//LOAD IMAGES
vector<Mat> licensePlate(String namefolder)
{
	vector<String> fn;
	vector<Mat> LicenseVector;
	glob(namefolder + "/*.jpg", fn, false);

	for (int i = 0; i < fn.size(); i++) {
		LicenseVector.push_back(imread(fn[i]));
		//cout << "done" << i << endl;
	}
	return LicenseVector;


}

//license plate image equalization
Mat equalizeHist(Mat in)
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

// use Pythagorean theorem to calculate distance between two chars
double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) {
	int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
	int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

//CHECK IF THE RECTANGLE FOUND IS OF THE RIGHT DIMENSIONS
bool verifySizesContours(RotatedRect candidate) {
	float error = 0.4; 
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 12 * aspect * 12; // minimum area
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
bool checkIfPossibleChar(PossibleChar &possibleChar) {
	// this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
	// note that we are not (yet) comparing the char to other chars to look for a group
	if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
		possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
		MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
		return(true);
	}
	else {
		return(false);
	}
}
std::vector<PossibleChar> findPossibleCharsInScene( Mat &imgThresh) {
	std::vector<PossibleChar> vectorOfPossibleChars;            // this will be the return value

	Mat imgContours(imgThresh.size(), CV_8UC3, Scalar(0,0,0));
	int intCountOfPossibleChars = 0;

	 Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector< Point> > contours;

	 findContours(imgThreshCopy, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);        // find all contours

	for ( int i = 0; i < contours.size(); i++) {                // for each contour

		 drawContours(imgContours, contours, i, Scalar(255, 255, 255));
		PossibleChar possibleChar(contours[i]);

		if (checkIfPossibleChar(possibleChar)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
			intCountOfPossibleChars++;                          // increment count of possible chars
			vectorOfPossibleChars.push_back(possibleChar);      // and add to vector of possible chars
		}
	}

	
		imshow("contours 1", imgContours);
		waitKey(0);
	

	return(vectorOfPossibleChars);
}

 Mat extractValue( Mat &imgOriginal) {
	 Mat imgHSV;
	std::vector< Mat> vectorOfHSVImages;
	 Mat imgValue;

	 cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

	 split(imgHSV, vectorOfHSVImages);

	imgValue = vectorOfHSVImages[2];

	return(imgValue);
}

std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars) {
	// the purpose of this function is, given a possible char and a big vector of possible chars,
	// find all chars in the big vector that are a match for the single possible char, and return those matching chars as a vector
	std::vector<PossibleChar> vectorOfMatchingChars;                // this will be the return value

	for (auto &possibleMatchingChar : vectorOfChars) {              // for each char in big vector

																	// if the char we attempting to find matches for is the exact same char as the char in the big vector we are currently checking
		if (possibleMatchingChar == possibleChar) {
			// then we should not include it in the vector of matches b/c that would end up double including the current char
			continue;           // so do not add to vector of matches and jump back to top of for loop
		}
		// compute stuff to see if chars are a match
		double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
		double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
		double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
		double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

		// check if chars match
		if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
			dblChangeInArea < MAX_CHANGE_IN_AREA &&
			dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
			dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {
			vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
		}
	}

	return(vectorOfMatchingChars);          // return result
}

 Mat maximizeContrast( Mat &imgGrayscale) {
	 Mat imgTopHat;
	 Mat imgBlackHat;
	 Mat imgGrayscalePlusTopHat;
	 Mat imgGrayscalePlusTopHatMinusBlackHat;

	 Mat structuringElement =  getStructuringElement(0,  Size(3, 3));

	 morphologyEx(imgGrayscale, imgTopHat, MORPH_TOPHAT, structuringElement);
	 morphologyEx(imgGrayscale, imgBlackHat, MORPH_BLACKHAT, structuringElement);

	imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
	imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

	return(imgGrayscalePlusTopHatMinusBlackHat);
}

std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar> &vectorOfPossibleChars) {
	// with this function, we start off with all the possible chars in one big vector
	// the purpose of this function is to re-arrange the one big vector of chars into a vector of vectors of matching chars,
	// note that chars that are not found to be in a group of matches do not need to be considered further
	std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;             // this will be the return value

	for (auto &possibleChar : vectorOfPossibleChars) {                  // for each possible char in the one big vector of chars

																		// find all chars in the big vector that match the current char
		std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

		vectorOfMatchingChars.push_back(possibleChar);          // also add the current char to current possible vector of matching chars

																// if current possible vector of matching chars is not long enough to constitute a possible plate
		if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
			continue;                       // jump back to the top of the for loop and try again with next char, note that it's not necessary
											// to save the vector in any way since it did not have enough chars to be a possible plate
		}
		// if we get here, the current vector passed test as a "group" or "cluster" of matching chars
		vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);            // so add to our vector of vectors of matching chars

																					// remove the current vector of matching chars from the big vector so we don't use those same chars twice,
																					// make sure to make a new big vector for this since we don't want to change the original big vector
		std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;

		for (auto &possChar : vectorOfPossibleChars) {
			if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
				vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
			}
		}
		// declare new vector of vectors of chars to get result from recursive call
		std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

		// recursive call
		recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);	// recursive call !!

		for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      // for each vector of matching chars found by recursive call
			vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // add to our original vector of vectors of matching chars
		}

		break;		// exit for loop
	}

	return(vectorOfVectorsOfMatchingChars);
}

vector<Mat> initialProcessing(Mat input, int foldername, int DEBUG) {

	int imagenumber = foldername;
	Mat imgGrayscale = extractValue(input);                           // extract value channel only from original image to get imgGrayscale

	 Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // maximize contrast with top hat and black hat

	 Mat imgBlurred;

	 Mat imgThresh;
	 GaussianBlur(imgMaxContrastGrayscale, imgBlurred, Size(5,5), 0);          // gaussian blur

	vector<Mat> possiblePlates;

	// call adaptive threshold to get imgThresh
	 adaptiveThreshold(imgBlurred, imgThresh, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9);
	
	if (DEBUG == 0) {
		imshow("threshold", imgThresh);
		waitKey(0);
	}
	
	vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThresh);
	Mat imgContours2 =  Mat(imgThresh.size(), CV_8UC3, Scalar(0, 0, 0));
	std::vector<std::vector< Point> > contours;

	for (auto &possibleChar : vectorOfPossibleCharsInScene) {
		contours.push_back(possibleChar.contour);
	}
	drawContours(imgContours2, contours, -1, Scalar(255, 255, 255));
	imshow("contorni scremati", imgContours2);
	waitKey(0);

	
	std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

	Mat imgContours3 =  Mat(imgThresh.size(), CV_8UC3, Scalar(0, 0, 0));

	 RNG rng;

	 for (int i = 0; i < vectorOfVectorsOfMatchingCharsInScene.size(); i++) {
	
		int intRandomBlue = rng.uniform(0, 256);
		int intRandomGreen = rng.uniform(0, 256);
		int intRandomRed = rng.uniform(0, 256);

		std::vector<std::vector< Point> > contours;

		/*for (size_t i = 0; i < vectorOfMatchingChars[i].size(); i++) {
			joined.insert(joined.end(), contours[i].begin(), contours[i].end());
		}*/
		//area to check if it is a license plate

		vector<Point> joined;
		vector<PossibleChar> temp = vectorOfVectorsOfMatchingCharsInScene[i];
		for (int i = 0; i < temp.size(); i++) {
			contours.push_back(temp[i].contour);
			joined.insert(joined.end(), contours[i].begin(), contours[i].end());

		}
		RotatedRect mr = minAreaRect(joined);
		drawContours(imgContours3, contours, -1,  Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));


		if (DEBUG == 0) {
			imshow("FINAL CONTOURS", imgContours3);
			waitKey(0);
		}
		int patchnumber = 0;
		//if (verifySizesContours(mr)) {

		//	RotatedRect minRect;
		//	// rotated rectangle drawing 
		//	Point2f rect_points[4]; minRect.points(rect_points);
		//	//for (int j = 0; j < 4; j++)
		//	//	line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);

		//	//Get rotation matrix
		//	float r = (float)minRect.size.width / (float)minRect.size.height;
		//	float angle = minRect.angle;
		//	if (r < 1)
		//		angle = 90 + angle;
		//	Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

		//	//Create and rotate image
		//	Mat img_rotated;
		//	warpAffine(input, img_rotated, rotmat, input.size(), INTER_CUBIC);

		//	//Crop image
		//	Size rect_size = minRect.size;
		//	if (r < 1)
		//		swap(rect_size.width, rect_size.height);
		//	Mat img_crop;
		//	getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

		//	Mat resultResized;
		//	resultResized.create(33, 144, CV_8UC3);
		//	resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
		//	//Equalize croped image
		//	Mat grayResult;
		//	cvtColor(resultResized, grayResult, COLOR_BGR2GRAY);
		//	blur(grayResult, grayResult, Size(3, 3));
		//	grayResult = equalizeHist(grayResult);

		//	if (DEBUG == 1) {
		//		imshow("result", grayResult);
		//		waitKey(0);
		//	}
		//	possiblePlates.push_back(grayResult);

		//	string filename = string("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\license-plates\\") + std::to_string(imagenumber) + "-" + std::to_string(patchnumber) + "-possiblelicense.jpg";
		//	cout << filename << endl;
		//	patchnumber++;
		//	try {
		//		imwrite(filename, grayResult);
		//		cout << "done" << endl;

		//	}
		//	catch (runtime_error& ex) {
		//		fprintf(stderr, "exception saving image: %s\n", ex.what());
		//		return possiblePlates;
		//	}
		//}
		patchnumber++;
	}
		
	return possiblePlates;
}



int main() {
	srand(time(NULL));

	//possible parameters to change
	/////////////////////////////// 
	int DEBUG = 0;

	int IMAGE_NUMBER = rand () % 6;

	String pathdir = "data";

	///////////////////////////////

	vector<Mat> input_images = licensePlate(pathdir);

	vector<Mat> Licenses = initialProcessing(input_images[IMAGE_NUMBER], IMAGE_NUMBER, DEBUG);
	if (Licenses.empty()) {
		imshow("FALLITO", input_images[IMAGE_NUMBER]);
		waitKey(0);
		return 0;
	}

	vector<int> char_found;
	for (int i = 0; i < Licenses.size(); i++) {

		char_found.push_back(characterProcessing(Licenses[i], i, DEBUG));

	}
	int index = findMax(char_found);
	cout << "END" << endl;
	imshow("END", Licenses[index]);
	
	waitKey(0);
	
	return 0;
}