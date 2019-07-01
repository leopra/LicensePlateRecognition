#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <filesystem>
#include "Char.h"

//using namespace experimental::filesystem::v1;
using namespace std;
using namespace cv;

//FORWARD DECLARATION (METHODS THAT ARE IN THE OTHER CPP)
vector<Mat> licensePlateLoad();

void imgProcessing(Mat input);

string type2str(int type);

int characterProcessing(Mat input, int foldername, int DEBUG);

double LetterMatching(Mat img);
////////////////////////////////////////////////////////

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

// use Pitagora to calculate distance between two chars
double distanceBetweenChars(Char firstChar, Char secondChar) {
	int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
	int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

//CHECK IF THE RECTANGLE FOUND IS OF THE RIGHT DIMENSIONS
bool verifySizesContours(RotatedRect candidate) {
	float error = 0.2; 
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 1500; // minimum area
	int max = 6000; // maximum area
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

// check if sizes correspond
bool checkIfPossibleChar(Char &possibleChar) {
	if (possibleChar.boundingRect.area() > 80 &&
		possibleChar.boundingRect.width > 2 && possibleChar.boundingRect.height > 8 &&
		0.25 < possibleChar.Ratio && possibleChar.Ratio < 1) {
		return true;
	}
	else {
		return false;
	}
}

// search for chars
vector<Char> searchAlltheChars( Mat imgThresh) {
	vector<Char> vectorOfPossibleChars;            // this will be the return value

	Mat imgContours(imgThresh.size(), CV_8UC3, Scalar(0,0,0));
	int intCountOfPossibleChars = 0;

	 Mat imgThreshCopy = imgThresh.clone();

	vector<vector< Point> > contours;

	 findContours(imgThreshCopy, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);        // find all contours

	for ( int i = 0; i < contours.size(); i++) {                // for each contour

		 drawContours(imgContours, contours, i, Scalar(255, 255, 255));
		Char possibleChar(contours[i]);

		if (checkIfPossibleChar(possibleChar)) {               
			intCountOfPossibleChars++;                          
			vectorOfPossibleChars.push_back(possibleChar);    
		}
	}

	
		imshow("contours 1", imgContours);
		waitKey(0);
	

	return vectorOfPossibleChars;
}

double angleof2Chars(Char firstChar, Char secondChar) {
	double modX = abs(firstChar.intCenterX - secondChar.intCenterX);
	double modY = abs(firstChar.intCenterY - secondChar.intCenterY);

	double radAngle = atan(modY / modX);

	return radAngle;
}

//search one char for a match in all chars
vector<Char> findVectorOfMatchingChars(Char possibleChar, vector<Char> vectorOfChars) {
	// find all chars in the big vector that are a match for the single possible char
	vector<Char> vectorOfMatchingChars;                
	for (int i = 0; i < vectorOfChars.size(); i++) {
		Char possibleMatchingChar = vectorOfChars[i];
         // for each char in big vector

			
		// I calculate the difference between parameters of Char Class
		double diffAngle = angleof2Chars(possibleChar, possibleMatchingChar);
		double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
		double DiffArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
		double DiffWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
		double DiffAlt = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

		// then I compare it with some thresholds
		if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * 5) &&
			DiffArea < 0.5 &&
			DiffWidth < 0.8 &&
			DiffAlt < 0.2 
			&& diffAngle < 38
			) {
			vectorOfMatchingChars.push_back(possibleMatchingChar);     
		}
	}

	return(vectorOfMatchingChars); 
}

// using top hat black hat operator to get best contrast
//enhance dark objects of interest in a bright background
 Mat enhanceBlackOnWhite( Mat imgGrayscale) {
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

 bool isNotChecked(Char a, vector<Char> b) {
	 for (int i = 0; i < b.size(); i++) {
		 if (a == b[i])
			 return false;
		 else
			 return true;
	 }
 }

vector<vector<Char> > findVECofVECofChars(vector<Char> vectorOfPossibleChars) {
	
	vector<vector<Char> > vectorOfVectorsOfChars;  
	vector<Char> alreadyChecked;
	vector<Char> vectorOfMatchingChars;

	for (int i = 0; i < vectorOfPossibleChars.size(); i++) {
		int kkk = 0;
		Char tempChar = vectorOfPossibleChars[i];
		if (isNotChecked(tempChar, alreadyChecked)) {
			vectorOfMatchingChars = findVectorOfMatchingChars(tempChar, vectorOfPossibleChars);
			for (int i = 0; i < vectorOfMatchingChars.size(); i++) {
				if (!isNotChecked(vectorOfMatchingChars[i], alreadyChecked)) {
					kkk = 1;
				}
				if (kkk == 0) {
					alreadyChecked.push_back(tempChar);
				}
			}
			if (kkk == 0 && vectorOfMatchingChars.size() != 0)
				vectorOfVectorsOfChars.push_back(vectorOfMatchingChars);
				

		}//add the current char to current char

	}

	return vectorOfVectorsOfChars;
}

vector<Mat> initialProcessing(Mat input, int foldername, int DEBUG) {

	int imagenumber = foldername;
	Mat imgGrayscale;
	cvtColor(input, imgGrayscale, COLOR_BGR2GRAY);                          //  get imgGrayscale

	Mat imgMaxContrastGrayscale = enhanceBlackOnWhite(imgGrayscale);       // maximize contrast with top hat and black hat

	Mat imgBlurred;

	Mat imgThresh;
	GaussianBlur(imgMaxContrastGrayscale, imgBlurred, Size(5, 5), 0);          // gaussian blur

	vector<Mat> possiblePlates;

	// call adaptive threshold to get imgThresh
	adaptiveThreshold(imgBlurred, imgThresh, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9);

	if (DEBUG == 1) {
		imshow("threshold", imgThresh);
		waitKey(0);
	}

	vector<Char> vectorOfPossibleCharsInScene = searchAlltheChars(imgThresh);
	Mat imgContours2 = Mat(imgThresh.size(), CV_8UC3, Scalar(0, 0, 0));
	vector<vector< Point> > contours;

	for (auto &possibleChar : vectorOfPossibleCharsInScene) {
		contours.push_back(possibleChar.contour);
	}
	drawContours(imgContours2, contours, -1, Scalar(255, 255, 255));
	imshow("contorni scremati", imgContours2);
	waitKey(0);


	for (int i = 0; i < vectorOfPossibleCharsInScene.size(); i++) {
		vector<vector<Char> > vectorOfVectorsOfMatchingCharsInScene = findVECofVECofChars(vectorOfPossibleCharsInScene);

		Mat imgContours3 = Mat(imgThresh.size(), CV_8UC3, Scalar(0, 0, 0));

		int patchnumber = 0;

		for (int i = 0; i < vectorOfVectorsOfMatchingCharsInScene.size(); i++) {

			int intRandomBlue = rand() % 250;
			int intRandomGreen = rand() % 250;
			int intRandomRed = rand() % 250;

			vector<vector< Point> > contours;
			RotatedRect mr;
			vector<Point> joined;
			vector<Char> temp;

			temp = vectorOfVectorsOfMatchingCharsInScene[i];
			for (int i = 0; i < temp.size(); i++) {
				contours.push_back(temp[i].contour);
				joined.insert(joined.end(), contours[i].begin(), contours[i].end());

			}
			mr = minAreaRect(joined);
			drawContours(imgContours3, contours, -1, Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));


			if (DEBUG == 1) {
				imshow("FINAL CONTOURS", imgContours3);
				waitKey(0);
			}
			Point2f rect_points[4]; mr.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(imgContours3, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);

			if (DEBUG == 1) {
				imshow("FINAL CONTOURS", imgContours3);
				waitKey(0);
			}

			if (verifySizesContours(mr) && temp.size() > 4) {
				//Get rotation matrix
				float r = (float)mr.size.width / (float)mr.size.height;
				float angle = mr.angle;
				if (r < 1)
					angle = 90 + angle;
				Mat rotmat = getRotationMatrix2D(mr.center, angle, 1);

				//Create and rotate image
				Mat img_rotated;
				warpAffine(input, img_rotated, rotmat, input.size(), INTER_CUBIC);

				//Crop image
				Size rect_size = mr.size;
				if (r < 1)
					swap(rect_size.width, rect_size.height);
				Mat img_crop;
				getRectSubPix(img_rotated, rect_size, mr.center, img_crop);

				Mat resultResized;
				resultResized.create(33, 144, CV_8UC3);
				resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
				//Equalize croped image
				Mat grayResult;
				cvtColor(resultResized, grayResult, COLOR_BGR2GRAY);
				blur(grayResult, grayResult, Size(3, 3));
				grayResult = equalizeHist(grayResult);

				if (DEBUG == 1) {
					imshow("result", grayResult);
					waitKey(0);
				}
				possiblePlates.push_back(grayResult);

				string filename = string("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\license-plates\\") + to_string(imagenumber) + "-" + to_string(patchnumber) + "-possiblelicense.jpg";
				cout << filename << endl;
				patchnumber++;
				try {
					imwrite(filename, grayResult);
					cout << "done" << endl;

				}
				catch (runtime_error& ex) {
					fprintf(stderr, "exception saving image: %s\n", ex.what());
					cout << "erroraccio" << endl;
					return possiblePlates;
				}
			}
			//patchnumber++;
		}

		return possiblePlates;
	}

}

	int main() {
		srand(time(NULL));

		//possible parameters to change
		/////////////////////////////// 
		int DEBUG = 1;

		int IMAGE_NUMBER = 4;

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
		cout << "END" << endl;
		imshow("END", Licenses[0]);

		waitKey(0);

		return 0;
	}
