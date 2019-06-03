//#include "LicensePlate.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "opencv2/features2d.hpp"


#include <limits>

using namespace std;
using namespace cv;

void LicensePlate(String namefolder)
{
	vector<Mat> input_images;

	vector<String> fn;
	vector<Mat> panoramaImgVector;
	glob(namefolder + "/*.bmp", fn, false);

	for (int i = 0; i < fn.size(); i++) {
		panoramaImgVector.push_back(imread(fn[i]));
		cout << "done" << i << endl;
	}
	input_images = panoramaImgVector;


}