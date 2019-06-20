#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

class LetterSplitting {

	public: void characterProcessing(Mat input, int foldername);
			vector<Mat> licensePlateLoad();
			string type2str(int type);
			bool verifySizesChar(Rect candidate);
			void imgProcessing(Mat input);

private:
	vector<Mat> licensePlatess
};