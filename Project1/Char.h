

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class Char {
public:

	Char(std::vector<cv::Point> contours);
	std::vector<cv::Point> contour;

	cv::Rect boundingRect;

	int intCenterX;
	int intCenterY;

	double dblDiagonalSize;
	double Ratio;

	//need to overload == operator
	bool operator == (Char otherPossibleChar) const {
		if (this->contour == otherPossibleChar.contour) return true;
		else return false;
	}
};

