#include "Char.h"



///////////////////////////////////////////////////////////////////////////////////////////////////
Char::Char(std::vector<cv::Point> _contour) {
	contour = _contour;

	boundingRect = cv::boundingRect(contour);

	intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
	intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

	dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

	Ratio = (float)boundingRect.width / (float)boundingRect.height;
}