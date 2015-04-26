#ifndef CHAR_EXTRACTOR
#define CHAR_EXTRACTOR

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <stdio.h>
#include <string>
#include <iostream>

class CharacterExtractor
{
public:
  int preprocessImage(cv::Mat, cv::Mat&, cv::Mat&, bool);
  int findBoundingBoxes(cv::Mat, std::vector<cv::Rect>&);
  int findFullCharBoxes(std::vector<cv::Rect>&, std::vector<cv::Rect>&);
  int cropImage(cv::Mat, cv::Mat&, cv::Rect&);
private:
  bool rectContainRect(cv::Rect&, cv::Rect&);
};

#endif 