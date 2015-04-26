#ifndef NEURAL_NET
#define NEURAL_NET 

#include <vector>
#include <stdio.h>
#include <string>

#include <math.h>

#include "mat.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"

class NeuralNet
{
public:
  bool loadNN(std::string);
  char classify(cv::Mat);
private:
  std::vector<cv::Mat> stackW, stackB;
  cv::Mat softmaxTheta;
  int inputSize, hiddenSize, numLayers, numClasses;

  bool netLoaded = false;

  char lookupTable[63] =
  {
    '?', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z'
  };

  mxArray* extractElement(mxArray*, std::string);
  bool buildNetwork(mxArray*, mxArray*);
  char lookup(int);
  cv::Mat sigmoid(cv::Mat);
  bool reshape(cv::Mat&);
};

#endif