#include "NeuralNet.h"

/*
 * A function for extracting elements from MATLAB structs
 * 
 * Parameters:
 *   structure -- an mxArray corresponding to a MATLAB struct
 *   fieldname -- a string containg the name of the field to extract from the struct
 *
 * Return Value:
 *   An mxArray containing the requested field OR null if it coulf not be found
 */
mxArray* NeuralNet::extractElement(mxArray* structure, std::string fieldname)
{
  int numel = mxGetNumberOfElements(structure);
  int i;
  for (i = 0; i < numel; ++i)
  {
    mxArray* extracted = mxGetField(structure, i, fieldname.c_str());
    if (extracted != NULL) return extracted;
  }
  return NULL;
}

/*
 * A function for taking a vector containing all of the weights for the network
 *   and a struct describing the network's structure and recreating the weight
 *   matrices of the neural network (stored into stackW and stackB)
 *
 * Paramters:
 *   field_theta -- an mxArray contating the vector of all weights
 *   field_theta -- an mxArray contating a struct describing the netowrk structure
 *
 * Return Value:
 *   boolean according to the success of the function
 */
bool NeuralNet::buildNetwork(mxArray* field_theta, mxArray* field_netconfig)
{
  /*
   * Read the requisite values from the passed in mxArrays
   */
  double* theta = mxGetPr(field_theta);
  mxArray* depth = extractElement(field_netconfig, "layersizes");
  if (depth == NULL)
  {
    printf("The netconfig field is improperly formatted\n");
    return false;
  }
  numLayers = mxGetNumberOfElements(depth);

  /*
   * Extract the softmax weights
   */
  double smax[hiddenSize*numClasses];
  std::memcpy(smax, theta, hiddenSize*numClasses*sizeof(double));
  softmaxTheta = cv::Mat(hiddenSize, numClasses, CV_64F, smax);
  softmaxTheta = softmaxTheta.t();

  /*
   * Extract the weights for all autoencoder layers
   */
  int prevLayerSize = inputSize, layerSize = 0;
  int index = hiddenSize*numClasses;
  int i, j, k, wLength, bLength;
  cv::Mat weight, bias;
  for (i = 0; i < numLayers; ++i)
  {
    mxArray* field_layerSize = mxGetCell(depth, i);
    layerSize = (int) *(mxGetPr(field_layerSize));

    wLength = layerSize * prevLayerSize;
    bLength = layerSize;

    double w[wLength];
    double b[bLength];
    std::memcpy(w, theta + index, wLength*sizeof(double));
    weight = cv::Mat(prevLayerSize, layerSize, CV_64F, w);
    stackW.push_back(weight.t());

    for (j = 0; j < stackW[i].rows; ++j)
    {
      for (k = 0; k < stackW[i].cols; ++k)
      {
        if (isnan(stackW[i].at<double>(j,k)))
        {
          printf("NaN found in weight matrix on iteration %i at index (%i, %i)\n", i, j, k);
          return false;
        }
      }
    }

    index += wLength;
    std::memcpy(b, theta + index, bLength*sizeof(double));
    index += bLength;
    bias = cv::Mat(1, bLength, CV_64F, b);
    stackB.push_back(bias.t());

    for (j = 0; j < stackB[i].rows; ++j)
    {
      for (k = 0; k < stackB[i].cols; ++k)
      {
        if (isnan(stackB[i].at<double>(j,k)))
        {
          printf("NaN found in bias matrix on iteration %i at index (%i, %i)\n", i, j, k);
          return false;
        }
      }
    }

    prevLayerSize = layerSize;
  }

  return true;
}

/*
 * Open the specified file and load in the classifier data contained within
 * 
 * The file must be a .mat file containing five fields:
 *   theta -- a vector containing all of the weight data
 *   inputSize -- the size of the input vector to the neural network
 *   hiddenSize -- the size of every hidden layer of the network
 *   numClasses -- the number of classes the network is trained to predict
 *   netconfig -- a struct describing the structure of the network
 *
 * Paramters:
 *   filename -- a string containing the name of the .mat file to be read from
 *
 * Return Value:
 *   True if the network loaded successfully; false otherwise
 */
bool NeuralNet::loadNN(std::string filename)
{
  /*
   * Open the classifier for reading
   */
  MATFile* matfile = matOpen(filename.c_str(), "r");

  if (matfile == NULL)
  {
    printf("The specified classifier file does not exist\n");
    return false;
  }

  /*
   * Extract the classifier struct from the file
   */
  mxArray* classifier = matGetVariable(matfile, "classifier");
  if (classifier == NULL)
  {
    printf("The classifier was not loaded successully\n");
    return false;
  }

  /*
   * Extract the individual elements from the struct
   */
  std::vector<std::string> failed;

  mxArray* field_theta = extractElement(classifier, "theta");
  if (field_theta == NULL) failed.push_back("theta");
  mxArray* field_inputSize = extractElement(classifier, "inputSize");
  if (field_inputSize == NULL) failed.push_back("inputSize");
  mxArray* field_hiddenSize = extractElement(classifier, "hiddenSize");
  if (field_hiddenSize == NULL) failed.push_back("hiddenSize");
  mxArray* field_numClasses = extractElement(classifier, "numClasses");
  if (field_numClasses == NULL) failed.push_back("numClasses");
  mxArray* field_netconfig = extractElement(classifier, "netconfig");
  if (field_netconfig == NULL) failed.push_back("netconfig");

  if (failed.size() > 0)
  {
    printf("The following elements failed to load from the classifier file:\n");
    std::vector<std::string>::iterator iter;
    for (iter = failed.begin(); iter != failed.end(); ++iter)
    {
      printf("\t%s\n", iter->c_str());
    }
    return false;
  }

  /*
   * Convert the extracted elements into the proper format
   */
  inputSize  = (int) *(mxGetPr(field_inputSize));
  hiddenSize = (int) *(mxGetPr(field_hiddenSize));
  numClasses = (int) *(mxGetPr(field_numClasses));
  netLoaded = buildNetwork(field_theta, field_netconfig);
  return netLoaded;
}

char NeuralNet::classify(cv::Mat image)
{
  /*
   * Return if the network has not been successfully loaded
   */
  if (!netLoaded)
  {
    printf("The network has not been properly initialized\n");
    return '?';
  }

  /*
   * Return if the image failed in reshaping
   */
  if (!reshape(image))
  {
    printf("The provided image could not be classified\n");
    return '?';
  }

  /*
   * Apply the autoencoder layers
   */
  cv::Mat pred = image, w, b;
  int i;
  for (i = 0; i < numLayers; ++i)
  {
    pred = sigmoid(stackW[i]*pred + stackB[i]);
  }

  /*
   * Apply the softmax layer and find the maximally likely value
   */
  pred = softmaxTheta*pred;
  double maxValue = 0;
  int prediction = 0;
  for (i = 0; i < numClasses; ++i)
  {
    if (pred.at<double>(i) > maxValue)
    {
      maxValue = pred.at<double>(i);
      prediction = i + 1;
    }
  }

  /*
   * Convert the prediction (in the range 0-61) to ASCII
   */
  return lookup(prediction);
}

/*
 * Reshapes an image into a vector such that it can be classified
 *
 * Parameters:
 *   image -- a reference to a cv::Mat to be reshaped into a vector
 *
 * Return Value:
 *   True if the reshaped cv::Mat has the appropriate number of dimensions,
 *     false otherwise
 */
bool NeuralNet::reshape(cv::Mat& image)
{
  image = image.reshape(0,1).t() / 255;
  image.convertTo(image, CV_64F);
  return (image.rows == inputSize);
}

/*
 * Applies the sigmoid function to a matrix
 *
 * sigmoid(x) = 1 / (1 + e^-x)
 * sigmoid'(x) = (sigmoid(x))(1 - sigmoid(x))
 *
 * Paramters:
 *   x -- a cv::Mat to which the sigmoid function should be applied
 *
 * Return Value:
 *   The results of the sigmoid function as applied to x in a different cv::Mat
 */
cv::Mat NeuralNet::sigmoid(cv::Mat x)
{
  cv::exp(-x, x);
  return 1 / (1 + x);
}

/*
 * Converts the prediction of the neural network into ASCII characters
 * 
 * The order of the values in the lookup table is (0-9), (A-Z), (a-z)
 *
 * Paramters:
 *   prediction -- an int between 0 and 62 correspsonding to the 62 possible
 *     character classifications and '?' for unrecognized or otherwise
 *     invalid inputs
 *
 * Return Value:
 *   The character which corresponds to the given prediction code
 */
char NeuralNet::lookup(int prediction)
{
  return (prediction < 0 || prediction > 62)  ? lookupTable[0] :  lookupTable[prediction];
}