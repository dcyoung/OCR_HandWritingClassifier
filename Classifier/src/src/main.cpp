#include "NeuralNet.h"
#include "CharacterExtractor.h"

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: %s [classifier] [image] ...\n", argv[0]);
    return 0;
  }
  
  NeuralNet nn;
  CharacterExtractor ce;

  /*
   * Load the neural network from the specified classifier file
   */
  if (!nn.loadNN(argv[1]))
  {
    fprintf(stderr, "There were errors while loading the neural network.\n");
    return 0;
  }

  int i;
  for (i = 2; i < argc; ++i)
  {
    printf("Beginning classification on: %s\n", argv[i]);

    /*
     * Load the specified image
     */
    cv::Mat srcImage = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
    if (!srcImage.data)
    {
      printf("The specified image could not be found.\n");
      return 0;
    }

    /*
     * Preprocess the image
     */
    int errCode;
    cv::Mat destImage, threshImage, croppedImage;
    std::vector<cv::Rect> boundingBoxes, charBoundingBoxes;
    if ((errCode = ce.preprocessImage(srcImage, destImage, threshImage, false)) != 0)
    {
      printf("Preprocessing failed with code %i\n", errCode);
      return errCode;
    }
    if ((errCode = ce.findBoundingBoxes(destImage, boundingBoxes)) != 0)
    {
      printf("Finding bounding boxes failed with code %i\n", errCode);
      return errCode;
    }
    if ((errCode = ce.findFullCharBoxes(boundingBoxes, charBoundingBoxes)) != 0)
    {
      printf("Correcting bounding boxes failed with code %i\n", errCode);
      return errCode;
    }

    cv::imwrite("intermediary/preprocessed.jpg", destImage);
    cv::imwrite("intermediary/thresholded.jpg", threshImage);

    cv::Mat outImage = srcImage.clone(), boundImage = srcImage.clone();

    /*
     * Classify every character found within the image
     */
    std::vector<cv::Rect>::iterator iter;
    int j = 0;
    for (iter = charBoundingBoxes.begin(); iter != charBoundingBoxes.end(); ++iter)
    {
      if ((errCode = ce.cropImage(threshImage, croppedImage, *iter)) != 0)
      {
        printf("Cropping failed with code %i\n", errCode);
        return errCode;
      }

      cv::imwrite("intermediary/cropped_" + std::to_string(j) + ".jpg", croppedImage);
      ++j;

      char label = nn.classify(croppedImage);
      cv::putText(outImage, std::string(&label), iter->br(), CV_FONT_HERSHEY_SIMPLEX, 1.25, CV_RGB(255,0,0), 2);
      cv::rectangle(boundImage, iter->tl(), iter->br(), CV_RGB(0,0,255));
    }

    /*
     * Save the results
     */
    std::string filename(argv[i]);
    int lastindex = filename.find_last_of("."); 
    filename = filename.substr(0, lastindex); 
    filename += "_output.jpg";
    cv::imwrite(filename, outImage);
    cv::imwrite("intermediary/boundingBoxes.jpg", boundImage);
  }
  return 0;
}