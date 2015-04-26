#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <stdio.h>
#include <string>
#include <iostream>
#include <dirent.h>
#include <windows.h>

using namespace cv;
// use a few things from the std namespace for console I/O
using std::cout;
using std::cin;
using std::endl;

//specify some parameters that will be used for every processed image
string rootPath;
Size blur_kernel = Size(4,4);
int erode_kernel_size = 2; //article says 4
double area;
Size size(32, 32);
int pad = 4;
int square_dimension,x,y,max_x, max_y, processed_count;
Size image_dim;

/*
Help: prints help information
*/
static void help(){
	printf("\nThis program is my first attempt to modify an image using openCV.\n"
		"Usage:\n"
		"   ...to be written\n");
}

/*
Helper function for converting a string to an LPCWSTR
*/
std::wstring s2ws(const std::string& s){
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

/*
Process a found image
*/
int processImage(Mat src_image, Mat& dst_image, string image_name){
	//convert it to grayscale
	Mat gray_image;
	cvtColor(src_image, gray_image, CV_BGR2GRAY);

	//Kernel Smoothing: apply a blur to reduce noise using an 8x8 kernel
	Mat blur_image;
	blur(gray_image, blur_image, blur_kernel);

	//Threshold
	//convert to b / w image, and invert: ie: (black, white) = (0, 1)
	Mat threshold_image;
	threshold(blur_image, threshold_image, 200, 255, 1); //1 for inverted binary threshold type

	//Erosion:
	Mat eroded_image;
	Mat eroded_element = getStructuringElement(1,
		Size(2 * erode_kernel_size + 1, 2 * erode_kernel_size + 1),
		Point(erode_kernel_size, erode_kernel_size));
	erode(threshold_image, eroded_image, eroded_element);
	
	////Dilation:
	//Mat dilated_image;
	//int dilate_kernel_size = 2; //article says 2
	//Mat dilated_element = getStructuringElement(1,
	//	Size(2 * dilate_kernel_size + 1, 2 * dilate_kernel_size + 1),
	//	Point(dilate_kernel_size, dilate_kernel_size));

	//dilate(eroded_image, dilated_image, dilated_element);

	//Contours, polygonal conversions and bouding boxes
	/// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(eroded_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	////draw contours
	//RNG rng(12345);
	//Mat contour_image = Mat::zeros(eroded_image.size(), CV_8UC3);
	//for (int i = 0; i< contours.size(); i++){
	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	drawContours(contour_image, contours, i, color, 2, 8, hierarchy, 0, Point());
	//}

	// Approximate the contours as polygons, and define a bounding rectangle that holds each polygon
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	float largest_area = 0;
	int largest_contour_index = 0;
	for (int i = 0; i < contours.size(); i++){
		area = contourArea(contours[i], false);  //  Find the area of contour
		if (area>largest_area){
			largest_area = area;
			largest_contour_index = i; 
		}
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	
	/*Mat boundingBox_image = Mat::zeros(contour_image.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++){
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(boundingBox_image, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(boundingBox_image, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}*/
	

	square_dimension = 2*pad + max(boundRect[largest_contour_index].width, boundRect[largest_contour_index].height);
	x = boundRect[largest_contour_index].x - pad;
	y = boundRect[largest_contour_index].y - pad;
	image_dim = threshold_image.size();
	max_x = image_dim.width;
	max_y = image_dim.height;
	
	//check to make sure the padding doesn't try to select pixels outside the original dimension
	try{
		if ((x - pad <= 0) || (x + square_dimension + pad >= max_x) || (y - pad <= 0) || (y + square_dimension + pad >= max_y)){
			/*square_dimension = square_dimension - 2 * pad;
			x = x + pad / 2;
			y = y + pad / 2;*/

			Rect crop_region = boundRect[largest_contour_index];
			Mat cropped_image = threshold_image(crop_region); //crop the image using the largest bounding rectangle
			//Mat cropped_image = eroded_image(crop_region); //crop the image using the largest bounding rectangle		

			Mat resized_cropped_image;
			resize(cropped_image, resized_cropped_image, size);
			dst_image = resized_cropped_image;
			return 0;
		}
	}
	catch (int e){
		cout << "An exception occurred:" << e << endl;
		cout << "1. File in question was: " << image_name << endl;
		return -1;
	}




	try
	{
		Rect crop_region = Rect(x, y, square_dimension, square_dimension);
		Mat cropped_image = threshold_image(crop_region); //crop the image using the largest bounding rectangle
		//Mat cropped_image = eroded_image(crop_region); //crop the image using the largest bounding rectangle		
		
		Mat resized_cropped_image;
		resize(cropped_image, resized_cropped_image, size);
		dst_image = resized_cropped_image;
		return 0;
	}
	catch (int e){
		cout << "An exception occurred:" << e << endl;
		cout << "2. File in question was: " << image_name << endl;
		return -1;
	}
	
	return 0;
}


/*
Recursive function to check each subdirectory and call a processing function on any images inside
*/
int check_n_process_files(string rel_src_dir, string full_dst_dir){
	// Create a pointer to a directory
	DIR *pdir = NULL; // initialise pointer to NULL
	cout << "Opening directory..." << '\n';

	pdir = opendir(rel_src_dir.c_str()); //refer to the directory specified by the user
	struct dirent *pent = NULL;
	// check if pdir was initialised correctly
	if (pdir == NULL) {
		cout << "\nERROR! pdir could not be initialized correctly";
		exit(3);
	}
	// while there is still something in the directory to list, read it
	while (pent = readdir(pdir)){
		//check that pent was initialized properly
		if (pent == NULL){
			cout << "\nERROR! pent could not be initialized correctly";
			exit(3);
		}

		//check if the read element is a directory or a file(image). If it is a directory, recurse into it.
		string file_name = pent->d_name;
		if (pent->d_type == DT_DIR){
			if (file_name.find('.') == std::string::npos){
				//cout << "Directory: " + file_name << endl;
				string new_dir = full_dst_dir + file_name;
				std::wstring stemp = s2ws(new_dir);
				LPCWSTR result = stemp.c_str();
				if (CreateDirectory(result, NULL) || ERROR_ALREADY_EXISTS == GetLastError()){
					cout << "Created New Directory" << endl;
					string new_src = rel_src_dir + file_name + "\\";
					string new_dst = full_dst_dir + file_name + "\\";
					check_n_process_files(new_src, new_dst); //recurse into the directory
				}
				else
				{
					// Failed to create directory.
				}
			}
		}
		else if (file_name.find('.bmp') != std::string::npos){
			
			string image_name = file_name;
			//cout << "FileName: " + image_name << endl;

			////import the image
			Mat src_image;
			string imagePath = rel_src_dir + image_name;
			src_image = imread(imagePath, 1);
			if (!src_image.data){
				printf(" No image data \n ");
				return -1;
			}
			
			//Process the image
			
			Mat processed_image;
			int result = processImage(src_image, processed_image, image_name);
			if (result == 0){
				//Write/Output the processed image
				string final_name = full_dst_dir + image_name.substr(0, image_name.size() - 4) + ".jpg";
				imwrite(final_name, processed_image);
				processed_count++;
				if (processed_count % 1000 == 0){
					cout << "Processed " << processed_count << " images so far..." << endl;
				}
			}
			else{
				//don't write it out.
			}

		}
		else{
			//do nothing, this is not a directory or an image file
		}
	}
	//close the directory
	closedir(pdir);
	return EXIT_SUCCESS; // everything went OK
}


/*
Main Function: asks the user for a directory of images to be processed and kicks off the processing
*/
int main(){

	help();
	rootPath = "C:\\Users\\David\\Documents\\Visual Studio 2013\\Projects\\ConsoleApplication_test_OpenCV\\ConsoleApplication_test_OpenCV\\";

	//Ask the user to specify a directory name relative to the root directory.
	string src_dir;
	cout << "\n\n Enter a relative directory path to the root image folder. Be sure to use double backslash \\\\ between subdirectories. \n example: images\\\\test_images\\\\ \n";
	getline(cin, src_dir);
	if (src_dir == "") {
		exit(1);
	}
	cout << "You have specified directory: " + src_dir << endl;

	processed_count = 0;
	string dst_dir = rootPath + "images\\created_images\\";
	check_n_process_files(src_dir, dst_dir);

	cin.get();
	waitKey();
    return 0;
}

#ifdef _EiC
main(1,"drawing.c");
#endif
