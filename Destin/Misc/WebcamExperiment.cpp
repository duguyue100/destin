// supervised DeSTIN
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"

// DeSTIN Library
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "stdio.h"
#include "unit_test.h"
#include "BeliefExporter.h"

// standard library
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

void testNan(float * array, int len){
  for(int i = 0 ; i < len ; i++){
    if(isnan(array[i])){
      printf("input had nan\n");
      exit(1);
    }
  }
}

void convert(cv::Mat & in, float * out) {
  
  if(in.channels()!=1){
    throw runtime_error("Excepted a grayscale image with one channel.");
  }
  
  if(in.depth()!=CV_8U){
    throw runtime_error("Expected image to have bit depth of 8bits unsigned integers ( CV_8U )");
  }
  
  cv::Point p(0, 0);
  int i = 0 ;
  for (p.y = 0; p.y < in.rows; p.y++) {
    for (p.x = 0; p.x < in.cols; p.x++) {
      //i = frame.at<uchar>(p);
      //use something like frame.at<Vec3b>(p)[channel] in case of trying to support color images.
      //There would be 3 channels for a color image (one for each of r, g, b)
      out[i] = (float)in.at<uchar>(p) / 255.0;
      i++;
    }
  }
}

vector<int> readClassLabel(string classLabel)
{
  ifstream fin(classLabel.c_str());
  vector<int> ClassLabel;
  
  int label;
  while (fin >> label)
    {
      ClassLabel.push_back(label);
    }
  fin.close();
  
  return ClassLabel;
}

vector<string> readImageAddress(string imageAddress)
{
  ifstream fin(imageAddress.c_str());
  vector<string> imageadr;
  
  string filename="";
  while (fin >> filename)
    {
      imageadr.push_back(filename);
      filename="";
    }
  fin.close();
  return imageadr;
}

vector<cv::Mat> loadImage(vector<string> imageadr)
{
  vector<cv::Mat> loadedImg;
  
  int size=imageadr.size();
  
  for (int i=0;i<size;i++)
    {
      cv::Mat image, buffer;
      string imagename=imageadr[i];
      buffer=imread(imagename.c_str(), CV_LOAD_IMAGE_COLOR);
      cv::resize(buffer,buffer, Size(256,256),0,0,INTER_LINEAR);
      
      if (!buffer.data)
        {
	  cout << "error" << endl;
	  exit(-1);
        }
      
      cv::cvtColor(buffer, image, CV_BGR2GRAY);
      
      loadedImg.push_back(image);
      
      buffer.release();
      image.release();
    }
  
  return loadedImg;
}

float * callImage(Mat &image)
{
  float * float_image=new float[256*256];
  convert(image, float_image);
  
  testNan(float_image, 256*256);
  
  return float_image;
}

cv::Mat extractSIFTimage(cv::Mat &image)
{
  int nFeatures=64;
  int startx=8;
  int starty=8;
  int width=image.cols-startx*2;
  int height=image.rows-starty*2;
  
  SurfFeatureDetector detector(nFeatures);
  std::vector<KeyPoint> keypoints;
  
  Mat img(image, Rect(startx, starty, width, height));
  detector.detect(img, keypoints);
  //Mat img_keypoints_1;
  //drawKeypoints( image, keypoints, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  //imshow("test1", img_keypoints_1);
  
  cv::Mat result=Mat::ones(64, 64, CV_8UC1);
  
  int counter=0;
  for (int i=0;i<8;i++)
    {
      for (int j=0;j<8;j++)
        {
	  int x=i*8;
	  int y=j*8;
          
	  //cout << "[x] " << cvRound(keypoints[counter].pt.x) << " [y] " << cvRound(keypoints[counter].pt.y) << endl;
	  
	  Mat subimage(image, cv::Rect(cvRound(keypoints[counter].pt.x)+4, cvRound(keypoints[counter].pt.y)+4, 8, 8));
	  Mat insert(result, Rect(x,y,8,8));
	  subimage.copyTo(insert);
	  counter++;
        }
    }
  
  //imshow("test", result);
  //waitKey(0);
  
  return result;
}

vector<cv::Mat> processSIFTImages(vector<cv::Mat> &images)
{
  vector<cv::Mat> result;
    
  int size=images.size();
  
  for (int i=0;i<size;i++)
    {
      cv::Mat buffer=extractSIFTimage(images[i]);
      result.push_back(buffer);
      //cout << "Image: " << i << endl;
      buffer.release();
    }
  
  return result;
}

int findClassStart(int classNo, vector<int> classLabel)
{
  int result = 0;
  
  for (int i=0;i<classLabel.size();i++)
    {
      if (classLabel[i]==classNo)
	{
	  result=i;
	  return result;
	}
    }
  
  return -1;
}

int findClassEnd(int classNo, vector<int> classLabel)
{
  int result=0;

  for (int i=classLabel.size()-1;i>=0;i--)
    {
      if (classLabel[i]==classNo)
	{
	  result=i;
	  return result;
	}
    }
  
  return -1;
}

int main(int argc, char ** argv){
  
  // for video
  //cv::VideoCapture capture;
  
  //capture.open(0);
  
  //cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
  //moveWindow("test", 300, 300);
  
  // development plan
  // 1. change the source to image
  // 2. extract the largest object in the image using OpenCV built-in library
  // 3. learn the objects as DeSTIN features.
  // 4. Classification (Maybe OpenCV has a general classifier for this purpose)
  // 5. Change source to camera source
  // 6. Use DeSTIN learned feature to recognize object.
  
  /*
    while (true)
    {
    cv::Mat frame, buffer;
    if (!capture.isOpened()) break;
    
    capture >> buffer;
    
    
    cv::imshow("test", frame);
    
    if (cv::waitKey(5)==27)
    {
    capture.release();
    cv::destroyWindow("test");
    }
    }
  */
  
  // Next moves:
  // 1. Read image before training, try to solve memory issue [DONE]
  // 2. Change training process to single round but multiple times and remove beliefs after each image training [DONE]
  // 3. Write a program to organize SIFT features [DONE]
  // 4. Try to figure out how to organize a good feature [DONE]
  // 5. Change to color image
  // 6. Link semantic pointer
  // 7. If possible, integrate depth image as another channel
  // 8. Link to ROS [DONE]
  // 9. Smooth the process to meet online learning requirement
  // 10. Change training process [DONE]
  // 11. Try to use two training method to test temporal modelling of DeSTIN
  // 12. Find theory backup for DeSTIN
  
  // Further Plans:
  // 1. Tidy up code, write several utility functions as class
  // 2. Write classes for processing image source
  // 3. Try to study the whole library to know where I can change for improving or testing.
  // 4. Set up GitHub for updating changes.
  
  /////// TRAINING AND TESTING IMAGE //////
  vector<string> imageTrainAdr=readImageAddress("training.txt");
  vector<string> imageTestAdr=readImageAddress("testing.txt");
  cout << "[MESSAGE] Training and testing images address loaded" << endl;
  
  vector<int> trainingClassLabel=readClassLabel("trainingclassLabel.txt");
  vector<int> testingClassLabel=readClassLabel("testingclassLabel.txt");
  cout << "[MESSAGE] Training and testing images label loaded" << endl;
  
  vector<cv::Mat> trainingImages=loadImage(imageTrainAdr);
  vector<cv::Mat> testingImages=loadImage(imageTestAdr);
  cout << "[MESSAGE] Training and testing images loaded" << endl;
  
  //vector<cv::Mat> trainingSIFTImages=processSIFTImages(trainingImages);
  //cout << "[MESSAGE] Training Image Processed" << endl;
  //vector<cv::Mat> testingSIFTImages=processSIFTImages(testingImages);
  //cout << "[MESSAGE] Testing Image Processed" << endl;
  
  ////// PARAMETERS //////
  
  int noTrainingImage=imageTrainAdr.size();
  int noTestingImage=imageTestAdr.size();
  int noClass=trainingClassLabel[trainingClassLabel.size()-1];
  int counter=0;
   
  for (int classNo=1; classNo<=noClass; classNo++)
    {
      int trainClassStart=findClassStart(classNo, trainingClassLabel);
      int trainClassEnd=findClassEnd(classNo, trainingClassLabel);
      int testClassStart=findClassStart(classNo, testingClassLabel);
      int testClassEnd=findClassEnd(classNo, testingClassLabel);
      
      ////// SETTING FOR DeSTIN //////
      SupportedImageWidths siw = W256;
      uint centroid_counts[]  = {32,32,32,32,32,16,10};
      bool isUniform = true;
      //bool isUniform = false;
      int nLayers=7;
      DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform);
      network->setFixedLearnRate(.1);
      
      cout << "[MESSAGE] DeSTIN network initialization completed. No. " << classNo << endl;
      
      BeliefExporter * featureExactor=new BeliefExporter(*network, 5);
      cout << "[MESSAGE] Feature extractor initialization completed" << endl;

      ////// PROCESSING //////
      
      for (int i=trainClassStart; i<=trainClassEnd; i++)
	{
	  network->clearBeliefs();
	  for (int j=0;j<5*nLayers;j++)
	    {
	      float * float_image=callImage(trainingImages[i]);
	      network->doDestin(float_image);
	    }
	  cout << "[MESSAGE][TRAINING] IMAGE:" << i+1 << "\r";
	  cout.flush();
	}
      
      ////// SAVE DESTIN //////
      
      cout << "[MESSAGE] Saving DeSTIN network" << endl;
      stringstream ss;
      
      string destin_name="destin_";
      ss << "result/destin_network/destin_" << classNo;
      ss >> destin_name;
      
      network->save((char *)destin_name.c_str());
      cout << "[MESSAGE] DeSTIN network saved" << endl;
      
      ////// EXTRACT FEATURES //////
      
      cout << "[MESSAGE] Extracting features" << endl;
      for (int i=0;i<nLayers;i++) network->setLayerIsTraining(i, false);
      
      for (int i=trainClassStart;i<=trainClassEnd;i++)
	{
	  network->clearBeliefs();
	  for (int j=1;j<=2*nLayers;j++)
	    {
	      counter++;
	      float * float_image=callImage(trainingImages[i]);
	      network->doDestin(float_image);
	    }
	  
	  featureExactor->writeBeliefToMat("result/destin_features/trainOutput.txt");
	  cout << "[MESSAGE][EXTRACT FEATURE][TRAINING] IMAGE:" << i+1 << "\r";
	  cout.flush();
	}
      
      cout << endl;
      
      for (int i=testClassStart;i<=testClassEnd;i++)
	{
	  network->clearBeliefs();
	  for (int j=1;j<=2*nLayers;j++)
	    {
	      counter++;
	      float * float_image=callImage(testingImages[i]);
	      network->doDestin(float_image);
	    }
	  
	  featureExactor->writeBeliefToMat("result/destin_features/testOutput.txt");
	  cout << "[MESSAGE][EXTRACT FEATURE][TESTING] IMAGE:" << i+1 << "\r";
	  cout.flush();
	}
      cout << endl;
      cout << "[MESSAGE] Features are extracted" << endl;
      
      ////// REMOVE DESTIN ///////
      delete network;
      
    }    
  /*
    for (int layer=6;layer<=6;layer++)
    {
        string run_dir="./training/result/";
        string orig_dir=run_dir+"orig/";
        string orige_dir=run_dir+"orig_e/";
        string highweighted_dir = run_dir+"highweight/";
        string highweightede_dir = run_dir+"highweight_e/";
        
        int bn=network->getBeliefsPerNode(layer);
        cout << bn << endl;
        network->setCentImgWeightExponent(1);
        network->updateCentroidImages();
        
        for (int i=0;i<bn;i++)
        {
            stringstream ss;
            string name="";
            ss << layer << '_' << i;
            ss >> name;
            string fn=orig_dir+name+".png";
            network->saveCentroidImage(layer, i, fn, 256, false);
            
            fn=orige_dir+name+".png";
            network->saveCentroidImage(layer, i, fn, 256, true );
        }
        
        network->setCentImgWeightExponent(4);
        network->updateCentroidImages();
        for (int i=0;i<bn;i++)
        {
            stringstream ss;
            string name="";
            ss << layer << '_' << i;
            ss >> name;
            string fn=highweighted_dir+name+".png";
            network->saveCentroidImage(layer, i, fn, 256, false);
            
            fn=highweightede_dir+name+".png";
            network->saveCentroidImage(layer, i, fn, 256, true );
        }
    }
    */
    
  return 0;
}
