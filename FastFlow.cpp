// 光流与物体运动1.cpp : 定义控制台应用程序的入口点。
//
#include <time.h>
#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "flow.h"

using namespace cv;
using namespace std;


void duan_OpticalFlow(Mat &frame, Mat & result);
bool addNewPoints();
bool acceptTrackedPoint(int i);


Mat curgray;	// 当前图片
Mat pregray;	// 预测图片
vector<Point2f> point[2];	// point0为特征点的原来位置，point1为特征点的新位置
vector<Point2f> initPoint;	// 初始化跟踪点的位置
vector<Point2f> features;	// 检测的特征
vector<uchar> status;	// 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;
vector<KeyPoint> keypoints;

int main()
{
	//VideoCapture cap(0);
	VideoCapture cap("3.avi");
	Mat matSrc;
	Mat matRst;
	Mat matRoi;
	int flag = 1;
	
	//set ROI
	Size dsize = Size(IMG_WIDTH, IMG_HEIGHT);
	const double roi_x0 = IMG_WIDTH * 0.0;
	const double roi_y0 = IMG_HEIGHT * 0.25;
	const double roi_width = IMG_WIDTH;
	const double roi_height = IMG_HEIGHT * 0.65;


	// perform the tracking process
	printf("Start the tracking process, press ESC to quit.\n");
	clock_t start, end;
	while(1) 
	{
		cap >> matSrc;
		if (matSrc.empty())
		{
			cout << "Error : Get picture is empty!" << endl;
			return 0;
		}


		resize(matSrc, matSrc, dsize);
		//Mat matRoi(matSrc, Rect(roi_x0, roi_y0, roi_width, roi_height));
		//rectangle(matSrc, Rect(roi_x0, roi_y0, roi_width, roi_height), Scalar(0, 255, 0), 2);
		//imshow("ROI", matSrc);
		start = clock();
		//duan_OpticalFlow(matRoi, matRst);
		duan_OpticalFlow(matSrc, matRst);
		end = clock();
		printf("总时间为:%lfs\n", (double)(end - start) / CLOCKS_PER_SEC);


		char key = waitKey(flag);
		if (key == 'q') break;
		if (key == 's') flag = ~flag;
	}

	return 0;

}



void duan_OpticalFlow(Mat &frame, Mat & result)
{
	cvtColor(frame, curgray, CV_BGR2GRAY);
	frame.copyTo(result);
	 Mat Feature;
    Feature = frame.clone(); 

	if (addNewPoints())
	{
		keypoints.clear();
		features.clear();
		

		clock_t start1, end1;
		start1 = clock();
        Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
        fast->detect(curgray,keypoints, 40);
		for (int i = 0; i < (int)keypoints.size(); i++)
			features.push_back(keypoints[i].pt);
		end1 = clock();
		printf("特征点检测时间为:%lfs  ", (double)(end1 - start1) / CLOCKS_PER_SEC);


		if (features.size() > 0)
		{
			point[0].insert(point[0].end(), features.begin(), features.end());
			initPoint.insert(initPoint.end(), features.begin(), features.end());
		}
	}


	if (pregray.empty())
	{
		curgray.copyTo(pregray);
	}

	
	clock_t start2, end2;
    start2 = clock();
    calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);  //使用光流算法得到移动后的特征点point[1]
    end2 = clock();
    printf("光流计算时间:%lfs     ", (double)(end2 - start2) / CLOCKS_PER_SEC);


	int k = 0;  
    float dis_x = 0;
    float dis_y = 0;
	for (size_t i = 0; i<point[1].size(); i++)
	{
		if (acceptTrackedPoint(i))
		{
			dis_x += point[1][i].x - point[0][i].x;
            dis_y += point[1][i].y - point[0][i].y;
			initPoint[k] = initPoint[i];
			point[1][k++] = point[1][i];
		}
	}



	point[1].resize(k);
	initPoint.resize(k);

	for (size_t i = 0; i<point[1].size(); i++)
	{
		line(result, initPoint[i], point[1][i], Scalar(0, 0, 255));
		circle(result, point[1][i], 3, Scalar(0, 255, 0), -1);
	}

	for(size_t i=0;i<features.size();i++)     // 画出选择的特征点
    {
        circle(Feature, features[i],3,Scalar(255, 0, 0),-1);
    }

	if(k == 0) cout<<"没有检测到移动的点"<<"    ";
    else cout<<fixed<<setprecision(2)<<"dis_x = "<<dis_x/(float)k<<"   "<<"dis_y = "<<dis_y/(float)k<<"  ";

	swap(point[1], point[0]);
	swap(pregray, curgray);


	imshow("Optical Flow Demo", result);
	imshow("特征点", Feature);
}


bool addNewPoints()
{
	return point[0].size() <= 10;
}


bool acceptTrackedPoint(int i)
{
	return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 2);
}