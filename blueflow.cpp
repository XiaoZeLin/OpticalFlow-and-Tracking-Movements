#include <iostream>    
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur  
#include <opencv2/ml/ml.hpp>   
#include <chrono>
#include <time.h>
#include <string>
#include <vector>
#include "flow.h"
    
using namespace cv;  
using namespace std;  


void duan_OpticalFlow(Mat &frame, Mat & result);  
bool addNewPoints();  
bool acceptTrackedPoint(int i);  
    
    
Mat curgray;    // 当前图片  
Mat pregray;    // 预测图片  
vector<Point2f> point[2]; // point0为特征点的原来位置，point1为特征点的新位置  
vector<Point2f> initPoint;    // 初始化跟踪点的位置  
vector<Point2f> features; // 检测的特征  
int maxCount = 100;         // 检测的最大特征数  
double qLevel = 0.01;   // 特征检测的等级  
double minDist = 10.0;  // 两特征点之间的最小距离  
vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0  
vector<float> err;  
    
int main(int argc, char **argv)  
{  
    
    Mat matSrc;  
    Mat matRst;  
    
 
    // perform the tracking process
    char key = 'a';  
    printf("Start the tracking process, press 's' to start or stop, press 'q' to quit.\n");  
    int flag = 1;
    //VideoCapture cap(0);
    VideoCapture cap(0);

    while(1) 
    {  
        // get frame from the bluefox2
        cap>>matSrc;
        if( matSrc.empty() )
        {
            cout << "***捕获到空帧***"<<endl;
            return 0;
        }


        //改变图像尺寸
        Size dsize = Size(IMG_WIDTH, IMG_HEIGHT);
        resize(matSrc, matSrc, dsize);


        clock_t start, end;
        start = clock();
        duan_OpticalFlow(matSrc, matRst);  
        end = clock();
        printf("time=%lfs     ", (double)(end - start) / CLOCKS_PER_SEC);
        
        key = cvWaitKey(flag);

        if(key == 's') flag = ~flag;
        if(key == 'q') break;
    }  
    
    return 0;  
    
}  
    
    
    
void duan_OpticalFlow(Mat &frame, Mat & result)  
{  
    cvtColor(frame, curgray, CV_BGR2GRAY);  
    result = frame.clone(); 
    Mat Feature;
    Feature = frame.clone(); 
    
    if (addNewPoints())    //添加新的跟踪的特征点
    {  
        clock_t start1, end1;
        start1 = clock();
        goodFeaturesToTrack(curgray, features, maxCount, qLevel, minDist);  
        end1 = clock();
        printf("time1=%lfs     ", (double)(end1 - start1) / CLOCKS_PER_SEC);

        point[0].insert(point[0].end(), features.begin(), features.end());  
        initPoint.insert(initPoint.end(), features.begin(), features.end());  
    }  
    
    
    if (pregray.empty())  
    {  
        curgray.copyTo(pregray);  
    }  
    
    clock_t start2, end2;
    start2 = clock();
    calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);  //使用光流算法得到移动后的特征点point[1]
    end2 = clock();
    printf("time2=%lfs     ", (double)(end2 - start2) / CLOCKS_PER_SEC);
    
    
    int k = 0;  
    float dis_x = 0;
    float dis_y = 0;
    for (size_t i = 0; i<point[1].size(); i++)    //选择前后对应的特征点： point[0] 和 对应的 point[1]
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
    
    for (size_t i = 0; i<point[1].size(); i++)    // 画出点的移动方向
    {  
        line(result, initPoint[i], point[1][i], Scalar(0, 0, 255));  
        circle(result, point[1][i], 3, Scalar(0, 255, 0), -1);  
    }  

    for(size_t i=0;i<features.size();i++)     // 画出选择的特征点
    {
        circle(Feature, features[i],3,Scalar(255, 0, 0),-1);
    }
    
    if(k == 0) cout<<"没有检测到移动的点"<<endl;
    else cout<<"dis_x = "<<dis_x/(float)k<<"   "<<"dis_y = "<<dis_y/(float)k<<endl;
    
    swap(point[1], point[0]);  
    swap(pregray, curgray);  
    
    
    imshow("Optical Flow Result", result);  
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


