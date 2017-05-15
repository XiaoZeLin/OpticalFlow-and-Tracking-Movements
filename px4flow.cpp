#include "flow.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iomanip>
#include <ctime>



using namespace std;
using namespace cv;


int main()
{
	//VideoCapture cap(0);
    VideoCapture cap("4.avi");

    Mat pre, cur;
    int flag = 1;
	cap >> pre;
    Size dsize = Size(IMG_WIDTH, IMG_HEIGHT);
    resize(pre, pre, dsize);
	clock_t pre_t = clock();
	waitKey(15);
	while(1)
	{
		float flow_x;
		float flow_y;
		cap>>cur;
        resize(cur, cur, dsize);
		clock_t cur_t  = clock();
		cout << "time: " << (double)(cur_t - pre_t)/CLOCKS_PER_SEC << "s\t";
		compute_flow((uint8_t*)pre.data, (uint8_t*)cur.data, 0, 0, 0, &flow_x, &flow_y, (double)(cur_t-pre_t)*1e6/CLOCKS_PER_SEC);
		pre_t = cur_t;
		cout << fixed << setprecision(4) << setw(10) << left << flow_x << flow_y << endl;
		pre = cur;
		imshow("pic", cur);
		char key = waitKey(flag);
        if(key == 'q') break;
        if(key == 's') flag = ~flag;
	}
	return 0;
}