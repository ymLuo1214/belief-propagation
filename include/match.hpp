#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <array>
#include <vector>
#define inf 1e5
#define MAX_DISP 28
#define row 1280
#define col 800
#define lamda 10  //constant in calCoorelation func
#define MAX_SUB 5 //constant in calCoorelation func
#define KSize 8
#define KernelSize 8
enum Direction
{
    UP,
    DOWN,
    LEFT,
    RIGHT
};

struct MarkovNode
{
    float *mat_cost;
    float *up_mes;
    float *down_mes;
    float *left_mes;
    float *right_mes;
    int best_disp;
};

class match
{
public:
    match(){};
    match(const cv::Mat &ref, const cv::Mat &test);
    ~match(){};
    double iteRate();
    cv::Mat disparity;
    cv::Mat pts;
    void disImgConvert();

    void ptsConvert(pcl::PointCloud<pcl::PointXYZRGB> &cloud,cv::Mat rgb);

private:
    std::vector<MarkovNode> grid;
    cv::Mat img_L;
    cv::Mat img_R;
    void sendMsg(int x, int y, Direction dir);
    float matchCost(int r1, int c1, int dist) const;
    float calCoorelation(int x, int y)
    {
        int sub = x - y;
        float res = 5 * std::min(abs(sub), 60) + 1;
        return log(res);
    }
};

cv::Mat preProcess(const cv::Mat &mat, bool clear);

float pDfunc(int cost);