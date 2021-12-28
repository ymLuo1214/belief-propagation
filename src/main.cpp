#include "match.hpp"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "tictoc.hpp"

#define KernelSize 8
int readImage(std::string filename, cv::Mat &mat)
{
    std::ifstream refFile(filename, std::ios::binary);
    if (refFile.fail())
        return -1;
    uint16_t raw[row * col];
    refFile.read((char *)raw, sizeof(raw));
    refFile.close();
    for (size_t r = 0; r < row; r++)
    {
        uchar *mat_ptr = mat.ptr<uchar>(row - r - 1);
        for (size_t c = 0; c < col; c++)
        {
            mat_ptr[col - c - 1] = raw[c * row + r] & 0xff;
        }
    }
    return 0;
}

void colormap(const cv::Mat &src, cv::Mat &dst)
{
    const int kCountSize = 4096;
    int gCount[kCountSize] = {
        0,
    };
    int num_points = 0;
    for (int r = 0; r < row; ++r)
    {
        const uint16_t *src_ptr = src.ptr<uint16_t>(r);
        for (int c = 0; c < col; ++c)
        {
            uint16_t val = src_ptr[c];
            if (val == 0)
                continue;
            num_points++;
            gCount[val]++;
        }
    }
    float sum = 0;
    for (int i = 0; i < kCountSize; ++i)
    {
        sum += gCount[i];
        gCount[i] = sum / num_points * 4095;
    }

    cv::Mat map(1, 4096, CV_8UC3, cv::Scalar::all(0));
    map.forEach<cv::Vec3b>([](cv::Vec3b &pixel, const int *pos)
                           {
                               int x = pos[1] / 4;
                               if (x < 128)
                                   pixel = cv::Vec3b(128 + x, 0, 0);
                               else if (x < 384)
                                   pixel = cv::Vec3b(255, x - 128, 0);
                               else if (x < 640)
                                   pixel = cv::Vec3b(255 - x + 384, 255, x - 384);
                               else if (x < 896)
                                   pixel = cv::Vec3b(0, 255 - x + 640, 255);
                               else if (x < 1024)
                                   pixel = cv::Vec3b(0, 0, 255 - x + 896);
                               else
                                   pixel = cv::Vec3b(0, 0, 128);
                           });

    dst.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int *pos)
                           {
                               uint16_t val = src.at<uint16_t>(pos[0], pos[1]);
                               if (val == 0 || pos[0] < KernelSize || pos[0] > row - KernelSize || pos[1] < KernelSize || pos[1] > col - KernelSize)
                                   return;
                               pixel = map.at<cv::Vec3b>(0, gCount[val]);
                           });
}
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("usage: %s m700.raw test.raw color.png output.pcd\n", argv[0]);
        return 0;
    }
    cv::Mat src(row, col, CV_8UC3);
    cv::Mat dist;
    TicToc tt;
    int cnt = 0;
    // cv::Mat ref_o(row, col, CV_8UC1);
    // cv::Mat test_o(row, col, CV_8UC1);
    cv::Mat ref_o(row, col, CV_16UC1);
    cv::Mat test_o(row, col, CV_16UC1);
    // readImage(argv[1], ref_o);
    // readImage(argv[2], test_o);
    ref_o = cv::imread(argv[2]);
    test_o = cv::imread(argv[1]);
    cv::resize(ref_o, ref_o, cv::Size(1280, 800));
    cv::resize(test_o, test_o, cv::Size(1280, 800));
    match my_match(ref_o, test_o);
    while (cnt < 5)
    {
        cnt++;
        std::cout << cnt << "," << my_match.iteRate() << std::endl;
    }
    my_match.disImgConvert();
    my_match.disparity.convertTo(dist, CV_16UC1);
    colormap(dist * 16, src);
    // my_match.destoryCost();
    cv::imshow("L", ref_o);
    cv::imshow("R", test_o);
    cv::imshow("disparity", src);
    cv::imshow("dis", my_match.disparity);
    cv::waitKey(1000000);
    return 0;
}