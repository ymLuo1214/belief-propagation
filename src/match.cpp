#include "match.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
match::match(const cv::Mat &ref, const cv::Mat &test)
{
    srand(time(0));
    img_L = preProcess(ref, true);
    img_R = preProcess(test, true);
    grid.resize(col * row);
    disparity = cv::Mat(row, col, CV_8UC1);
    for (u_int i = 0; i < col * row; i++)
    {
        grid[i].left_mes = new float[MAX_DISP];
        grid[i].right_mes = new float[MAX_DISP];
        grid[i].up_mes = new float[MAX_DISP];
        grid[i].down_mes = new float[MAX_DISP];
        grid[i].mat_cost = new float[MAX_DISP];
        grid[i].best_disp = rand() % 29;
        for (int j = 0; j < MAX_DISP; j++)
        {
            grid[i].mat_cost[j] = 0;
            grid[i].up_mes[j] = 0;
            grid[i].down_mes[j] = 0;
            grid[i].left_mes[j] = 0;
            grid[i].right_mes[j] = 0;
        }
    }
    for (u_int x = MAX_DISP; x < row - MAX_DISP; x = x + 1)
    {
        for (u_int y = MAX_DISP; y < col - MAX_DISP; y = y + 1)
        {
            float sum_costs = 0;
            for (int d = 0; d < MAX_DISP; d++)
            {
                grid[x * col + y].mat_cost[d] = matchCost(x, y, d);
                sum_costs += grid[x * col + y].mat_cost[d] * grid[x * col + y].mat_cost[d];
            }
            sum_costs = sqrt(sum_costs);
            for (int d = 0; d < MAX_DISP; d++)
            {
                grid[x * col + y].mat_cost[d] /= sum_costs;
            }
        }
    }
    std::cout << "Initialize finished~~" << std::endl;
}

cv::Mat preProcess(const cv::Mat &mat, bool clear)
{
    cv::Mat res = mat.clone();
    for (int r = KernelSize; r < row - KernelSize; r++)
    {
        uchar *res_ptr = res.ptr<uchar>(r);
        for (int c = KernelSize; c < col - KernelSize; c++)
        {
            float sum = 0;

            float squareSum = 0;

            for (int i = -KernelSize; i <= KernelSize; i++)
            {
                const uchar *mat_ptr = mat.ptr<uchar>(r + i);
                for (int j = -KernelSize; j <= KernelSize; j++)
                {
                    float ptmp_cj = mat_ptr[c + j];

                    sum += ptmp_cj;
                    squareSum += ptmp_cj * ptmp_cj;
                }
            }
            float average = static_cast<float>(sum) / 289;
            float x = static_cast<float>(mat.at<uchar>(r, c)) - average;
            x /= sqrtf(static_cast<float>(squareSum) / 289 - average * average);
            int res_tmp = clear ? (x * 256) : (x + 1) * 128;
            if (res_tmp > 255)
                res_tmp = 255;
            else if (res_tmp < 0)
                res_tmp = 0;
            res_ptr[c] = res_tmp;
        }
    }
    return res;
}

float match::matchCost(int r1, int c1, int dist) const
{
    float cost = 0;
    int sum = 0;
    for (int i = -KSize; i <= KSize; i++)
    {

        const uchar *ref_ptr = img_L.ptr<uchar>(r1 + i);
        const uchar *mat_ptr = img_R.ptr<uchar>(r1 + i);

        for (int j = -KSize; j <= KSize; j++)
        {
            sum += abs(ref_ptr[c1 + j] - mat_ptr[c1 - dist + j]);
        }
    }
    sum /= ((2 * KSize + 1) * (2 * KSize + 1));
    // cost = log(sum) < 0 ? 0 : log(sum);
    return sum;
}
// float match::matchCost(int r1, int c1, int dist) const
// {
//     float cost = 0;
//     for (int i = -KSize; i <= KSize; i++)
//     {

//         const uchar *ref_ptr = img_L.ptr<uchar>(r1 + i);
//         const uchar *mat_ptr = img_R.ptr<uchar>(r1 + i);
//         for (int j = -KSize; j <= KSize; j++)
//         {
//             if (ref_ptr[c1 + j] == 0)
//                 continue;
//             else
//             {
//                 float ref_ij = static_cast<float>(ref_ptr[c1 + j]);
//                 cost += ref_ij * log(ref_ij / (1e-3 + mat_ptr[c1 - dist + j]));
//             }
//         }
//     }
//     return cost;
// }

void match::sendMsg(int x, int y, Direction dir)
{
    float *msgs = new float[MAX_DISP];
    float sum = 0;
    for (int i = 0; i < MAX_DISP; i++)
    {
        float coore = 0;
        if (dir != UP)
            coore += grid[(x + 1) * col + y].up_mes[i];
        if (dir != DOWN)
            coore += grid[(x - 1) * col + y].down_mes[i];

        if (dir != LEFT)
            coore += grid[x * col + y + 1].left_mes[i];

        if (dir != RIGHT)
            coore += grid[x * col + y - 1].right_mes[i];
        coore += grid[x * col + y].mat_cost[i];
        switch (dir)
        {
        case UP:
            coore += calCoorelation(i, grid[(x - 1) * col + y].best_disp);
            break;
        case DOWN:
            coore += calCoorelation(i, grid[(x + 1) * col + y].best_disp);
            break;
        case LEFT:
            coore += calCoorelation(i, grid[x * col + y - 1].best_disp);
            break;
        default:
            coore += calCoorelation(i, grid[(x - 1) * col + y + 1].best_disp);
            break;
        }
        sum += coore * coore;
        msgs[i] = coore;
    }
    sum = sqrt(sum);
    for (int i = 0; i < MAX_DISP; i++)
    {
        msgs[i] /= sum;
        if (dir == UP)
            grid[x * col + y].up_mes[i] = msgs[i];
        if (dir == DOWN)
            grid[x * col + y].down_mes[i] = msgs[i];
        if (dir == LEFT)
            grid[x * col + y].left_mes[i] = msgs[i];
        if (dir == RIGHT)
            grid[x * col + y].right_mes[i] = msgs[i];
    }
    delete msgs;
}

double match::iteRate()
{
    double update_rate;
    float change_num = 0;
    for (int i = MAX_DISP; i < row - MAX_DISP; i++)
    {
        for (int j = MAX_DISP; j < col - MAX_DISP; j++)
        {
            sendMsg(i, j, LEFT);
            sendMsg(i, j, RIGHT);
            sendMsg(i, j, UP);
            sendMsg(i, j, DOWN);
        }
    }
    for (int i = MAX_DISP; i < row - MAX_DISP; i++)
    {
        for (int j = MAX_DISP; j < col - MAX_DISP; j++)
        {
            float min_value = 1e20;
            int best_pos = grid[i * col + j].best_disp;
            float dist_a = 0;
            float dist_b = 0;
            bool last_min = false;
            for (int k = 0; k < MAX_DISP; k++)
            {
                float bs = grid[i * col + j - 1].right_mes[k] + grid[i * col + j + 1].left_mes[k] + grid[(i - 1) * col + j].down_mes[k] + grid[(i + 1) * col + j].up_mes[k] + grid[i * col + j].mat_cost[k] * 100;
                if (last_min)
                    dist_b = bs;
                if (min_value > bs)
                {
                    last_min = true;
                    dist_a = min_value;
                    min_value = bs;
                    best_pos = k;
                }
                else
                    last_min = false;
                // std::cout << grid[i * col + j - 1].right_mes[k] << "," << grid[i * col + j + 1].left_mes[k] << "," << grid[(i - 1) * col + j].down_mes[k] << "," << grid[(i + 1) * col + j].up_mes[k] << "," << grid[i * col + j].mat_cost[k] << std::endl;
            }
            // std::cout << dist_a - dist_b << std::endl;
            if (grid[i * col + j].best_disp != best_pos)
            {
                grid[i * col + j].best_disp = best_pos + ((dist_a - dist_b) / (dist_a + dist_b - 2 * min_value) / 2);
                change_num = change_num + 1;
            }
            int k = best_pos;
            // std::cout << grid[i * col + j - 1].right_mes[k] << "," << grid[i * col + j + 1].left_mes[k] << "," << grid[(i - 1) * col + j].down_mes[k] << "," << grid[(i + 1) * col + j].up_mes[k] << "," << grid[i * col + j].mat_cost[k] << std::endl;
        }
    }

    float total_pts = row * col;
    update_rate = change_num / total_pts;
    return update_rate;
}

void match::disImgConvert()
{
    for (int i = 0; i < row; i++)
    {
        uchar *dis_ptr = disparity.ptr<uchar>(i);
        for (int j = 0; j < col; j++)
        {
            dis_ptr[j] = grid[i * col + j].best_disp;
        }
    }
}

void match::ptsConvert(pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat rgb)
{

    cloud.width = col - 2 * KSize;
    cloud.height = row - 2 * KSize;
    cloud.is_dense = true;
    cloud.points.resize(cloud.width * cloud.height);
    int cnt = 0;
    for (int i = KSize; i < row - KSize; i++)
    {
        cv::Vec3b *color_ptr = rgb.ptr<cv::Vec3b>(i);
        uchar *dis_ptr = disparity.ptr<uchar>(i);
        for (int j = KSize; j < col - KSize; j++)
        {
            int dis = -dis_ptr[j];
            float z = 2.51 / (dis * 3e-3 / 50 + 2.51 / 700); // mm
            float y = z * (i - 640) / (2.51 / 0.003);
            float x = z * (j - 400) / (2.51 / 0.003);
            int m = 960 * (x - 16) / z + 337;
            int n = 960 * y / z + 640;
            // if (n < 0 || n >= 1280 || m < 0 || m >= 720)
            //     color_ptr[j] = cv::Vec3b(0, 0, 0);
            // else
            //     color_ptr[j] = rgb.at<cv::Vec3b>(n, m);
            cloud.points[cnt].x = x;
            cloud.points[cnt].y = y;
            cloud.points[cnt].z = z;
            cloud.points[cnt].b = color_ptr[j][0];
            cloud.points[cnt].g = color_ptr[j][1];
            cloud.points[cnt].r = color_ptr[j][2];
            cnt++;
            color_ptr[j][2] = z / 10;
        }
    }
}