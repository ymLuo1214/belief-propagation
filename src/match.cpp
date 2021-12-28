#include "match.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
match::match(const cv::Mat &ref, const cv::Mat &test)
{
    // img_L = preProcess(ref, true);
    // img_R = preProcess(test, true);
    img_L = ref;
    img_R = test;
    grid.resize(col * row);
    disparity = cv::Mat(row, col, CV_8UC1);
    for (u_int i = 0; i < col * row; i++)
    {
        grid[i].left_mes = new float[MAX_DISP];
        grid[i].right_mes = new float[MAX_DISP];
        grid[i].up_mes = new float[MAX_DISP];
        grid[i].down_mes = new float[MAX_DISP];
        grid[i].mat_cost = new float[MAX_DISP];
        grid[i].best_disp = 0;
        for (int j = 0; j < MAX_DISP; j++)
        {
            grid[i].mat_cost[j] = 0;
            grid[i].up_mes[j] = 0;
            grid[i].down_mes[j] = 0;
            grid[i].left_mes[j] = 0;
            grid[i].right_mes[j] = 0;
        }
    }
    for (u_int x = MAX_DISP + KSize; x < row - MAX_DISP - KSize; x++)
    {
        for (u_int y = MAX_DISP + KSize; y < col - MAX_DISP - KSize; y++)
        {
            for (int d = 0; d < MAX_DISP; d++)
            {
                grid[x * col + y].mat_cost[d] = matchCost(x, y, d);
            }
            // std::cout << x << "," << y << std::endl;
        }
    }
    std::cout << "Initialize finished~~" << std::endl;
}

// void match::destoryCost()
// {
//     for (u_int i = 0; i < col * row; i++)
//     {
//         delete grid[i].mat_cost;
//         delete grid[i].up_mes;
//         delete grid[i].down_mes;
//         delete grid[i].left_mes;
//         delete grid[i].right_mes;
//     }
// }

cv::Mat preProcess(const cv::Mat &mat, bool clear)
{
    cv::Mat res = mat.clone();
    for (int r = 8; r < row - 8; r++)
    {
        uchar *res_ptr = res.ptr<uchar>(r);
        for (int c = 8; c < col - 8; c++)
        {
            float sum = 0;

            float squareSum = 0;

            for (int i = -8; i <= 8; i++)
            {
                const uchar *mat_ptr = mat.ptr<uchar>(r + i);
                for (int j = -8; j <= 8; j++)
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
    for (int i = -KSize; i <= KSize; i++)
    {
        // const uchar *ref_ptr = img_L.ptr<uchar>(r1 + i);
        // const uchar *mat_ptr = img_R.ptr<uchar>(r1 + i);
        const float *ref_ptr = img_L.ptr<float>(r1 + i);
        const float *mat_ptr = img_R.ptr<float>(r1 + i);
        for (int j = -KSize; j <= KSize; j++)
        {
            if (ref_ptr[c1 + j] == 0)
                continue;
            else
            {
                float ref_ij = static_cast<float>(ref_ptr[c1 + j]);
                cost += ref_ij * log(ref_ij / (1e-3 + mat_ptr[c1 - dist + j]));
            }
        }
    }
    return cost;
}

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
        for (int j = 0; j < MAX_DISP; j++)
        {
            coore += (calCoorelation(i, j) / 25);
        }
        // switch (dir)
        // {
        // case UP:
        //     coore += calCoorelation(i, grid[(x - 1) * col + y].best_disp);
        //     break;
        // case DOWN:
        //     coore += calCoorelation(i, grid[(x + 1) * col + y].best_disp);
        //     break;
        // case LEFT:
        //     coore += calCoorelation(i, grid[x * col + y-1].best_disp);
        //     break;
        // default:
        //     coore += calCoorelation(i, grid[(x - 1) * col + y+1].best_disp);
        //     break;
        // }
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
            for (int k = 0; k < MAX_DISP; k++)
            {
                float bs = grid[i * col + j - 1].right_mes[k] * grid[i * col + j + 1].left_mes[k] + grid[(i - 1) * col + j].down_mes[k] + grid[(i + 1) * col + j].up_mes[k] * grid[i * col + j].mat_cost[k];
                if (min_value > bs)
                {
                    min_value = bs;
                    best_pos = k;
                }
            }
            if (grid[i * col + j].best_disp != best_pos)
            {
                grid[i * col + j].best_disp = best_pos;
                change_num = change_num + 1;
            }
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