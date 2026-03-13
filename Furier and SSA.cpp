// Furier and SSA.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>

#define PI 3.1415926
#define TARGET_HZ 100

static std::vector<std::vector<std::vector<double>>> furier(std::vector<double>& xn);

int main()
{
    std::cout << "Hello World!\n";
    std::vector<double> signal = std::vector<double>();

    for (int i = 1; i <= 100; i++) {
        signal.push_back(cos(i * PI / 2) + cos(i * PI / 3));
    }
    furier(signal);
}

static std::vector<std::vector<std::vector<double>>> furier(std::vector<double>& xn) {
    std::vector<std::vector<std::vector<double>>> result;
    result.push_back(std::vector<std::vector<double>>()); // cos
    result.push_back(std::vector<std::vector<double>>()); // sin

    for (int k = 1; k <= TARGET_HZ; k++) {
        result[0].push_back(std::vector<double>());
        result[1].push_back(std::vector<double>());
        double result_cos = 0;
        double result_sin = 0;

        for (int i = 0; i < xn.size(); i++) {
            result_cos += xn[i] * cos((2 * PI * k * i) / xn.size());
            result_sin += xn[i] * sin((2 * PI * k * i) / xn.size());
        }

        printf("%d hz, |Xn|/n = %f, phase = %f\n", k, 2 * sqrt(result_cos * result_cos + result_sin * result_sin) / xn.size(), atan2(result_sin, result_cos));
        result[0].back().push_back(result_cos);
        result[1].back().push_back(result_sin);
    }
    return result;
}
