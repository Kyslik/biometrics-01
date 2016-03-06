//
//  main.cpp
//  biometrics01
//
//  Created by Martin Kiesel on 06/03/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include <array>
#include <random>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int CROSS_RATIO = 70;

inline int ceilPercent(int i, int p);

int main(int argc, const char * argv[]) {

    random_device rd;
    mt19937 generate(rd());
    bernoulli_distribution distribution(0.5);

    const array<string, 11> face_types = {"centerlight", "glasses", "happy", "leftlight", "noglasses", "normal", "rightlight", "sad", "sleepy", "surprised", "wink"};

    const int train_size = ceilPercent(face_types.size(), CROSS_RATIO);
    vector<Mat> train_images;
    vector<int> train_labels;

    const int test_size = face_types.size() - train_size;
    vector<Mat> test_images;
    vector<int> test_labels;

    // load images, loop starts from 1 because first image is 01
    for (int i = 1; i < 3; i++)
    {
        //converting i to string
        char buffer[8];
        sprintf(buffer, "%02d", i);
        string number(buffer);
        //init sizes for next iteration of loop
        int _train_size = train_size;
        int _test_size = test_size;

        for(const auto &face_type : face_types)
        {
            if ((_train_size > 0 && distribution(generate)) || _test_size == 0)
            {
                //train Mat filling
                train_images.push_back(imread("./faces/subject" + number + "." + face_type + ".png", CV_LOAD_IMAGE_GRAYSCALE));
                train_labels.push_back(i);
                _train_size--;
            }
            else
            {
                //test Mat filling
                test_images.push_back(imread("./faces/subject" + number + "." + face_type + ".png", CV_LOAD_IMAGE_GRAYSCALE));
                test_labels.push_back(i);
                _test_size--;
            }
        }
    }
    return 0;
}

inline int ceilPercent(int i, int p)
{
    if (i * p == 0) return 0;
    return (i * p + 99)/100;
}
