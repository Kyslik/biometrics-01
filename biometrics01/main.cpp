//
//  main.cpp
//  biometrics01
//
//  Created by Martin Kiesel on 06/03/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    vector<Mat> images;
    vector<int> labels;
    // images for first person
    images.push_back(imread("./faces/subject01.centerlight.gif", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    cout << "wat: " << (bool) images[0].data << endl;
    images.push_back(imread("./faces/subject01.glasses.gif", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    images.push_back(imread("./faces/subject01.happy.gif", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    // images for second person
    images.push_back(imread("./faces/subject02.centerlight.gif", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);

    return 0;
}
