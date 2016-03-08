//
//  main.cpp
//  biometrics01
//
//  Created by Martin Kiesel on 06/03/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;


const int CROSS_RATIO = 70;
const int SAMPLE_COUNT = 15;
const char* DATA_FILE = "./data-model";
const char* TEST_LABELS_FILE = "./data-labels";
const char* TEST_PATHS_FILE = "./data-paths";

inline int ceilPercent(int i, int p);
inline bool fileExists (const string& file_name);
void saveVector(vector<string> v, const string &file_name);
void loadVector(vector<string> &v, const string &file_name);
void saveVector(vector<int> v, const string &file_name);
void loadVector(vector<int> &v, const string &file_name);
void saveEigenFaces(Ptr<BasicFaceRecognizer> model);

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

int main(int argc, const char * argv[]) {

    //generating random boolean
    random_device rd;
    mt19937 generate(rd());
    bernoulli_distribution distribution(0.5);

    //vector of eigen models
    vector<Ptr<BasicFaceRecognizer>> models(SAMPLE_COUNT);

    for(auto &model : models)
        model = createEigenFaceRecognizer(3);

    const array<string, 11> face_types = {"centerlight", "glasses", "noglasses", "normal", "rightlight", "sad", "sleepy", "surprised", "wink", "leftlight", "happy"};

    const int train_size = ceilPercent(face_types.size(), CROSS_RATIO);
    vector<vector<Mat>> train_images(SAMPLE_COUNT);
    vector<vector<int>> train_labels(SAMPLE_COUNT);

    const int test_size = face_types.size() - train_size;
    vector<vector<Mat>> test_images(SAMPLE_COUNT);
    vector<vector<int>> test_labels(SAMPLE_COUNT);

    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        //converting i to string
        char buffer[8];
        sprintf(buffer, "%02d", i + 1);
        string number(buffer);

        //init sizes for next iteration of loop
        int _train_size = train_size;
        int _test_size = test_size;

        for(const auto &face_type : face_types)
        {
            string path = "./faces/subject" + number + "." + face_type + ".png";
            if ((_train_size > 0 && distribution(generate)) || _test_size == 0)
            {
                //train Mat filling
                train_images[i].push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
                train_labels[i].push_back(i);
                _train_size--;
            }
            else
            {
                //test Mat filling
                test_images[i].push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
                test_labels[i].push_back(i);
                _test_size--;
            }
        }

        models[i]->train(train_images[i], train_labels[i]);
    }


    //tpr

    //fmr = fpr uz mam a tpr = 100 - fmr
    for (int g = 0; g < 10100; g += 100)
    {
        double fnm = 0;
        int fnmc = 0;
        double fm = 0;
        int fmc = 0;

        double tp = 0;

        for (int i = 0; i < SAMPLE_COUNT; i++) {
            models[i]->setThreshold(g);

            for (int j = 0; j < test_size; j++)
            {
                //predict test images known to model
                int p = models[i]->predict(test_images[i][j]);

                if (p == -1)
                {
                    fm++;
                } else
                {
                    tp++;
                }
                fmc++;
            }

            for (int j = 0; j < SAMPLE_COUNT; j++)
            {
                for (int k = 0; k < test_size; k++) {
                    if (i != j)
                    {
                        if (models[i]->predict(test_images[j][k]) != -1)
                        {
                            fnm++;
                        }

                        fnmc++;
                    }
                }
            }
        }

        //cout << g << "," << fm/fmc << "," << fnm/fnmc << endl;
        //cout << fnm/fnmc << "," << tp/fmc << endl;
    }
    return 0;
}


void saveVector(vector<string> v, const string &file_name)
{
    ofstream outf;
    outf.open(file_name);
    ostream_iterator<string> output_iterator(outf, "\n");
    copy(v.begin(), v.end(), output_iterator);
    outf.close();
}

void loadVector(vector<string> &v, const string &file_name)
{
    if (!fileExists(file_name)) abort();
    string line;
    ifstream inf;
    inf.open(file_name);
    if (!inf) abort();
    while (getline(inf, line)) v.push_back(line);
}

void saveVector(vector<int> v, const string &file_name)
{
    ofstream outf;
    outf.open(file_name);
    ostream_iterator<int> output_iterator(outf, "\n");
    copy(v.begin(), v.end(), output_iterator);
    outf.close();
}

void loadVector(vector<int> &v, const string &file_name)
{
    if (!fileExists(file_name)) abort();
    string line;
    ifstream inf;
    inf.open(file_name);
    if (!inf) abort();
    while (getline(inf, line)) v.push_back(atoi(line.c_str()));
}

inline int ceilPercent(int i, int p)
{
    if (i * p == 0) return 0;
    return (i * p + 99)/100;
}

inline bool fileExists (const string& file_name)
{
    struct stat buffer;
    return (stat (file_name.c_str(), &buffer) == 0);
}

void saveEigenFaces(Ptr<BasicFaceRecognizer> model)
{
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();

    for (int i = 0; i < min(10, W.cols); i++) {
        //string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        //cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, 243));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);

        imwrite(format("./eigenface_%d.png", i), norm_0_255(cgrayscale));

    }
}
