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


const int CROSS_RATIO = 80;
const int SAMPLE_COUNT = 16;
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

    //eigen model
    Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer(0);

    const array<string, 11> face_types = {"centerlight", "glasses", "happy", "leftlight", "noglasses", "normal", "rightlight", "sad", "sleepy", "surprised", "wink"};
    const int train_size = ceilPercent(face_types.size(), CROSS_RATIO);
    vector<Mat> train_images;
    vector<int> train_labels;

    const int test_size = face_types.size() - train_size;
    vector<Mat> test_images;
    vector<string> test_images_paths;
    vector<int> test_labels;

    if (fileExists(DATA_FILE) && fileExists(TEST_LABELS_FILE) && fileExists(TEST_PATHS_FILE))
    {
        loadVector(test_labels, TEST_LABELS_FILE);
        loadVector(test_images_paths, TEST_PATHS_FILE);
        if (test_labels.size() != test_images_paths.size()) return -1;

        model->load(DATA_FILE);

        for ( auto &i : test_images_paths ) {
            if (fileExists(i))
                test_images.push_back(imread(i, CV_LOAD_IMAGE_GRAYSCALE));
            else return -1;
        }
    }
    else
    {
        for (int i = 1; i < SAMPLE_COUNT; i++)
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
                string path = "./faces/subject" + number + "." + face_type + ".png";
                if ((_train_size > 0 && distribution(generate)) || _test_size == 0)
                {
                    //train Mat filling
                    train_images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
                    train_labels.push_back(i);
                    _train_size--;
                }
                else
                {
                    //test Mat filling
                    test_images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
                    test_images_paths.push_back(path);
                    test_labels.push_back(i);
                    _test_size--;
                }
            }
        }

        //train model
        model->train(train_images, train_labels);

        //save data for further use
        model->save(DATA_FILE);
        saveVector(test_images_paths, TEST_PATHS_FILE);
        saveVector(test_labels, TEST_LABELS_FILE);
    }


    int predicted = -1;
    //int p = -1;
    double confidence = 0.0;
    //model->predict(test_images[4], p, j);
    //cout << "p: " << p << " j: " << j << endl;
    model->setThreshold(500);
    //model->set("threshold", 0.0);
    for (int i = 0; i < test_images.size(); i++) {
        model->predict(test_images[i], predicted, confidence);
        cout << "predicted: " << predicted << " correct: " << test_labels[i] << " confidence: " << confidence << endl;
    }

    saveEigenFaces(model);

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
    cout << "Total EigenFaces: " << W.cols << endl;

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
