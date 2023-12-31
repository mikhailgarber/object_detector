/*
Copyright 2021-2024 Mikhail Garber. ALL RIGHTS RESERVED.
See LICENSE for copying
*/
#include <fstream>
#include <sstream>
#include <iostream>

// Required for dnn modules.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;

// Get the names of the output layers

vector<String> getOutputsNames(const Net &net)
{
    vector<String> names;
    // Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();

    // get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    return names;
}

void doOne(Net net, string aFile, float confThreshold)
{

    Mat frame, blob;

    if (aFile == "-")
    {
        // read image from stdin
        vector<uchar> buffer;
        int c;
        while ((c = getchar()) != EOF)
        {
            buffer.push_back(c);
        }

        // Decode the buffer into a Mat object
        frame = imdecode(Mat(buffer), IMREAD_COLOR);
    }
    else
    {
        // read the image from file
        frame = imread(aFile, 1);
    }

    // convert image to blob
    // blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);
    blobFromImage(frame, blob, (float)1 / 255, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    vector<string> outputNames = getOutputsNames(net);
    net.forward(outs, outputNames);

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    // float confThreshold = std::stof(argv[5]);

    float nmsThreshold = 0.4;

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < classIds.size(); i++)
    {
        if (confidences[i] >= confThreshold)
        {
            class2indices[classIds[i]].push_back(i);
        }
    }
    std::vector<Rect> nmsBoxes;
    std::vector<float> nmsConfidences;
    std::vector<int> nmsClassIds;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
    {
        std::vector<Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++)
        {
            localBoxes.push_back(boxes[classIndices[i]]);
            localConfidences.push_back(confidences[classIndices[i]]);
        }
        std::vector<int> nmsIndices;
        NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++)
        {
            size_t idx = nmsIndices[i];
            nmsBoxes.push_back(localBoxes[idx]);
            nmsConfidences.push_back(localConfidences[idx]);
            nmsClassIds.push_back(it->first);
        }
    }
    boxes = nmsBoxes;
    classIds = nmsClassIds;
    confidences = nmsConfidences;

    std::cout << "[" << std::endl;
    for (size_t i = 0; i < classIds.size(); i++)
    {
        std::cout << "    {" << std::endl;
        std::cout << "        \"token\" : \"" << classes.at(classIds[i]) << "\"," << std::endl;
        std::cout << "        \"height\" : " << boxes[i].height << "," << std::endl;
        std::cout << "        \"x\" : " << boxes[i].x << "," << std::endl;
        std::cout << "        \"y\" : " << boxes[i].y << "," << std::endl;
        std::cout << "        \"width\" : " << boxes[i].width << "," << std::endl;
        std::cout << "        \"percent\" : " << (int)(confidences[i] * 100) << std::endl;
        std::cout << "    }";
        if (i == classIds.size() - 1)
        {
            std::cout << std::endl;
        }
        else
        {
            std::cout << "," << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

// driver function
int main(int argc, char **argv)
{
    // USAGE: object_detector names.file model.config model.weights "many" | image-file-name confidence-float
    // "./object_detector model1.names model1.cfg yolov3.weights myimage.jpeg 0.4"
    // or "many" instead of file name to keep reading them from stdin until word "done"
    // "-" instead of file name to read image from stdin

    // get labels of all classes
    string classesFile = argv[1];
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

    // load model weights and architecture
    String configuration = argv[2];
    String model = argv[3];

    // Load the network
    Net net = readNetFromDarknet(configuration, model);

    string fileName;

    if (string(argv[4]) == "many")
    {
        while (true)
        {
            getline(cin, fileName);
            if (fileName == "done")
            {
                break;
            }
            doOne(net, fileName, stof(argv[5]));
        }
    }
    else
    {
        doOne(net, string(argv[4]), stof(argv[5]));
    }
}