***Simple Object Detection in images using openCV and YOLO(darknet) Neural Network, YOLO V3 or V4***

**Install dependencies**

*Mac OS*
```
brew install opencv cmake
```

*Linux*
```
apt install libopencv-dev cmake
```

**Build**

```
cmake . && make
```

**Usage**

```
object_detector names.file model.config model.weights what confidence-float
```
* names, model.config and model.weights - yolov3 or v4 config and weights files such as ones from https://github.com/kiyoshiiriemon/yolov4_darknet
* what - name of the image file to read or word "many" to keep running and reading file names from stdin or "-" to read actual image from stdin
* confidence - float number indicating confidence of the detection, for example 0.6

Example:

```
cat alarm.jpg | ./object_detector model1.names model1.cfg model1.weights - 0.4
```
**Output**

string representing JSON array of objects like:

```
[
    {
        "token" : "person",
        "height" : 123,
        "x" : 53,
        "y" : 8,
        "width" : 59,
        "percent" : 99
    },
    {
        "token" : "person",
        "height" : 91,
        "x" : 101,
        "y" : 7,
        "width" : 28,
        "percent" : 99
    },
    {
        "token" : "car",
        "height" : 22,
        "x" : 277,
        "y" : 0,
        "width" : 41,
        "percent" : 74
    },
    {
        "token" : "dog",
        "height" : 39,
        "x" : 120,
        "y" : 76,
        "width" : 51,
        "percent" : 99
    }
]

```





