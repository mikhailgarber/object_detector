***Simple Object Detection using openCV and YOLO(darknet) Neural Network, YOLO V3 or V4***

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
* names - list of names in the dataset such as included model1.names
* model.config and model.weights - yolov3 or v4 config and weights files from https://github.com/kiyoshiiriemon/yolov4_darknet
* what - name of the image file to read or word "many" to keep running and reading file names from stdin or "-" to read actual image from stdin
* confidence - float number indicating confidence of the detection, for example 0.6





