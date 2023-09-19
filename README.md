# Canny-Edge-Detector

A script (so far) that detects the major edges in an image! This will eventually turn into a webtool that (hopefully) anyone would be able to use with more functionality than just edge detection. I am currently looking into Harris Corner detection, image sharpening, and other computer vision algorithms I am currently learning and looking to learn.

# Demo work:
Featuring some images I've processed and visible evidence of the script doing something. There are 5 steps to the process:
- Blurring image to reduce noise
- Gradient intensities of image (looks the coolest in my opinion)
- Non-maximum suppression to thin edges
- Double threshold to classify edge significance
- Hysteresis to finally determine major edges

2 demos:

[Simple Cat](#simple-cat)

[Cat](#cat)

## Simple Cat
### Input
![cat](simple_cat.jpg)
### Blurred
![blurred_simple_cat](simple_cat/blurred_simple_cat_G_5x5.png)
### Gradient Intensity
![gi_simple_cat](simple_cat/gi_simple_cat_G_5x5.png)
### Non-Maximum Suppression
![nms_simple_cat](simple_cat/nms_simple_cat_G_5x5.png)
### Double Threshold
![dt_simple_cat](simple_cat/dt_simple_cat_G_5x5.png)
### Hysteris (Final)
![hys_simple_cat](simple_cat/hys_simple_cat_G_5x5.png)
## Cat
### Input
![cat](cat.png)
### Blurred
![blurred_cat](cat/blurred_cat_G_5x5.png)
### Gradient Intensity
![gi_cat](cat/gi_cat_G_5x5.png)
### Non-Maximum Suppression
![nms_cat](cat/nms_cat_G_5x5.png)
### Double Threshold
![dt_cat](cat/dt_cat_G_5x5.png)
### Hysteris (Final)
![hys_cat](cat/hys_cat_G_5x5.png)
