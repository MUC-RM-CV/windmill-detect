# 算法设计思路

## 参考资料

> [RoboMaster视觉教程（0）绪论 - Raring_Ringtail - CSDN](https://blog.csdn.net/u010750137/article/details/90698203) -> [RoboMaster视觉教程（7）风车能量机关识别](https://blog.csdn.net/u010750137/article/details/98503529)

## 算法介绍

首先是老生常谈的图像预处理：通道相减提取红色、二值化、开闭运算。

之后就可以提取轮廓了，`cv::findContours` 走起。

然后通过判断轮廓占画幅的比例来决定是否为扇叶（图像大小的 1/1000 ~ 80/1000）。

提取轮廓的特征，然后计算轮廓的特征和事先计算的特征的“距离”，分辨出待击打的叶片。

找到待击打的叶片后，再找到在其中的较小的矩形轮廓，即为要击打的装甲板区域。



## 细节

### 查找轮廓

OpenCV 这个函数其实是可以在查找轮廓时记录层次信息的，但是感觉不大好写，最后便放弃了，选择直接不分层次查找轮廓。层次信息可能用好了会更高效吧。

使用 `cv::minAreaRect` 获得到轮廓的最小包围矩形，然后通过判断矩形的大小，可以判断是否为可能的扇叶。若直接使用`cv::contourArea` 可能会因为点不是顺次相连的而导致面积计算错误。

### 特征提取

判断特征时使用了图像矩特征，并计算 Hu’s Moments （“Moment”即“图像矩”）。因为扇叶旋转时，形状是会变的，而 Hu’s Moments 得到的特征有旋转不变性，因此考虑使用这个来分类可能的叶片轮廓。

但是，现场光照条件不同、图像分辨率不同，形态学运算的有无，以及参数的不同，都会影响到最后的图像矩。因此可能需要根据具体情况调整图像预处理参数，使得图像与预先计算图像矩时使用的图像情况相近；或者重新计算当前条件下的图像矩。

距离的阈值选择也需要实际调试。

### cv::RotatedRect 图像拉直

说到二分类问题，其实一开始是想用支持向量机（SVM）进行分类，这就需要获得扇叶的图像。

但是 `cv::RotatedRect` 包裹的扇叶形状存在旋转角度的问题，拉直后可能会有左右的不同，担心 SVM 不能识别，故没有使用。

不过好像是可以判断出拉直后扇叶的朝向的，比如用主成分分析？

写代码过程中，被 `cv::RotatedRect` 坑了比较久，主要是不清楚 `points` 方法返回的四个点到底是一个怎么样的顺序，以及 `width`、`height` 到底表示哪条边。看了 OpenCV 的文档以及网上的一些资料，也没有很明白，而且好像是错的（可能是太困了）。但是要想正确拉直图像，是必须弄清楚这个的。最后总算弄明白了，看下边这个图：

![cv::RotatedRect](./assets/RotatedRect.svg)

也就是，最左边的点是 0 号点，然后顺时针编号；遇到的第一条边是 `height`，紧接着是 `width`，和长短无关。



明白了之后，拉直图像就轻松了。



```cpp
auto rrect = cv::minAreaRect(contour); // contour is on img

int height = rrect.size.height;
int width = rrect.size.width;

cv::Mat straighten_rect;

Point2f dstPt[4];

if (rrect.size.width > rrect.size.height) {
    straighten_rect = cv::Mat(height, width, CV_8UC1); // rows, cols
    dstPt[0] = Point2f(0, height - 1);          // -> bottom left of dst rect
    dstPt[1] = Point2f(0, 0);                   // -> top left of dst rect
    dstPt[2] = Point2f(width - 1, 0);           // -> top right of dst rect
    dstPt[3] = Point2f(width - 1, height - 1);  // -> bottom right of dst rect
} else {
    straighten_rect = cv::Mat(width, height, CV_8UC1);
    dstPt[0] = Point2f(0, 0);                   // -> top left of dst rect
    dstPt[1] = Point2f(height - 1, 0);          // -> top right of dst rect
    dstPt[2] = Point2f(height - 1, width - 1);  // -> bottom right of dst rect
    dstPt[3] = Point2f(0, width - 1);           // -> bottom left of dst rect
}

cv::Mat M = cv::getPerspectiveTransform(srcPt, dstPt);
cv::warpPerspective(img, straighten_rect, M, straighten_rect.size());
cv::imshow("straighten", straighten_rect); 
```

但是最后用的图像矩，其实也用不到拉直图像，白忙活了……

### 判断旋转矩形之间的相交关系

这个资料真不好找，网上有些是自己实现的，很麻烦，而 OpenCV 其实有相应的函数，但是它的官方文档藏在一个角落里，也没有从 `cv::RotatedRect` 过去的链接。

> 不过有兴趣的话其实可以了解一下怎么实现的，很巧妙；还有前面提到轮廓种的点没有按照顺序时点的排序问题，也很有意思。

`cv::RotatedRect` 的官方文档，介绍了一些基本的成员函数：

[OpenCV: cv::RotatedRect Class Reference](https://docs.opencv.org/master/db/dd6/classcv_1_1RotatedRect.html)

关于 `cv::rotatedRectangleIntersection` 函数的介绍在这个链接的最后（链接里也有一些其他的函数，很有意思）：

[OpenCV: Structural Analysis and Shape Descriptors](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html)

关于这个函数，文档说的应该很明白了，就不再赘述了……嗯，还是说一下吧，怕读者走弯路……

函数有一个返回值，可以直接和 `cv::RectanglesIntersectTypes` 里面的枚举值比较，比较省事。

| Enumerator        | Description                                          |
| ----------------- | ---------------------------------------------------- |
| INTERSECT_NONE    | No intersection.                                     |
| INTERSECT_PARTIAL | There is a partial intersection.                     |
| INTERSECT_FULL    | One of the rectangle is fully enclosed in the other. |

