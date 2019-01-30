---
layout: "post"
title: "Elastic distortion"
author: "HJ-harry"
mathjax: true
---


## Data augmentation

**Dataset의 양이 적은 경우**, data augmentation은 classification, semantic segmentation등의 task의 **accuracy를 올리는 데에 도움이 됩니다**. 특히나 Biomedical 분야와 같이 필연적으로 방대한 양의 데이터를 얻기 힘든 상황에서는 어쩔 수 없이 이런 data augmentation에 기대서 조금이라도 accuracy를 올려야 하는 경우가 일반적입니다. 어떤 식으로 data의 양을 늘리는지, 얼마만큼의 양이 적당하고 어떤 방식이 가장 효율적인지에 대한 연구가 활발합니다. 논문을 여러 편 읽어보고 구글링을 해서 다른 사람들이 **일반적으로** 어떻게 data augmentation을 하는지에 대한 내용은 없었습니다. 아직 **어떤 방식이 optimal하다**라고 말할 수 있는 방법이 없고, 상황마다 heuristic하게 정하는 것이 일반적입니다.

## Traditional method -- affine transformation

[**Affine transformation**](https://en.wikipedia.org/wiki/Affine_transformation)이란 **Image의 선형적인 변환입니다**.

$$y = \bf{W}x + \bf{b}$$

로 나타낼 수 있으며, $$x$$는 affine transformation으로 mapping 되기 전 image의 pixel 위치를, $$y$$는 affine transformation으로 mapping된 후 image pixel의 위치를 나타낸다. 이 간단한 식으로 나타낼 수 있는 data augmentation method는 다양합니다. **translation, rotation, skewing** 등은 모두 이 affine transformation 카테고리에 속합니다.  

예를 들어, 가장 간단한 **translation**을 보자면 $$\bf{W} = 0, \bf{b} = [1.75,, -0.5]$$ 인 경우를 들 수 있고, $$\Delta{x} = 1.75, \Delta{y} = -0.5$$ 라고 할 수 있습니다.

![Imgur](https://i.imgur.com/oh7yzNA.png)

$$\bf{W}$$ 와 $$\bf{b}$$ 가 vector of **integers**라고 제한되지 않는다면 mapping되는 y가 integer가 아닐 수 있습니다. 이 경우에는 존재하지 않는 pixel value이므로 주변 pixel value로부터 interpolation을 해서 값을 정하게 되며, interpolation에도 nearest neighbor, spline, bicubic등 여러가지 방식이 있지만 일반적으로 많이 쓰이는 방식은 [**bilinear interpolation**](https://en.wikipedia.org/wiki/Bilinear_interpolation)방식입니다. 이 방식의 python implementation은 다음과 같고, opencv나 scipy와 같은 이미지를 다룰 수 있는 library에는 기본적으로 구현이 되어있습니다.  

```python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)
```
```python
>>> n = [(54.5, 17.041667, 31.993),
         (54.5, 17.083333, 31.911),
         (54.458333, 17.041667, 31.945),
         (54.458333, 17.083333, 31.866),
    ]
>>> bilinear_interpolation(54.4786674627, 17.0470721369, n)
31.95798688313631
```

![Imgur](https://i.imgur.com/VaVpQuF.png)

위 사진은 kaggle competition 중 [Data science bowl 2018]()에 있는 data중 하나로, 여러가지 affine transformation을 위 사진에 적용한 모습을 보여드리겠습니다.


### Flipping
```python
flipped_img = np.fliplr(img)
plt.imshow(flipped_img)
plt.show()
```
![Imgur](https://i.imgur.com/aSnGL4Y.png)
왼쪽, 오른쪽이 뒤집힌 모습입니다.  

### Translation
```python
for i in range(height, 1, -1):
  for j in range(width):
     if (i < height-20):
       img[j][i] = img[j][i-20]
     elif (i < height-1):
       img[j][i] = 0
plt.imshow(img)
plt.show()
```

![Imgur](https://i.imgur.com/7uOHRbp.png)
오른쪽으로 20 pixel만큼 translation된 모습입니다.

### Other Affine transformations
이런식으로 모든 affine transformation은 직접 함수를 만들어서도 적용할 수 있겠지만, opencv에 [**warpAffine**](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)이라는 함수로 구현이 되어있습니다. Mapping되기 전의 점 세 개와, mapping된 이후의 점 세 개만 **getAffineTransform** 함수에 넣어주면 해당 affine transformation에 해당하는 image를 output하게 됩니다.

```python
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```
![Imgur](https://i.imgur.com/tW0acDF.png)

## Elastic Distortion

![Imgur](https://i.imgur.com/xENCs7f.png)

**Elastic distortion**은 기본적인 affine transform에 **probabilistic spin**을 준 것이라고 할 수 있습니다. Affine transform은 linear하기 때문에 설정한 방식으로 이미지가 일정하게 뒤틀리게 됩니다. 반면에 elastic distortion은 pixel별로 image가 다른 방향으로 뒤틀립니다. 조금 더 자연스러운 image의 변화를 가져온다고 할 수 있습니다. 예를들어, 손으로 숫자를 쓸 때 손의 떨림이 있다면 조금씩 손에 떨림이 있다면 이상적인 모양에서 **부분부분별로 흔들림이 있게 됩니다**. 그 흔들림의 방향은 일정하지 않고 다양한 방향으로 일어날 수 있습니다.  

**Biomedical data에도 이러한 elastic distortion은 일반적입니다**. 살아있는 것을 관찰한 data이기 때문에, **순간순간 모습의 변형은 elastic distortion으로 가장 잘 표현이 됩니다**. U-net 에서도 data augmentation을 제시할 때, elastic distortion을 가장 많이 사용했다고 명시하고 있습니다. [U-net tutorial](https://hj-harry.github.io/HJ-blog/2019/01/25/U-Net.html)  

### How does it work?

[Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis](http://cognitivemedium.com/assets/rmnist/Simard.pdf)에서는 Elastic distortion을 어떻게 수행할 수 있는지 나와있습니다. 이 논문에서는 **MNIST dataset**에 elastic distortion을 이용해서 얼마나 accuracy를 높였는지 나와있는데, parameter를 잘 고른다면 biomedical dataset에 충분히 적용 가능합니다.

1) Random displacement vector를 generate합니다.
$$\Delta{x}(x,y) = unif(-1,1), \Delta{y}(x,y) = unif(-1,1)$$
2) parameter중 하나인 **elasticity coefficient**, Gaussian standard deviation $$\sigma$$ 와 convolution을 진행합니다.
3) intensity of deformation을 결정하는 parameter $$\alpha$$ 로 scaling합니다.

어떤 뜻인지 명확하지 않은 분들을 위해, python에서는 [opencv remap function](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=remap)을 이용해서 어떤 식으로 implement 할 수 있는지 보여드리겠습니다.

```python
def elastic_distortion(img, rows, cols, sigma, alpha):
    true_dst = np.zeros((rows,cols,ch))

    # Sampling from Unif(-1, 1)
    dx = np.random.uniform(-1,1,(rows,cols))
    dy = np.random.uniform(-1,1,(rows,cols))

    # STD of gaussian kernel
    sig = sigma

    dx_gauss = cv2.GaussianBlur(dx, (7,7), sig)
    dy_gauss = cv2.GaussianBlur(dy, (7,7), sig)

    n = np.sqrt(dx_gauss**2 + dy_gauss**2) # for normalization

    # Strength of distortion
    alpha = alpha

    ndx = alpha * dx_gauss/ n
    ndy = alpha * dy_gauss/ n

    indy, indx = np.indices((rows, cols), dtype=np.float32)

    # dst_img = cv2.remap(img,ndx - indx_x, ndy - indx_y, cv2.INTER_LINEAR)

    map_x = ndx + indx
    map_x = map_x.reshape(rows, cols).astype(np.float32)
    map_y = ndy + indy
    map_y = map_y.reshape(rows, cols).astype(np.float32)

    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    plt.imshow(dst)
    plt.show()
```

위 함수에서 sigma는 gaussian kernel로 convolution을 진행할 때 얼마나 smoothing을 할지, 즉 **gaussian kernel의 standard deviation**을 조절합니다. 물론 kernel size도 parameter로 넣을 수 있지만 7 * 7로 임의로 고정했습니다. 또 다른 parameter인 **alpha**는 distortion의 세기를 조절합니다. 기존에 sampling을 할 때 Uniform(-1,1)에서 했지만, 여기에다 alpha를 곱함으로써 범위를 늘릴 수 있는 것으로 생각할 수 있습니다.  

parameter를 바꿔가며 여러가지 실험을 해볼 수 있습니다.  

![Imgur](https://i.imgur.com/jMlGz9k.png)

**1. sigma = 4, alpha = 1**

![Imgur](https://i.imgur.com/LTNkYk4.png)

**2. sigma = 4, alpha = 3**

![Imgur](https://i.imgur.com/HtPMT2o.png)

**3. sigma = 10, alpha = 2**  

변형되기 전과 비교했을 때, 꽤나 realistic한 image sample들을 만들어내는 것을 볼 수 있습니다. alpha값이 커지면 distortion되는 것이 심해지기 때문에 이를 modify하기 위해서는 sigma 값을 좀 더 크게해야 realistic한 이미지를 얻을 수 있는 것도 확인할 수 있습니다.

다음 포스트에서는 이것을 실제로 이용한 data augmentation으로 위 데이터셋을 training해보겠습니다. Elastic distortion에 관련된 코드도 곧 github에 업로드 하겠습니다. 혹시 잘못된 부분이나 질문이 있으시면 편하게 댓글로 남겨주세요 :)

## Reference
[1] [Simard et al., Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, 2003](http://cognitivemedium.com/assets/rmnist/Simard.pdf)  

[2] [Ronnenberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
