---
layout: "post"
title: "[Deep learning]Transposed Convolution"
author: "HJ-harry"
mathjax: true
---

## Transposed Convolution이 필요한 이유

**Semantic segmentation** 에서는 **Encoder-Decoder** 구조를 많이 이용합니다. CNN 아키텍쳐를 통해 image feature를 학습시키고 싶은 경우 다층의 convolutional layer를 활용해서 basic feature부터 점점 더 abstract한 feature를 단계별로 학습하게 됩니다. Kernel을 통한 convolution 연산을 하며 data(image)의 차원이 점점 축소되는데, 이 부분이 encoder부분입니다. Semantic segmentation은 보통 pixel-wise로 label을 정해주어야 하고, 따라서 축소된 차원의 data를 이용하는 것이 아니라 원래 input 이미지의 크기 그대로 data를 다시 늘려주는 과정이 필요합니다. 더 적은 데이터를 이용해서 더 많은 데이터를 **생성** 하는 개념이기 때문에 **interpolation**, 또는 **Upsampling**의 일종으로 볼 수 있고, 이 **Upsamling**을 하는 방법 중 대표적으로 **Transposed Convolution**이 있습니다.  

![U-net Architecture](http://openresearch.ai/uploads/default/original/1X/ec0ac2e2d2df8f213b916453375ccee95a254ac3.png)
*Image from [https://arxiv.org/abs/1505.04597](U-net paper)*

물론, Semantic segmentation에서 대표적으로 Transposed convolution을 사용한다는 것이지 이 task에서만 transposed convolution이 활용되는 것은 아닙니다. Upsampling이 필요한 모든 아키텍쳐에서 Transposed convolution은 자주 사용됩니다.

## Deconvolution, Upconvolution, Transposed convolution...

paper들을 읽다 보면, 또는 인터넷에 검색하다 보면 이 연산에 대한 다양한 이름이 나옵니다. 그 중에서도 이 포스트에서 이용한 이름인 **Transposed** convolution을 사용한 이유도 물론 있습니다. 우선, deconvolution, upconvolution 그리고 transposed convolution은 모두 동일한 연산을 의미합니다. 이로 인해서 헷갈리는 일이 없기를 바랍니다 :)  

하지만 Deconvolution은 잘못된 이름입니다. **쓰는 것을 지양하는 것이 맞습니다.** Convolution은 수학적으로 정의되는 연산이며, 만약 convolution의 역연산을 하는 것이 transposed convolution이 하는 일이라면 deconvolution으로도 부를 수 있을 것입니다. 하지만 transposed convolution은 convolution의 역연산을 하지 않습니다, 아니 하지 못합니다. 어떤 kernel을 통해 convolution이 수행되는 경우 각 pixel에 weight를 곱해서 더해지는 연산들이 이뤄지는데, 이미 차원이 축소된 이후에는 그 이전으로 definite하게 돌아가는 함수를 알 수 없기 때문입니다.

![2D convolution](http://colah.github.io/posts/2014-07-Understanding-Convolutions/img/RiverTrain-ImageConvDiagram.png)

앞으로 이 포스트에서는 Transposed convolution이라는 이름으로 이 연산을 통일할 것이며, 다음으로 왜 Transposed convolution이 맞는 이름이고 정확히 어떤 연산을 Transposed convolution이라고 하는지 알아보도록 하겠습니다.

## 연산 과정
![TC gif](https://i.stack.imgur.com/YyCu2.gif)
Convolution 연산과 관련된 자세한 detail을 알고 싶다면 이 30 페이지 남짓 되는 paper를 보는 것을 추천합니다. 예시 까지 아주 자세히 들어놔서 이해하기도 수월합니다. [All about convolutions](https://arxiv.org/abs/1603.07285)  

위 gif와 이름에서 알 수 있듯이 transposed convolution도 **convolution 연산의 일종**입니다. 밑의 2 x 2 matrix가 input이고, 위의 4 x 4 matrix가 output이죠. 두껍게 한 zero padding을 3 x 3 kernel로 convolution한 것과 같다고 생각할 수 있습니다. 연산을 수학적으로 정의하기 전에, 예시를 먼저 들어보죠.

**Ex1) No zero padding, Unit stride**

- 3 x 3 kernel (k = 3)
- 4 x 4 input (i = 4)
- unit stride (s = 1)

![Imgur](https://i.imgur.com/cOjpHEb.png)

방식은 다음과 같습니다. **Input의 왼쪽 위 부분을 기준으로 보았을 때, 이 부분이 Output의 왼쪽 위 끝 부분으로 mapping되게 하려면** kernel size를 고려해서 바깥쪽으로 2칸씩 zero padding을 수행한 후, 기존의 convolution 방식 그대로 수행하면 됩니다. 이대로 convolution을 진행하면, input의 각 꼭지점에 있는 pixel이 output의 각 꼭지점으로 mapping되는 것을 보실 수 있습니다.  

이제 예시를 하나 들었으니, Transposed convolution이 수학적으로 어떻게 정의되는지 확인하고 가겠습니다. 이를 말하기에 앞서, **Convolution은 두 행렬의 곱으로 나타낼 수 있다**는 점을 아셔야 합니다.

![Imgur](https://i.imgur.com/lq7MRTx.png)

위 그림과 같은 Input에 3 x 3 kernel로 convolution을 하는 연산을 **matrix multiplication으로 나타낼 수 있는데**, Input을 vectorize하고 kernel을 알맞은 matrix로 표현하기만 하면 됩니다. 이 경우에는 **Input vector가 4-d vector**가 될 것이고, output은 4 x 4 matrix를 vectorize한 16-d vector가 됩니다. 그렇다면 kernel의 matrix는 16 x 4 matrix가 될 것입니다.

$$\begin{pmatrix} w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{2,0} & w_{2,1} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 & 0\\ 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{2,0} & w_{2,1} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 \\ 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{2,0} & w_{2,1} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 \\ 0 & 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{2,0} & w_{2,1} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0\end{pmatrix}$$

Input을 output으로 mapping하는 위 convolution kernel을 $$\mathbf{C}$$ 라고 하겠습니다. 위 sparse matrix의 특성상, $$\mathbf{C}$$ 에 input을 곱하면 output이 나오는 것처럼, **$$\mathbf{C^T}$$ 를 output에 곱하면 input이 나옵니다.** 다시 말해, 한 convolution연산에 대한 transposed 된 연산이 transposed convolution입니다.  

Transposed convolution 이름의 이유와 연산방법을 알아보았으니, 다른 예시를 통해 transposed convolution을 어떻게 활용할 수 있는지 알아보도록 하겠습니다.

## Strided transposed convolution

convolution에서 dropout과 유사한 regularization effect를 가지는 strided convolution이 존재하는 것처럼, transposed convolution에도 stride를 unit이 아닌 다른 수로 주는 것이 가능합니다.

![Imgur](https://i.imgur.com/RIjUkyD.png)

기존 convolution에서의 stride를 1이 아닌 다른 숫자로 바꾸는 것은 **input 이미지 크기의 비율과 output이미지 크기의 비율을 결정합니다.** 만약 stride가 2라면 한 pixel씩 건너뛰면서 kernel이 동작하기 때문이죠. 이는 **Downsampling을하는 효과**가 있고, 따라서 최근에는 downsample을 해야 할 때 Maxpooling보다는 strided convolution을 많이 이용합니다. 이 방식의 이점은 DCGAN paper에도 소개가 되어있습니다.

Strided transposed convolution은 **stride의 역수에 해당하는 비율로 Upsampling**을 수행합니다. 예컨대 기존 stride = 2 convolution에서 input대 output의 비율이 2:1이었다면, stride = 2인 transposed convolution에서는 input 대 output의 비율이 1:2입니다. 이러한 이유로, Transposed convolution은 **Fractionally strided convolution**으로 불리기도 합니다.

위 그림과 같이, stride가 unit이 아니라면 input pixel과 pixel 사이에 zero-padding이 들어갑니다. stride = 2라면 한 칸씩, stride = 3이라면 두 칸씩 들어가겠죠. zero padding을 사이사이에 해 준 후 unit-stride에서의 방식처럼 각 꼭지점에 있는 pixel이 input에서 output으로 동일하게 mapping되도록 맞춰둔 후, 그대로 convolution을 적용하면 됩니다.  

**Ex2) No zero padding, Non-unit stride**

- 3 x 3 kernel (k = 3)
- 2 x 2 input (i = 4)
- unit stride (s = 1)
- output size (o)

이 때 output의 크기는 다음과 같은 식으로 표현됩니다.  

$$o = s(i-1) + k$$

이 식을 어디에 활용할 수 있을 까요? 만약 $$s = 2, k = 2$$라면 $$o = 2s$$임을 알 수 있습니다.
다시 말해, 2 x 2 size의 kernel과 stride = 2인 transposed convolution을 이용한다면, size를 두배로 upsampling하는 layer를 만들 수 있다는 뜻이죠. 이는 여러 architecture에서 유용하게 사용될 수 있습니다. Keras로 간단하게 U-net의 구조를 유사하게 만든 코드를 보시겠습니다.

```python
inputs = Input(shape = (128,128,3))
s = Lambda(lambda x: x / 255.) (inputs)

# Depth = 1
encoding_1 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same') (s)
encoding_1 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same') (encoding_1)

# Depth = 2
encoding_2 = MaxPool2D() (encoding_1)
encoding_2 = Conv2D(128, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same') (encoding_2)

# Depth = 3
encoding_3 = MaxPool2D() (encoding_2)
encoding_3 = Conv2D(256, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same') (encoding_3)

# From here we upsample

# Depth = 2
decoding_2 = Conv2DTranspose(128, (2,2), strides = 2, activation = 'relu', kernel_initializer= 'he_normal', padding = 'same') (encoding_3)
decoding_2 = concatenate([encoding_2, decoding_2])
decoding_2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal') (decoding_2)

# Depth = 1
decoding_1 = Conv2DTranspose(64, (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal') (decoding_2)
decoding_1 = concatenate([encoding_1, decoding_1])
decoding_1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal') (decoding_1)
decoding_1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal') (decoding_1)

# Output
outputs = Conv2D(2, (3,3), activation = 'sigmoid', padding = 'same', kernel_initializer= 'he_normal') (decoding_1)

model = Model(inputs = inputs, outputs = outputs)
```
이 모델을 확인하기 위해 summary를 보면,
```python
model.summary()
```
![Imgur](https://i.imgur.com/LyWNNKV.png)

위 식에서 확인할 수 있다시피, Upsampling이 된 것을 확인할 수 있습니다.  

cs231n에서나 stackoverflow에서 사람들이 질문을 꽤 많이 하는 부분이지만, 답변들이 서로 다른 경우가 많아서 좋은 논문을 토대로 하고 제 생각을 첨부하여 작성한 포스트입니다. 누군가에게 도움이 되었으면 좋겠습니다:)

### Reference
**A guide to convolutional arithmetic for deep learning**
[https://arxiv.org/abs/1603.07285]
