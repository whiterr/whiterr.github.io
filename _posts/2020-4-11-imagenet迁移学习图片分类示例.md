```python
import keras
from keras.applications import Xception
from keras.applications.vgg19 import VGG19
```


```python
model_xception = Xception(weights="imagenet")
model_vgg19 = VGG19(weights='imagenet')
```


```python
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

image = load_img("./ship.jpeg",target_size=(224,224))
plt.imshow(image)

```


![png](https://whiterr.github.io/images/transfer_learning_examples/output_2_1.png)



```python
image = img_to_array(image)
image = np.expand_dims(image,axis = 0)
image.shape
```


```python
import json
from keras.applications import imagenet_utils

preds = model_xception.predict(image)
# P = imagenet_utils.decode_predictions(preds) #去下载imagenet对应的类别表json
with open('./imagenet_class_index.json') as f:  #也可以直接下载好 进行引用
    js = json.load(f)

for p in preds: #遍历每个input
    s= [(js[str(i)],p[i]) for i in p.argsort()[-1:-4:-1]] #输出最大3个
    print(s)
```

    [(['n04127249', 'safety_pin'], 1.0), (['n03775546', 'mixing_bowl'], 4.132528e-20), (['n02319095', 'sea_urchin'], 1.25324e-35)]

