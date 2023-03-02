# 新メンター技術課題解答
[新メンター技術課題](https://github.com/Geeksalon-AI-Mentor/New_employee_training)の解答です。

## データの収集
データの収集は今回は手作業で集めました。
スクレイピングで集める場合は[selenium](https://www.selenium.dev/ja/documentation/)や[beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)などのスクレイピング用のPythonライブラリを使ってみてください。

集めた画像はこちらのフォルダに入っています。

## データの前処理
1. `ImageDataGenerator`インスタンスを生成してください。

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    fill_mode="constant",
    val
```

2. `ImageDataGenerator`クラスの`flow_from_directory`メソッドを用いて学習用のデータと検証用のデータのオブジェクトを生成してください。

```python
train = gen.flow_from_directory(
    <学習用データフォルダのパス>,
    class_mode='categorical',
    subset = "training"
)

validation = gen.flow_from_directory(
    <テストデータフォルダのパス>,
    class_mode='categorical',
    subset = "validation"
)
```

```python
label_dic = train.class_indices
print(label_dic)
```


## モデルの導入

```python
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import experimental
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras import Model

ResNet50 = ResNet50(weights='imagenet',include_top=False, input_tensor=Input(shape=(256,256,3
)))
```

## 転移学習用に再構築


```python
inputs = ResNet50.output
x = Flatten()(inputs)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.25)(x)

prediction = Dense(3,activation='softmax')(x)

model=Model(inputs=ResNet50.input,outputs=prediction)
```


```python
model.summary()
```
## コンパイル
```
model.compile(optimizer=experimental.SGD(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
```

## 学習

```
history = model.fit(
        train,
        epochs=10,
        verbose=1,
        validation_data=validation
)
```

## 分析
```
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

![image](https://user-images.githubusercontent.com/86188830/222321469-ce2bf506-7fa7-46cf-af74-28812de38857.png)



## テスト

```python
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
```




```python
def get_label(path):
  if "golf" in path:
    return "golfball"
  elif "tennis" in path:
    return "tennisball"
  elif "basket" in path:
    return "basketball"
  else:
    pass
```

```python
root_path = <テストデータフォルダパス>
test_img_list = os.listdir(root_path)
true_num = 0
false_num = 0
for path in test_img_list:
  input = image.load_img(os.path.join(root_path,path),target_size=(256,256))
  input = np.expand_dims(input,axis=0)
  input = preprocess_input(input)
  result = model.predict(input)
  predict_label = list(label_dic.keys())[np.argmax(result[0])]
  acc_label = get_label(path)
  print(predict_label, acc_label)
  # 正答率計算
  if predict_label == acc_label:
    true_num += 1

```

## モデルの保存

```python
model.save("<モデル名>.h5")
```
