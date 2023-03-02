# 新メンター技術課題解答
[新メンター技術課題](https://github.com/Geeksalon-AI-Mentor/New_employee_training)の解答です。

## データの収集
データの収集は今回は手作業で集めました。
スクレイピングで集める場合は[selenium](https://www.selenium.dev/ja/documentation/)や[beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)などのスクレイピング用のPythonライブラリを使ってみてください。

集めた画像はこちらのフォルダに入っています。

## データの前処理
> 1. `ImageDataGenerator`インスタンスを生成してください。

ここでは`ImageDataGenerator`クラスでデータを擬似的に増やす処理の設定を行なっています。ImageDataGeneratorは`リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します．また，このジェネレータは，データを無限にループするので，無限にバッチを生成します.`
オプションでは画像に対する処理を指定することができ、下の例では画像に対して以下の処理をしています。
* `width_shift_range`：画像に対してランダムに水平に移動させる。
* `rotation_range`：画像に対してランダムに回転させる。
* `height_shift_range`：画像に対してランダムに垂直に移動させる。
* `zoom_range`：画像に対してランダムに拡大する。
上記オプション以外の詳しい説明は[公式ドキュメント](https://keras.io/ja/preprocessing/image/#imagedatagenerator_1)を見てみてください。

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

> 2. `ImageDataGenerator`クラスの`flow_from_directory`メソッドを用いて学習用のデータと検証用のデータのオブジェクトを生成してください。

ここでは`flow_from_directory`メソッドを使って`ディレクトリへのパスを受け取り，拡張/正規化したデータのバッチを生成します．`
事前にデータを増やすというよりは**学習ごと**に`ImageDataGenerator`で設定した処理が施された画像が生成されると言った感じです。
※設定によっては加工した画像を保存することもできます。

また
`flow_from_directory`の便利なところは正解ラベルを作成してくれる

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


## 転移学習モデルの構築
> 1. [Kerasのドキュメント](https://keras.io/ja/applications/)から好きなモデルを選択してください。
> 2. 選択したモデルをImageNetで学習したモデルに設定してください。
```python
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import experimental
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras import Model

ResNet50 = ResNet50(weights='imagenet',include_top=False, input_tensor=Input(shape=(256,256,3
)))
```

3. 学習済みモデルに[追加層](https://keras.io/ja/layers/core/)を加えてください。
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

完成したモデルを確認するには`summary`メソッドを使います。

```python
model.summary()
```

## コンパイル
> [TensorFlowのドキュメント](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental)から任意の`optimizer`を選択してください。

モデルのコンパイルは[公式ドキュメント](https://keras.io/ja/models/model/#compile)を参考にして作成します。

```
model.compile(optimizer=experimental.SGD(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
```

## 学習
> 1. `epochs`は任意の数字に設定してください。**多すぎると過学習の恐れがあります。必要に応じて[EarlyStopping](https://keras.io/ja/callbacks/#earlystopping)を設定してください。**
> 2. 学習する際に用いるfitメソッドは学習履歴をHistoryオブジェクトを返すので`history`
という変数に格納してください。

```
history = model.fit(
        train,
        epochs=10,
        verbose=1,
        validation_data=validation
)
```

## 分析
> 先ほど格納した`history`というオブジェクトの`history`メソッドを用いて`accuracy`と`val_accuracy`の変化のグラフを`matplotlib`を用いて描写してください。

```
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

![image](https://user-images.githubusercontent.com/86188830/222321469-ce2bf506-7fa7-46cf-af74-28812de38857.png)



## 検証

> 1. 学習したモデルクラスの[predict](https://keras.io/ja/models/model/#predict)メソッドでtestフォルダに入っている画像を予測してください。**画像のままでは使えないので画像をNumpy配列に変換し、作成したモデルの入力のテンソルにリサイズする処理をしてください。**
> 2. 上記処理をtestフォルダ内にあるすべての画像で行い、正答率を算出してください。**正答率は(正解数/testフォルダ内の画像数)で求めることとする。**

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
> `検証`での正答率が8割ほどであればモデルを`save`メソッドで保存してください。**この時モデルの名前に`.h5`をつけてください。**

```python
model.save("<モデル名>.h5")
```
