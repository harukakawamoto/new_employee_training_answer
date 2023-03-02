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
* `validation_split`：学習用データと検証用データの割合を指定する。

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
    validation_split=0.1
)
```

> 2. `ImageDataGenerator`クラスの`flow_from_directory`メソッドを用いて学習用のデータと検証用のデータのオブジェクトを生成してください。

ここでは`flow_from_directory`メソッドを使って`ディレクトリへのパスを受け取り，拡張/正規化したデータのバッチを生成します．`

事前にデータを増やすというよりは**学習ごと**に`ImageDataGenerator`で設定した処理が施された画像が生成されると言った感じです。

※設定によっては加工した画像を保存することもできます。

今回は３つの分類(多値分類)のため`class_mode`は`categorical`に指定します。

また`subset`を指定することで学習用データと検証用データを分けてオブジェクトを生成してくれます。`ImageDataGenerator`の`validation_split`で指定した割合ごとに画像を分けてくれます。

詳しいオプションについては[公式ドキュメント](https://keras.io/ja/preprocessing/image/#flow_from_directory)を参考にしてください。

```python
train = gen.flow_from_directory(
    <学習用データフォルダのパス>,
    class_mode='categorical',
    subset = "training"
)

validation = gen.flow_from_directory(
    <学習用データフォルダのパス>,
    class_mode='categorical',
    subset = "validation"
)
```

また`flow_from_directory`の便利なところは**正解ラベルを作成してくれる**機能があるところです。指定したディレクトリのサブディレクトリごとに番号を割り振ってラベル付けしてくれます。
ラベル付けされた番号とサブディレクトリとの対応は生成したオブジェクトの変数の`class_indices`で確認できます。

```python
label_dic = train.class_indices
print(label_dic)

# {'basketball': 0, 'golfball': 1, 'tennisball': 2}
```


## 転移学習モデルの構築
> 1. [Kerasのドキュメント](https://keras.io/ja/applications/)から好きなモデルを選択してください。
> 2. 選択したモデルをImageNetで学習したモデルに設定してください。

ここからモデル構築を行なっていきます。

モデルは任意のものが選べますが、Kerasで提供されているモデルは以下の通りです。
* Xception
* VGG16
* VGG19
* ResNet50
* InceptionV3
* InceptionResNetV2
* MobileNet
* DenseNet
* NASNet
* MobileNetV2

解答例では`ResNet50`を採用していますが、他のモデルを使いたい時はモデルのインポートの部分の`resnet50`の部分をそれぞれのモデル名に変更すれば導入できます。

`weights`に何も指定しなければ重みが0のモデルが導入されますが、`imagenet`を指定することで、ImageNetで学習されたモデルを導入することができます。

ここでは導入したモデルを`ResNet50`という変数に代入しています。

`include_top`はモデルの出力層を含めるかどうかのオプションで、今回は導入した既存モデルに層を加えていくため`False`にします。

出力層を含むとImageNetで学習している都合上1000個の出力が行われることになり、今回の3つの出力(バスケットボール・テニスボール・ゴルフボール)に合わなくなります。

```python
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import experimental
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras import Model

ResNet50 = ResNet50(weights='imagenet',include_top=False, input_tensor=Input(shape=(256,256,3
)))
```

`input_tensor`でモデルに入力する画像の形を指定することができます。

> 3. 学習済みモデルに[追加層](https://keras.io/ja/layers/core/)を加えてください。

ここでは先ほど導入したモデルに層を追加していきます。モデルの出力層は`<モデルの変数名>output`で得ることができます。

追加層は[KerasのFunctionAPI](https://keras.io/ja/getting-started/functional-api-guide/)を使って追加していきます。

`Flatten`は多次元のテンソルを一次元に治すための層です。

`Dense`は全結合層です。

`Dropout`は重みを任意の割合で0にする層です。過学習防止に有効とされています。

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

モデルを学習にはモデルの`fit`メソッドを用います。`fit`メソッドは学習履歴をまとめた`History`オブジェクトを返します。

引数には、学習データのバッチを生成する`train`と学習回数を指定する`epoch`、学習のプログレスバーの表示を指定する`varbose`、最後に検証用データである`validation`を渡しています。

```python
history = model.fit(
        train,
        epochs=10,
        verbose=1,
        validation_data=validation
)
```

> 必要に応じて[EarlyStopping](https://keras.io/ja/callbacks/#earlystopping)を設定してください

EarlyStoppingは学習が進まなくなったら自動的に学習を終了する設定のことです。

導入例は以下の通りです。

```python

```

Historyオブジェクトのhistoryメソッドは`accuracy`と`val_accuracy`をキーとして、各学習履歴を値とした辞書を返します。

## 分析
> 先ほど格納した`history`というオブジェクトの`history`メソッドを用いて`accuracy`と`val_accuracy`の変化のグラフを`matplotlib`を用いて描写してください。

```python
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
