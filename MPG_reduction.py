import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import os 


"""
def raw_data_download():
    # 下载路径
    dataset_path = tf.keras.utils.get_file('auto-mpg.data',
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    data2csv = pd.read_csv(dataset_path)
    data2csv.to_csv('./Raw_data.csv', index=False)
    return None

# def preprocess()
raw_data_download()

def data_shave(datapath):
    # 数据修剪
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    raw_data = pd.read_csv(datapath, names=column_names, comment='\t', sep=' ', skipinitialspace=True, na_values='?', header=None)
    # 查看是否有NAN
    print(str(raw_data.isna().sum()) + '\n')
    # 丢弃nan值
    raw_data = raw_data.dropna()
    # 再次查看
    print(str(raw_data.isna().sum()))
    # 不需要自动创建index
    raw_data.to_csv('./MPG.csv', index=False)

    return None
data_shave('./Raw_data.csv')
"""

def preprocess():
    """
    预处理数据
    """

    datapath = './MPG.csv'
    dataset = pd.read_csv(datapath)

    # 数据集处理,将origin这一列类别型数据移除,更换成产地数据
    origin = dataset.pop('Origin')
    # dataset['USA'] = (origin == 1) * 1
    # dataset['Europe'] = (origin == 2) * 1
    # dataset['Japan'] = (origin == 3) * 1


    # 划分训练集和测试集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    # 移除训练测试MPG标签
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    
    def norm(set, stats):
        """
        进行归一化
        """
        return (set - stats['mean']) / stats['std']
    
    # 获得摘要
    train_stats = train_dataset.describe()
    train_stats = train_stats.T
    test_stats = test_dataset.describe()
    test_stats = test_stats.T
    # 获得归一化的训练集和测试集
    normed_train = norm(train_dataset, train_stats)
    normed_test = norm(test_dataset, test_stats)

    return (normed_train, train_labels), (normed_test, test_labels)

def get_db():
    """
    打包训练数据
    """
    (train_dataset, train_labels), (test_dataset, test_labels) = preprocess() 
    train_db = tf.data.Dataset.from_tensor_slices((train_dataset.values, train_labels.values))
    # test_db = tf.data.Dataset.from_tensor_slices((test_dataset.values, test_labels.values))
    train_db = train_db.shuffle(100).batch(32)
    # test_db = test_db.shuffle(100).batch(32)
    return train_db, test_dataset, test_labels

# 检查正确
# train_db, test_db = get_db()
# print(next(iter(train_db))[0].shape)
# print(next(iter(train_db))[1].shape)

# 创建神经网络类
# class MyModel(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.f1 = keras.layers.Dense(64, activation='relu')
#         self.f2 = keras.layers.Dense(64, activation='relu')
#         self.f3 = keras.layers.Dense(1)

#     def call(self, input):
#         x = self.f1(input)
#         x = self.f2(x)
#         y = self.f3(x)
#         return y

def build_model():

    model_savepath = './model'
    if os.path.exists(model_savepath):
        print('\n=========================Loading======================')
        model = tf.keras.models.load_model(model_savepath)
        model.summary()
    else:
        print('\n============================Saving========================')
        # 创建网络
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.build(input_shape=(4, 6))
        model.summary()
        
    return model

def train(model, train_db, test_dataset, test_labels, epochs=5, isprint=True, isplot=True):
    train_losses, test_losses = [], []

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                out = model(x)
                l = loss(y, out)

            grads = tape.gradient(l, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if isprint and step % 10 == 0:
                print('Epoch:',epoch, 'Step:', step, 'Loss:', l.numpy())
                
        train_losses.append(l)
        test_losses.append(loss(test_labels, out))

    if isplot:
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label='Train MSE')
        plt.plot(test_losses, label='Test MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MSE Loss Curves')
        plt.legend()
        # plt.show()

    # 保存模型
    model.save('./model')
    return None

# def predict(model, test_dataset):

#     # 训练完成得到预测输出
#     pred = model(tf.constant(test_dataset.values))
#     return pred

def predict(model, test_dataset, test_labels):
    """
    预测分别打包绘制
    """
    title_list = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']

    pred = model(tf.constant(test_dataset.values))
    plt.subplot(3, 3, 1)
    plt.plot(pred, label='Pred')
    plt.plot(test_labels.values, label='True')
    plt.legend()
    plt.xlabel('Num')
    plt.ylabel('MPG')
    plt.title('Full')

    for i in range(len(title_list)):
        # 相应列置零
        dataset = test_dataset[:]
        dataset[title_list[i]] = 0
        print(dataset)
        # 训练完成得到预测输出
        pred = model(tf.constant(dataset.values))

        plt.subplots_adjust(wspace=0.2, hspace=0.7)
        plt.subplot(3, 3, i+2)
        plt.plot(pred, label='Pred')
        plt.plot(test_labels.values, label='True')
        plt.legend()
        plt.xlabel('Num')
        plt.ylabel('MPG')
        plt.title(title_list[i])

    plt.show()
    return None


def predict_v2(model, test_dataset, test_labels):
    """
    预测绘制总图
    """
    title_list = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']

    pred = model(tf.constant(test_dataset.values))
    plt.figure(figsize=(10, 8))
    plt.plot(pred, label='Full')
    plt.plot(test_labels.values, label='True')
    plt.xlabel('Num')
    plt.ylabel('MPG')
    for i in range(len(title_list)):
        # 相应列置零
        dataset = test_dataset[:]
        dataset[title_list[i]] = 0
        print(dataset)
        # 训练完成得到预测输出
        pred = model(tf.constant(dataset.values))
        plt.plot(pred, label=title_list[i])
        plt.legend()
    plt.show()
    return None

def predict_v1(model, test_dataset, test_labels):
    """
    分别预测
    """

    pred = model(tf.constant(test_dataset.values))

    plt.plot(pred, label='Pred')
    plt.plot(test_labels.values, label='True')
    plt.legend()
    plt.xlabel('Num')
    plt.ylabel('MPG')
    plt.show()

def main():
    # 获得数据集
    train_db, test_dataset, test_labels = get_db()

    # 创建模型
    model = build_model()
    tf.keras.utils.plot_model(model, to_file='./model/shape.png' ,show_shapes=True)
    # 训练
    # train(model, train_db, test_dataset, test_labels, epochs=200)

    # 处理测试数据,将气缸数Cylinders置零
    # test_dataset['Cylinders'] = 0
    # test_dataset['Displacement'] = 0
    # test_dataset['Horsepower'] = 0
    # test_dataset['Weight'] = 0
    # test_dataset['Acceleration'] = 0
    # test_dataset['Model Year'] = 0
    # print(test_dataset)

    # 预测
    predict(model, test_dataset, test_labels)
    # predict_v1(model, test_dataset, test_labels)
    
    return None

if __name__ == '__main__':
    main()