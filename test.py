from model import MLP
import os
from utils import load_mnist
import numpy as np
import matplotlib.pyplot as plt

model_dir = './saved_model'
data_dir = './fashion-mnist'
dim_in = 784
dim_hidden = 128
dim_out = 10

model = MLP(dim_in, dim_out, dim_hidden)
test_img, test_label = load_mnist(data_dir, kind='t10k')
model.load(os.path.join(model_dir, 'parameters.npz'))

outputs = model.forward(test_img)
preds = np.argmax(outputs, axis=1)
classes = np.argmax(test_label, axis=1)
accuracy = np.mean(preds == classes)
print(f'Test accuracy:{accuracy}')


input_size = 784  # MNIST数据集输入大小
hidden_size1 = 256
hidden_size2 = 128
output_size = 10  # 10个类别的输出

# 随机生成权重矩阵（这里只是示例，实际应用中应该使用训练后的参数）
weights1 = np.random.randn(input_size, hidden_size1)
weights2 = np.random.randn(hidden_size1, hidden_size2)
weights3 = np.random.randn(hidden_size2, output_size)

# 可视化权重矩阵
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(model.weight1, cmap='viridis', aspect='auto')
axs[0].set_title('Weights 1')

axs[1].imshow(model.weight2, cmap='viridis', aspect='auto')
axs[1].set_title('Weights 2')

axs[2].imshow(model.weight3, cmap='viridis', aspect='auto')
axs[2].set_title('Weights 3')

plt.savefig('vis.png')
plt.tight_layout()
plt.show()
