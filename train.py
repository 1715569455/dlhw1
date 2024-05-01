from utils import load_mnist, SoftmaxCrossEntropy, lr_schedule
import numpy as np
from model import MLP
from copy import deepcopy
import matplotlib.pyplot as plt

data_dir = './fashion-mnist'
batch_size = 1024
max_epoch = 1000
dim_in = 784
dim_hidden = 128
dim_out = 10
max_lr = 1e-3
min_lr = 1e-5
weight_decay = 0
period = 800

X, y = load_mnist(data_dir)
indices = np.arange(len(y))
np.random.shuffle(indices)
train_indices = indices[:50000]
val_indices = indices[50000:60000]
train_images = X[train_indices]
train_labels = y[train_indices]
val_images = X[val_indices]
val_labels = y[val_indices]

model = MLP(dim_in, dim_hidden, dim_out, activ_name='relu')
criterion = SoftmaxCrossEntropy()

max_acc = 0
epoch_list = []
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(max_epoch):
    epoch_list.append(epoch+1)
    lr = lr_schedule(epoch, max_lr, min_lr, period)
    train_outputs = model.forward(train_images)
    train_loss = criterion.forward(train_outputs, train_labels)
    train_loss_list.append(deepcopy(train_loss))
    train_preds = np.argmax(train_outputs, axis=1)
    train_classes = np.argmax(train_labels, axis=1)
    train_accuracy = np.mean(train_preds == train_classes)
    train_acc_list.append(deepcopy(train_accuracy))
    grad_in = criterion.backward()
    model.backward(grad_in)
    model.update(lr, weight_decay)
    val_outputs = model.forward(val_images)
    val_loss = criterion.forward(val_outputs, val_labels)
    val_loss_list.append(deepcopy(val_loss))
    val_preds = np.argmax(val_outputs, axis=1)
    val_classes = np.argmax(val_labels, axis=1)
    val_accuracy = np.mean(val_preds == val_classes)
    val_acc_list.append(deepcopy(val_accuracy))
    print(f'Epoch num:{epoch+1} train loss:{train_loss:.5f} accuracy:{val_accuracy}')
    if val_accuracy > max_acc:
        max_acc = val_accuracy
        model.save('./saved_model')

print('Training end---------------------')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(epoch_list, train_loss_list, label='Training Loss', color='blue')
ax1.plot(epoch_list, val_loss_list, label='Validation Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epoch_list, train_acc_list, label='Training Accuracy', color='blue')
ax2.plot(epoch_list, val_acc_list, label='Validation Accuracy', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

plt.savefig('combined_plot.png')
plt.show()
