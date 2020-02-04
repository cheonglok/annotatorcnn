# l_names = ['LB1', 'LB2', 'LB3', 'LB4', 'LB5']
num_epochs = options['max_epochs']
train_split_perc = options['train_split']
batch_size = options['batch_size']

# convert labels to categorical
# y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
# print('before:', y_train.shape)
for label in range(0, 5):
    y_train[label] = keras.utils.to_categorical(y_train[label] == 1,
                                                len(np.unique(y_train[label] == 1)))
#
# print ('x_train_.shape[0]', x_train.shape[0])
# for i in range(0, 5):
#     print('y_train[i] before', y_train[i].shape[0])
# print ('after:',y_train.shape)
perm_indices = np.random.permutation(x_train.shape[0])
train_val = int(len(perm_indices) * train_split_perc)
temp = {}
for label in range(0, 5):
    temp[label] = y_train[label].shape[0]

temp[5] = train_val
train_val_X = min(temp.items(), key=lambda x: x[1])
train_val = train_val_X[1]
# split training and validation
# perm_indices = np.random.permutation(train_val)
# train_val = int(len(perm_indices)*train_split_perc)


# if int(train_val/batch_size) < 10:
#
#     while (int(train_val/batch_size) < 10):
#           batch_size = batch_size / 2
# else:
#      pass
#
# print ("batch size is or has been reduced to:", batch_size)
#
# batch_size = np.int32(batch_size)

# train_val = int(((train_val / batch_size) - (train_val / batch_size) * train_split_perc) * batch_size)
train_val = int(train_val / batch_size)
train_val = int(train_val * batch_size)

train_val = np.int32(train_val)
# the y_data need to be chosen very carefully
# train_val = 2

print("splitting training and validation, splitting index for training:", train_val)
y_train_ = {}
x_train_ = x_train[:train_val]

for label in range(0, 5):
    print('y_train[i] before', y_train[label].shape[0])
    y_train_[label] = y_train[label][:train_val]
    print('y_train_[i] after', y_train_[label].shape[0])
print('training voxel number:', x_train_.shape[0])
##############################################################################
for label in range(0, 5):
    Y_val[label] = keras.utils.to_categorical(Y_val[label] == 1,
                                              len(np.unique(Y_val[label] == 1)))
#
# print ('x_train_.shape[0]', x_train.shape[0])
# for i in range(0, 5):
#     print('y_train[i] before', y_train[i].shape[0])
# print ('after:',y_train.shape)
perm_indicesx = np.random.permutation(X_val.shape[0])
t_val = int(len(perm_indicesx) * train_split_perc)
tempv = {}
for label in range(0, 5):
    tempv[label] = Y_val[label].shape[0]

tempv[5] = t_val
t_val_X = min(tempv.items(), key=lambda x: x[1])
t_val = t_val_X[1]

t_val = int(t_val / batch_size)
t_val = int(t_val * batch_size)

t_val = np.int32(t_val)
# the y_data need to be chosen very carefully
# train_val = 2


print("splitting training and validation, splitting index for validation:", t_val)
# y_train_ = {}
x_val_ = X_val[:t_val]
y_val_ = {}
for label in range(0, 5):
    print('y_val_[i] before', Y_val[label].shape[0])
    y_val_[label] = Y_val[label][:t_val]
    print('y_val_i] after', y_val_[label].shape[0])

print('validation voxel number:', x_val_.shape[0])

# x_val_ = x_train[train_val:]
# y_val_ = {}
# for label in range(0,5):
#     # print ('y_val_[i] before', y_train[i].shape[0])
#     y_val_[label] = y_train[label][train_val:]
#     # print ('y_val_[i] after', y_val_[i].shape[0])
#
# temp = {}
# for label in range(0,5):
#     temp[label] = y_val_[label].shape[0]
#
#
# temp[5] = x_val_ .shape[0]
# t_val_X = min(temp.items(), key=lambda x: x[1])
# t_val = t_val_X[1]
#
# t_val = int(t_val/batch_size)
# t_val = int(t_val * batch_size)
# # the y_val_data need to be chosen very carefully
# # t_val = 4
#
# x_val_ = x_val_[:t_val]
# for i in range(0,5):
#     # print ('y_val_[i] before', y_train[i].shape[0])
#     y_val_[i] = y_val_[i][:t_val]
#     # print ('y_val_[i] after', y_val_[i].shape[0])