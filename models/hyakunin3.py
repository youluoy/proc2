#%%
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
import array
import glob
import librosa
import librosa.display
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import torchvision.models as models

import wave_IO as IO
import wave_VI as VI


waves, srs = IO.load_files_with_grob('../data/inputs/JKspeech/*.wav')
lenw = len(waves[0])
sr = 48000

'''
y = waves[0]
VI.show_wave(y)
VI.show_spec(y, sr)
VI.show_melspec(y, sr)
print()
D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
DB = librosa.amplitude_to_db(D, ref=np.max)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
S_dB = librosa.power_to_db(S, ref=np.max)
print(D.shape)
print(DB.shape)
print(S.shape)
print(S_dB.shape)
y1 = waves[0] + waves[1]
VI.show_wave(y1)
VI.show_spec(y1, sr)
VI.show_melspec(y1, sr)
print()
D = np.abs(librosa.stft(y1, n_fft=2048, hop_length=512))
DB = librosa.amplitude_to_db(D, ref=np.max)
S = librosa.feature.melspectrogram(y=y1, sr=sr, n_fft=2048, win_length=512, hop_length=512)
S_dB = librosa.power_to_db(S, ref=np.max)
print(D.shape)
print(DB.shape)
print(S.shape)
print(S_dB.shape)
'''






#%%
'''
cnt = 0
mixwv = np.zeros((2000, 200000))
isinc = np.zeros((2000))
for i in range(44):
    for j in range(44):
        for k in range(2):
            if j <= i or i <= 1:
                continue
            mixwv[cnt] = waves[i] + waves[j] + waves[k]
            isinc[cnt] = k
            cnt += 1
print(mixwv.shape)
print(cnt)

for i in range(44):
    for k in range(2):
        if i <= 1:
            continue
        mixwv[cnt] = waves[i] + waves[k]
        isinc[cnt] = k
        cnt += 1

print(cnt)

wvspec = np.zeros((cnt,128, 391))
for i in range(cnt):
    S = librosa.feature.melspectrogram(y=mixwv[i], sr=sr, n_fft=2048, win_length=512, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    wvspec[i] = S_dB

print(wvspec.shape)
'''


#%%
wave_num = 88
wave_length = 200000
clip_length = 150**2
clip_length_min = 9600
train_data_num = 2000
valid_data_num = 4000
test_data_num = 5000
mix_limit = 21
target = 0

rndn = np.random.RandomState(123)


#混声データ作成　各音声から長さlenのデータを抜き出してきて足すー＞保存 targetについて混ぜる混ぜないはランダム
def make_mix_data(data_num, target = 0):
    mixed_data = np.zeros((data_num, clip_length))
    label_data = np.zeros((data_num))
    for i in range(data_num):
        mixed_num = rndn.randint(2, mix_limit)
        len = rndn.randint(clip_length_min, clip_length) #一つの混声データ内の各clipは同じ長さ
        #target　について
        is_target_in = rndn.randint(0,2)
        label_data[i] = is_target_in #ラベル保存

        if is_target_in == 1:
            mixed_num -= 1
            while(True):
                beg = rndn.randint(0, wave_length - len) #切り取る場所決め
                clip = waves[target, beg:beg+len]
                if clip[0] == 0 and clip[(len//4)*1] == 0 and clip[(len//4)*2] == 0 or clip[(len//4)*2] == 0 and clip[(len//4)*3] == 0 and clip[len-1] == 0 :
                    continue
                mixed_data[i,0:len] += clip #データ混ぜる
                break

        #target 以外について
        j = 0
        while(j < mixed_num):
            idx = rndn.randint(0, wave_num)
            beg = rndn.randint(0, wave_length - len) #切り取る場所決め
            clip = waves[idx, beg:beg+len]
            if clip[0] == 0 and clip[(len//4)*1] == 0 and clip[(len//4)*2] == 0 or clip[(len//4)*2] == 0 and clip[(len//4)*3] == 0 and clip[len-1] == 0 :
                continue
            mixed_data[i,0:len] += clip #データ混ぜる
            j += 1
    return mixed_data, label_data

train_data, train_label = make_mix_data(train_data_num)
valid_data, valid_label = make_mix_data(valid_data_num)
#test_data,  test_label  = make_mix_data(test_data_num)

print(train_data.shape)
print(train_label.shape)


#%%

#wav:shape(num, cliplength) -> image:shape(num,3,224,224)
def reshape_224x224x3(data, label):
    tmp = np.zeros((len(label), 3, 224, 224))
    tmp[:,:,:150,:150] = data.reshape((len(label), 1, 150, 150))
    #tmp2 = np.zeros((len(label), 2))
    #for i in range(len(label)):
    #    tmp2[i] = np.eye(2)[int(label[i].item() + 10**-15)]
    return tmp, label

def to_melspectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

#wav:shape(num, cliplength) -> melspectr:shape(num,?,?) -> image:shape(num, 3, 224, 224)
def reshape_melspec_224x224x3(data, label):
    tmp = to_melspectrogram(data[0], sr=sr)
    specs = np.zeros((len(data),*tmp.shape))
    for i in range(len(data)):
        mel = to_melspectrogram(data[i], sr=sr)
        specs[i] = mel
    ret = np.zeros((len(label), 3, 224, 224))
    ret[:,:,:specs.shape[1],:specs.shape[2]] = specs.reshape(len(label), 1, specs.shape[1], specs.shape[2])
    return ret, label


#%%
batch_size = 100


#train_data, train_label = reshape_224x224x3(train_data, train_label)
train_data, train_label = reshape_melspec_224x224x3(train_data, train_label)
X_train = torch.tensor(train_data, dtype=torch.float32) #int64 float32
Y_train = torch.tensor(train_label, dtype = torch.int64)
#valid_data, valid_label = reshape_224x224x3(valid_data, valid_label)
valid_data, valid_label = reshape_melspec_224x224x3(valid_data, valid_label)
X_valid = torch.tensor(valid_data, dtype=torch.float32) #int64 float32
Y_valid = torch.tensor(valid_label, dtype = torch.int64)
#X_test = torch.tensor(test_data, dtype=torch.float32) #int64 float32
#Y_test = torch.tensor(test_label, dtype = torch.int64)

Dataset_train = torch.utils.data.TensorDataset(X_train, Y_train)
Dataset_valid = torch.utils.data.TensorDataset(X_valid, Y_valid)
#Dataset_test  = torch.utils.data.TensorDataset(X_test , Y_test)

X_sample, Y_sample = Dataset_train[0]
print(X_sample.size(), Y_sample.size())

#dataloaderの定義
Dataloader_train = torch.utils.data.DataLoader(
    dataset=Dataset_train,
    batch_size = batch_size,
    shuffle = False,
    drop_last = False #len(dataset) % batch_size != 0
)

Dataloader_valid = torch.utils.data.DataLoader(
    dataset=Dataset_valid,
    batch_size = batch_size,
    shuffle = False,
    drop_last = False #len(dataset) % batch_size != 0
)

#Dataloader_test = torch.utils.data.DataLoader(
#    dataset=Dataset_test,
#    batch_size = 20,
#    shuffle = False,
#    drop_last = False
#)


#%%

rng = np.random.RandomState(12345)
random_state = 42
'''
test_net2 = nn.Sequential(
    #nn.Linear(clip_length, 5000), #48000 -> 5000
    #nn.ReLU(),
    #nn.Linear(5000, 500),
    #nn.ReLU(),
    #nn.Linear(500, 50),
    #nn.Sigmoid(),
    #nn.Linear(50, 1),
    #nn.Sigmoid()

    nn.Conv1d(1, 16, 1000, stride=10), # (in,out,kernel)    #48000x1 -> 4701x16
    nn.MaxPool1d(3, stride=2),                              #        -> 2350x16
    nn.Conv1d(16, 32, 100, stride=2),                       #        -> 1126x32
    nn.MaxPool1d(2, stride=2),                              #        -> 563x32
    nn.Conv1d(32, 64, 10, stride=2),                        #        -> 277x32
    nn.MaxPool1d(2, stride=2),                              #        -> 138x32
    nn.Flatten(),
    nn.Linear(138*64, 3000),                                #        -> 5000
    nn.ReLU(),
    nn.Linear(3000, 1000),                                  #        -> 1000
    nn.ReLU(),
    #nn.Dropout(),
    nn.Linear(1000, 100),                                   #        -> 100
    nn.Sigmoid(),
    #nn.Dropout(),
    nn.Linear(100, 1),                                      #        -> 1
    nn.Sigmoid()

    #nn.Conv2d(1,10,30), #128x391x1 -> 99x362x10
    #nn.MaxPool2d(2, stride=2), #  -> 49x181x10
    #nn.Conv2d(10, 20, 10), #        -> 40x172x20
    #nn.MaxPool2d(2, stride=2), #  -> 20x86x20
    #nn.Flatten(),
    #nn.Linear(20*86*20, 1024),
    #nn.ReLU(),
    #nn.Linear(1024, 128),
    #nn.Sigmoid(),
    #nn.Linear(128, 1),
    #nn.Sigmoid()
    
    #nn.Flatten(),
    #nn.Linear(128*391*1, 1024),
    #nn.ReLU(),
    #nn.Linear(1024, 10),
    #nn.Sigmoid(),
    #nn.Linear(10,1),
    #nn.Sigmoid()
    
    
)
def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)

test_net2.apply(init_weights)

'''

test_net2 = models.resnet34(pretrained=True)
#print(test_net2)
#%%
test_net2.fc = torch.nn.Linear(test_net2.fc.in_features, 2)
def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
test_net2.apply(init_weights)


lr = 0.00005
n_epochs = 1000



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"

test_net2.to(device)
optimizer2 = optim.Adam(test_net2.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

history_loss_train = []
history_acc_train  = []
history_loss_valid = []
history_acc_valid  = []



for epoch in range(n_epochs):
    #訓練データの入れ替え
    del(train_data)
    del(train_label)
    del(X_train)
    del(Y_train)
    del(Dataset_train)
    del(Dataloader_train)
    train_data, train_label = make_mix_data(train_data_num)
    #train_data, train_label = reshape_224x224x3(train_data, train_label)
    train_data, train_label = reshape_melspec_224x224x3(train_data, train_label)
    X_train = torch.tensor(train_data, dtype=torch.float32) #int64 float32
    Y_train = torch.tensor(train_label, dtype = torch.int64)
    Dataset_train = torch.utils.data.TensorDataset(X_train, Y_train)
    Dataloader_train = torch.utils.data.DataLoader(
        dataset=Dataset_train,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False #len(dataset) % batch_size != 0
    )

    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    test_net2.train()  #train
    for x, t in Dataloader_train:

        # 勾配の初期化
        test_net2.zero_grad()


        # テンソルをGPUに移動
        #x = x.reshape(x.shape[0], 1, x.shape[1])
        x = x.to(device)
        t = t.to(device)
        #print(t.size())
        # 順伝播
        y = test_net2(x)    
        #y = test_net2.forward(x)
        #y = y.reshape(-1,)

        # 誤差の計算(BCE)
        #y = torch.clamp(y, 10^-15, 1-10^-15)
        #loss = -(t * torch.log(y) + (1 - t) * torch.log(1 - y)).mean()
        #print(y.size())

        loss = loss_function(y, t) #y: one_hot, t: label_number

        # 誤差の逆伝播
        loss.backward()

        # parametanokousin
        optimizer2.step()

        # the prediction is 
        #y = y.reshape(-1,)
        pred = torch.argmax(y, 1)
        #pred = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y))

        losses_train.append(loss.tolist())

        acc = torch.where(t - pred == 0, torch.ones_like(t), torch.zeros_like(t))
        train_num += acc.size()[0]
        train_true_num += acc.sum().item()

    test_net2.eval()  # 評価時eval
    for x, t in Dataloader_valid:

        # テンソルをGPUに移動
        #x = x.reshape(x.shape[0], 1, x.shape[1])
        x = x.to(device)
        t = t.to(device)

        # 順伝播
        y = test_net2(x)
        #y = test_net2.forward(x)
        #y = y.reshape(-1,)

        # 誤差の計算(BCE)
        #loss = -(t * torch.log(y) + (1 - t) * torch.log(1 - y)).mean()
        loss = loss_function(y, t)

        # the prediction is 
        #y = y.reshape(-1,)
        pred = torch.argmax(y,1)
        #pred = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y))

        losses_valid.append(loss.tolist())

        acc = torch.where(t - pred == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_true_num += acc.sum().item()

    #print(epoch)
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}],  train_num: {}'.format(
        epoch,
        np.mean(losses_train),
        train_true_num/train_num,
        np.mean(losses_valid),
        valid_true_num/valid_num,
        train_num
    ))
    history_loss_train.append(np.mean(losses_train))
    history_acc_train.append(train_true_num/train_num)
    history_loss_valid.append(np.mean(losses_valid))
    history_acc_valid.append(valid_true_num/valid_num)


    

#グラフ描画
plt.figure(figsize=(6,6))      #グラフ描画用

plt.plot(range(n_epochs), history_loss_train)
plt.plot(range(n_epochs), history_loss_valid, c='#00ff00')
plt.xlim(0, n_epochs)
plt.ylim(0, 1.3)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.show()
plt.clf()

plt.plot(range(n_epochs), history_acc_train)
plt.plot(range(n_epochs), history_acc_valid, c='#00ff00')
plt.xlim(0, n_epochs)
plt.ylim(0.4, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")
plt.show()
plt.clf()


#%%
#モデルの保存
torch.save(test_net2.state_dict(), "../data/outputs/model_06.pth") #次7 保存済み6

#%%

#混声データ作成　各音声から長さlenのデータを抜き出してきて足すー＞保存
def make_mix_data(data_num, target_num = 20):
    mixed_data = np.zeros((data_num, clip_length))
    label_data = np.zeros((data_num, target_num))
    for i in range(data_num):
        mixed_num = rndn.randint(2, mix_limit)
        len = rndn.randint(clip_length_min, clip_length) #一つの混声データ内の各clipは同じ長さ
        is_used = np.zeros(target_num) #重複なし
        #target 以外について
        j = 0
        while(j < mixed_num):
            idx = rndn.randint(0, wave_num)
            if is_used[idx] == 1:
                continue
            beg = rndn.randint(0, wave_length - len) #切り取る場所決め
            clip = waves[idx, beg:beg+len]
            if clip[0] == 0 and clip[(len//4)*1] == 0 and clip[(len//4)*2] == 0 or clip[(len//4)*2] == 0 and clip[(len//4)*3] == 0 and clip[len-1] == 0 :
                continue
            mixed_data[i,0:len] += clip #データ混ぜる
            label_data[i,idx] = 1
            is_used[idx] = 1
            j += 1
    return mixed_data, label_data

