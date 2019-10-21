import pickle as pkl
import numpy as np
from tqdm import tqdm
from model import ObamaNet
from my_math import my_mean,my_var,my_standardization_fit

with open('data/audio_kp/audio_kp22900_mel.pickle', 'rb') as pkl_file:
	audio_kp = pkl.load(pkl_file)
with open('data/pca20/pkp22902.pickle', 'rb') as pkl_file:
	video_kp = pkl.load(pkl_file)
with open('data/pca20/pca22902.pickle', 'rb') as pkl_file:
	pca = pkl.load(pkl_file)


x, y = [], [] # Create the empty lists
# Get the common keys
keys_audio = audio_kp.keys() #keyはaudioもvideoもtrimごとの名前
keys_video = video_kp.keys()
keys = sorted(list(set(keys_audio).intersection(set(keys_video)))) #set()で集合を表し、intersectionで共通集合を求める
#つまりaudioとvideoの両方にあるデータのみを扱う→エラー防止

# print('Length of common keys:', len(keys), 'First common key:', keys[0])

obamanet=ObamaNet()
look_back=obamanet.look_back
time_delay=obamanet.time_delay


n_videos=400 #使用するtrimed videoの個数。max 22900
print('You use',n_videos,'/',len(keys),'videos')

for key in tqdm(keys[:n_videos]):
	audio = audio_kp[key]
	video = video_kp[key]

	if (len(audio) > len(video)): #データの中身が同じになるようにエラー防止
		audio = audio[0:len(video)]
	else:
		video = video[0:len(audio)]

	start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
	for i in range(start, len(audio)-look_back):
		a = np.array(audio[i:i+look_back]) #look_back分の音声データ
		v = np.array(video[i+look_back-time_delay]).reshape((1, -1)) #look_back先からtime_delay分だけ後ろの画像
		x.append(a)
		y.append(v)

x_ori = np.array(x,dtype=np.float32) #画像数,look_back,audio_keypoints,x_oriは標準化前の全データ
y = np.array(y,dtype=np.float32) #画像数,1,mouth_pca

shapex = x_ori.shape

split0 = int(0.9*shapex[0]) #9割(8割train,1割validation)を用いて標準化。標準化情報を保持してtestにも用いる。
x_train_val=x_ori[:split0]

x_train_val = x_train_val.reshape(-1, shapex[2])#画像数の9割*look_back,26

mean,std=my_standardization_fit(x_train_val) #train,valで標準化のための平均と分散を取得
mean=mean.astype(np.float32)
std=mean.astype(np.float32)

x_ori = x_ori.reshape(-1, shapex[2]) #オリジナルの全データ

x = (x_ori-mean)/std #9割のデータのみから求められた標準化情報ですべてのデータを標準化

x_ori = x_ori.reshape(shapex) #あとで表示するときに便利だからreshape
x = x.reshape(shapex) #画像数,look_back,audio_keypoins
y = y.reshape(y.shape[0], y.shape[2]) #画像数, mouth_pca

split1 = int(0.8*x.shape[0]) #8割をテストデータ
split2 = int(0.9*x.shape[0]) #1割をバリデーション、残りをテストデータ

x_train = x[0:split1]
y_train = y[0:split1]
x_val = x[split1:split2]
y_val = y[split1:split2]
x_test=x[split2:]
y_test=y[split2:]

print('*********************************************************************')
print('audio keypoints shape:',x.shape,'| mouth keypoints shape:',y.shape)
print('used data length:',x.shape[0]//(100*60),'min')

#以下標準化で音声データの値がどうなったかを出力。
print('original all x mean:', my_mean(x_ori.reshape(-1,shapex[2])),'\noriginal all x var:',my_var(x_ori.reshape(-1,shapex[2]),only=True))
print('original x_train_val mean:', my_mean(x_ori[:split0].reshape(-1,shapex[2])),'\noriginal x_train_val var:',my_var(arr=x_ori[:split0].reshape(-1,shapex[2]),only=True))
print('scaled x_train_val mean:', my_mean(x[:split0].reshape(-1,shapex[2])),'\nscaled x_train_val var:',my_var(arr=x[:split0].reshape(-1,shapex[2]),only=True))
print('x_train mean:', my_mean(x_train.reshape(-1,shapex[2])), '\nx_train var:', my_var(x_train.reshape(-1,shapex[2]),only=True))
print('x_val mean:', my_mean(x_val.reshape(-1,shapex[2])), '\nx_val var:', my_var(x_val.reshape(-1,shapex[2]),only=True))
print('x_test mean:', my_mean(x_test.reshape(-1,shapex[2])), '\nx_train var:', my_var(x_test.reshape(-1,shapex[2]),only=True))

print('*********************************************************************')




#obamanet.train(x_train,y_train,x_val,y_val,model_name='obamanet',restore=True) #restoreがFalseのとき初めから学習。Trueのとき途中から学習
#obamanet.test(x_test,y_test,model_name='obamanet')
