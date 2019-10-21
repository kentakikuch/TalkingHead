import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import subprocess
import random

class ObamaNet(object):

    '''
    obamanet=ObamaNet()
    obamanet.train(x_train,y_train,x_val,y_val,model_name='obamanet',restore=True) #restoreがFalseのとき初めから学習。Trueのとき途中から学習
    obamanet.test(x_test,y_test)
    こんな感じで実行
    x.shape:(the number of frames,loock_back,the number of audio keypoint)
    y.shape:(the numbe of frames, the number of mouth keypoints)
    '''

    def __init__(self, seq_mouth=20, seq_audio=26, look_back=100, time_delay=20,
					lstm_size=60, num_layers=1, batch_size=100, learning_late=0.0001, gpu_list="0"):
        self.seq_mouth=seq_mouth
        self.seq_audio=seq_audio
        self.look_back=look_back
        self.time_delay=time_delay
        self.lstm_size=lstm_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.learning_rate=learning_late
        self.gpu_list=gpu_list

        self.g=tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver=tf.train.Saver()
            self.init_op=tf.global_variables_initializer()

    def create_batch_generator(self, x, y=None, batch_size=100, shuffle=False):
        if shuffle==True: #x,yを対応付けてシャッフル
            S=list(zip(x,y))
            random.shuffle(S)
            x,y=zip(*S)
            x=np.array(x)
            y=np.array(y)

        n_batches = x.shape[0]//batch_size
        x= x[:n_batches*batch_size] #batch sizeで割り切れる分だけ作成
        if y is not None:
	        y = y[:n_batches*batch_size]
        for ii in range(0, n_batches*batch_size, batch_size):
	        if y is not None:
	            yield x[ii:ii+batch_size], y[ii:ii+batch_size]
	        else:
	            yield x[ii:ii+batch_size]


    def build(self):
        tf_x=tf.placeholder(tf.float32, shape=(self.batch_size, self.look_back, self.seq_audio),name='tf_x')
        tf_y=tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_mouth),name='tf_y')
        tf_keepprob=tf.placeholder(tf.float32, name='tf_keepprob')

        cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                   tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
                   output_keep_prob=tf_keepprob)
                   for i in range(self.num_layers)])

        self.initial_state=cells.zero_state(self.batch_size, tf.float32) #よくわかっていない。もしかしたらlook_backも入れる？
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, tf_x, initial_state=self.initial_state)

        pr_y=tf.layers.dense(inputs=lstm_outputs[:,-1], units=self.seq_mouth, activation=None, name='FC')

        cost=tf.reduce_mean(tf.square(pr_y-tf_y),name='cost')

        optimizer=tf.train.AdamOptimizer(self.learning_rate)

        train_op=optimizer.minimize(cost,name='train_op')




    def train(self, x_train, y_train, x_val, y_val, max_epochs=100, model_name=None, restore=False):

        config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=self.gpu_list, allow_growth=True)) #gpuの指定 allow_growth=Trueは必要になったときのみ指定gpuをその分だけ複数使用する。

        with tf.Session(graph=self.g, config=config) as sess:
            sess.run(self.init_op)


            if restore: #restoreがTrueであれば途中から学習
                learning_model=tf.train.latest_checkpoint("model/"+model_name+"/")
                self.saver.restore(sess, learning_model)
                with open("model/"+model_name+"/learning_info.pickle",'rb') as load_learning_info:
                    learning_info = pkl.load(load_learning_info)
                    iteration = learning_info['iteration']
                    now_epoch = learning_info['epoch']
                    loss_shift = learning_info['loss_shift']
            else: #restoreがFalseであれば最初から学習
                #以下、間違って学習済みmodelを消さないようにするための処理
                if os.path.exists("model/"+model_name+"/"):
                    print("Are you sure to delete model/"+model_name+"/ ?")
                    your_input=input("Yes:y or No:n  ")
                    if your_input=="Yes" or your_input=="y":
                        subprocess.call("rm -r model/"+model_name+"/", shell=True)
                    else:
                        return #yes以外が入力されたときプログラム終了

                learning_info={'iteration':0,'epoch':0,'loss_shift':[]}
                iteration = 0
                now_epoch = 0
                loss_shift=[]

            for epoch in range(now_epoch,max_epochs):
                state = sess.run(self.initial_state)
                batch_loss=[]
                for batch_x, batch_y in self.create_batch_generator(x=x_train, y=y_train, batch_size=self.batch_size, shuffle=True):
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 1.0, self.initial_state : state} #drop_outいらないらしい
                    loss, _, state = sess.run( ['cost:0', 'train_op', self.final_state], feed_dict=feed)
                    batch_loss.append(loss)

                    if iteration % 100 == 0:
                        print("Epoch: %d/%d Iteration: %7d | Train loss: %.8f" % (epoch , max_epochs, iteration, loss))

                    iteration +=1

                if epoch % 1 == 0: #一定エポックごとに保存。1エポックごとだとif文で書く必要ないけどいじれるようにしている。
                    self.saver.save(sess,"model/"+model_name+"/leaned_model.ckpt")
                    learning_info['iteration']=iteration
                    learning_info['epoch']=epoch+1 #epochはループの始まりで更新されるからここで更新しないといけない。
                    loss_shift.append(sum(batch_loss)/len(batch_loss)) #このepochoにおけるtrainのlossの平均を保存。
                    #時間の関係でこうしているが、本来は各エポック学習後にtrainとvalのそれぞれ全部をネットワークに通したのlossを保存する。
                    learning_info['loss_shift']=loss_shift
                    with open("model/"+model_name+"/learning_info.pickle",'wb') as  save_learning_info:
                        pkl.dump(learning_info,save_learning_info)
                    print('Save model')

    def test(self):
        #lossを求めるための関数
        pass




    def predict(self, x_test, y_test, model_name=None, return_proba=False ): #入力音声に対してkeypointsを出力

        config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=self.gpu_list, allow_growth=True)) #gpuの指定 allow_growth=Trueは必要になったときのみ指定gpuをその分だけ複数使用する。

        with tf.Session(graph=self.g, config=config) as sess:
            sess.run(self.init_op)

            learning_model=tf.train.latest_checkpoint("model/"+model_name+"/")
            self.saver.restore(sess, learning_model)
            with open("model/"+model_name+"/learning_info.pickle",'rb') as load_learning_info:
                learning_info = pkl.load(load_learning_info)
                iteration = learning_info['iteration']
                now_epoch = learning_info['epoch']
                loss_shift = learning_info['loss_shift']


            for epoch in range(now_epoch,max_epochs):
                state = sess.run(self.initial_state)
                batch_loss=[]
                for batch_x, batch_y in self.create_batch_generator(x=x_train, y=y_train, batch_size=self.batch_size, shuffle=True):
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 1.0, self.initial_state : state} #drop_outいらないらしい
                    loss, _, state = sess.run( ['cost:0', 'train_op', self.final_state], feed_dict=feed)
                    batch_loss.append(loss)

                    if iteration % 100 == 0:
                        print("Epoch: %d/%d Iteration: %7d | Train loss: %.8f" % (epoch , max_epochs, iteration, loss))

                    iteration +=1

                if epoch % 1 == 0: #一定エポックごとに保存。1エポックごとだとif文で書く必要ないけどいじれるようにしている。
                    self.saver.save(sess,"model/"+model_name+"/leaned_model.ckpt")
                    learning_info['iteration']=iteration
                    learning_info['epoch']=epoch+1 #epochはループの始まりで更新されるからここで更新しないといけない。
                    loss_shift.append(sum(batch_loss)/len(batch_loss)) #このepochoにおけるtrainのlossの平均を保存。
                    #時間の関係でこうしているが、本来は各エポック学習後にtrainとvalのそれぞれ全部をネットワークに通したのlossを保存する。
                    learning_info['loss_shift']=loss_shift
                    with open("model/"+model_name+"/learning_info.pickle",'wb') as  save_learning_info:
                        pkl.dump(learning_info,save_learning_info)
                    print('Save model')
