import tensorflow as tf
import random
import os
import numpy as np


def readfile(file_name_path):

    board_vec = np.zeros([board_size*board_size],dtype='f')

    f = open(file_name_path, 'r')
    cnt = 0
 
    # 오목 학습데이터 (기보) 읽기
    while True:
        line = f.readline()
        if not line: break
        split_ini = line.split('.')

        for i in range(len(split_ini)):

            split_fur = split_ini[i].split(' ')

            if board_size*board_size <= cnt:
                break;

            elif split_fur[-1] != '\n':
                board_vec[cnt] = split_fur[-1]
                cnt = cnt + 1
    f.close()

    return board_vec


def data_aug(board_data): # 1개의 데이터를 회전, 반전으로 12개로 증강
    board_stack = []

    board_mat = board_data.reshape([board_size,board_size])
    
    # rotate
    board_stack.append(board_mat)
    board_stack.append(np.rot90(board_mat))
    board_stack.append(np.rot90(board_mat,2))
    board_stack.append(np.rot90(board_mat,3))

    board_fliplr = np.fliplr(board_mat)
    board_flipud = np.flipud(board_mat)
    
    # rotate and filp
    board_stack.append(board_fliplr)
    board_stack.append(np.rot90(board_fliplr))
    board_stack.append(np.rot90(board_fliplr,2))
    board_stack.append(np.rot90(board_fliplr,3))

    board_stack.append(board_flipud)
    board_stack.append(np.rot90(board_flipud))
    board_stack.append(np.rot90(board_flipud,2))
    board_stack.append(np.rot90(board_flipud,3))

    return board_stack


### set parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
board_size = 15
file_path = 'C:/Users/bin/Desktop/omok/omok_MLP_5_100/training_data/txt/' # 학습 기보 저장된 폴더
save_file = 'C:/Users/bin/Desktop/omok/omok_MLP_5_100/model/model.ckpt' # 모델 저장 폴더 / 파일명

### 학습 데이터 저장 공간
board_x_stack = []
board_y_stack = []

keep_prob = tf.placeholder(tf.float32)


### set network
X = tf.placeholder(tf.float32, [None, board_size*board_size])
X_img = tf.reshape(X, [-1, board_size, board_size, 1])
Y = tf.placeholder(tf.float32, [None, board_size*board_size])
 
W2 = tf.get_variable("W2", shape=[board_size * board_size , 100],initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]))

Z2 = tf.matmul(X,W2) + b2
A2= tf.nn.relu(Z2)

W3 = tf.get_variable("W3", shape=[100 , 100],initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([100]))

Z3 = tf.matmul(A2,W3) + b3
A3= tf.nn.relu(Z3)

W4 = tf.get_variable("W4", shape=[100 , 100],initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([100]))

Z4 = tf.matmul(A3,W4) + b4
A4= tf.nn.relu(Z4)

W5 = tf.get_variable("W5", shape=[100 , 100],initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([100]))

Z5 = tf.matmul(A4,W5) + b5
A5= tf.nn.relu(Z5)

W6 = tf.get_variable("W6", shape=[100 , 100],initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([100]))

Z6 = tf.matmul(A5,W6) + b6
A6= tf.nn.relu(Z6)

W7 = tf.get_variable("W7", shape=[100 , board_size*board_size],initializer = tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([board_size*board_size]))

logits = tf.matmul(A6,W7) + b7

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 

# data load / augmentation
files = os.listdir(file_path)

for file_name in files:
    file_name_path = os.path.join(file_path,file_name)
    
    # read data
    board_data = readfile(file_name_path)
    # data augmentation
    board_stack = data_aug(board_data)

    for board_i in range(len(board_stack)):
        board_vec = board_stack[board_i].flatten()
        max_step = np.max(board_vec)

        for data_cnt in range(1,int(round(max_step*0.8))):
            board_x = np.zeros([board_size*board_size],dtype='f')
            board_y = np.zeros([board_size*board_size],dtype='f')
            
            for i in range(len(board_vec)):

                ############# 흑/백 학습 데이터 생성
                # 학습시에는 어느턴이든 항상 자기 돌은 1 상대돌은 0.5 으로 표현하게 한다
                if data_cnt%2 !=0:
                    if board_vec[i] > 0 and board_vec[i] <= data_cnt: # 배경 skip
                        if board_vec[i]%2 != 0:
                            # 흑
                            board_x[i] = 1
                        else:
                            board_x[i] = 2

                    if board_vec[i] == data_cnt + 1: 
                        board_y[i] = 1

                else:
                    if board_vec[i] > 0 and board_vec[i] <= data_cnt: # 배경 skip
                        if board_vec[i]%2 == 0:
                            # 백
                            board_x[i] = 1
                        else:
                            board_x[i] = 2

                    if board_vec[i] == data_cnt + 1: 
                        board_y[i] = 1     
                
            if (np.sum(board_y)) != 1:
            	print('학습데이터에 문제가 있습니다.')
            
            board_x_stack.append(board_x/2)
            board_y_stack.append(board_y)


total_batch = (round(len(board_x_stack)/100)) - 1
print('학습데이터 생성이 완료 되었습니다: ', total_batch)
######### 학습 시작
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
 
print("Learning started, It takes sometime.")
for epoch in range(training_epochs):
    avg_cost = 0
    
    batch_xs = np.zeros([batch_size,board_size*board_size],dtype='f')
    batch_ys = np.zeros([batch_size,board_size*board_size],dtype='f')
    
 
    for i in range(total_batch):
        
        for j in range(batch_size):

            batch_xs[j,:] = board_x_stack[j+i*batch_size]
            batch_ys[j,:] = board_y_stack[j+i*batch_size]
        
        
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
 
    print('Epoch;', '%04d' % (epoch +1), 'cost =', '{:.9f}'.format(avg_cost))
 
print('학습이 완료 되었습니다.')



######### 학습 모델 저장
saver = tf.train.Saver()
saver.save(sess, save_file)


print('모델저장이 완료 되었습니다.')