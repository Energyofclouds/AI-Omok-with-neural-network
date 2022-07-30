import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import time
import tensorflow as tf


class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        load_model_path = 'C:/Users/bin/Desktop/omok/omok_CNN/model/model.ckpt'

        # 바둑판 크기
        board_size = 15
        self.game_end = 0
        self.board_size = board_size
        self.board = np.zeros([board_size,board_size])
        self.board_history = np.zeros([board_size,board_size])
        self.cnt = 1


        time_now = time.gmtime(time.time())
        self.save_name = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.txt'
        self.save_name_png = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.png'

        
        # read image in numpy array (using cv2)
        board_cv2 = cv2.imread('C:/Users/bin/Desktop/omok/omok_CNN/source/board_1515.png')
        self.board_cv2 = cv2.cvtColor(board_cv2, cv2.COLOR_BGR2RGB)

        white_ball = cv2.imread('C:/Users/bin/Desktop/omok/omok_CNN/source/white.png')
        self.white_ball = cv2.cvtColor(white_ball, cv2.COLOR_BGR2RGB)

        black_ball = cv2.imread('C:/Users/bin/Desktop/omok/omok_CNN/source/black.png')
        self.black_ball = cv2.cvtColor(black_ball, cv2.COLOR_BGR2RGB)

        # numpy to QImage
        height, width, channel = self.board_cv2.shape
        bytesPerLine = 3 * width
        qImg_board = QImage(self.board_cv2.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.player = 1 # 1: 흑  / 2: 백
        x = 0
        y = 0

        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(QPixmap(qImg_board))

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.lbl_img)
        self.setLayout(self.vbox)

        # load AI model
        self.X = tf.placeholder(tf.float32, [None, board_size*board_size])
        X_img = tf.reshape(self.X, [-1, board_size, board_size, 1])
         
        W1 = tf.Variable(tf.random_normal([7,7,1,64], stddev=0.1))
        L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME') 
        L1 = tf.nn.relu(L1)
        #L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
         
        W2 = tf.Variable(tf.random_normal([5,5,64,32], stddev = 0.1))
        L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
        L2 = tf.nn.relu(L2)
        #L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
         
        W3 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
        L3 = tf.nn.relu(L3)
        #L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        W4 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L4 = tf.nn.conv2d(L3, W4, strides=[1,1,1,1], padding='SAME')
        L4 = tf.nn.relu(L4)
        #L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


        W5 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L5 = tf.nn.conv2d(L4, W5, strides=[1,1,1,1], padding='SAME')
        L5 = tf.nn.relu(L5)
        #L5 = tf.nn.dropout(L5, keep_prob=keep_prob)


        W6 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L6 = tf.nn.conv2d(L5, W6, strides=[1,1,1,1], padding='SAME')
        L6 = tf.nn.relu(L6)

        W7 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L7 = tf.nn.conv2d(L6, W7, strides=[1,1,1,1], padding='SAME')
        L7 = tf.nn.relu(L7)

        W8 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L8 = tf.nn.conv2d(L7, W8, strides=[1,1,1,1], padding='SAME')
        L8 = tf.nn.relu(L8)

        W9 = tf.Variable(tf.random_normal([3,3,32,32], stddev = 0.1))
        L9 = tf.nn.conv2d(L8, W9, strides=[1,1,1,1], padding='SAME')
        L9 = tf.nn.relu(L9)
        #L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

        L9 = tf.reshape(L9, [-1, board_size * board_size * 32])

         
        W10 = tf.get_variable("W10", shape=[board_size * board_size * 32, board_size*board_size],initializer = tf.contrib.layers.xavier_initializer())

        b10 = tf.Variable(tf.random_normal([board_size * board_size]))
        self.logits =tf.matmul(L9,W10) + b10
        
    
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.saver.restore(self.sess, load_model_path)

        
        self.setWindowTitle('오목 시뮬레이션')
        self.move(100, 100)
        self.resize(500,500)
        self.show()


    def game_play(self, board_img, ball, pos_x, pos_y, turn):
        #human

        ball_size = ball.shape[0]
        step_size = 56
        off_set = 10

        # 판의 마지막 모서리에는 돌을 두지 못하게 한다.
        if pos_x < step_size/2+off_set+1 or pos_y < step_size/2+off_set+1:
            print('그곳에는 둘 수 없습니다')

        elif pos_x > step_size*self.board_size+step_size/2+off_set or pos_y > step_size*self.board_size+step_size/2+off_set:
            print('그곳에는 둘 수 없습니다')

        else:

            step_x = round((pos_x - off_set)/step_size)
            step_y = round((pos_y - off_set)/step_size)

            if self.board[step_x-1,step_y-1] != 0: # 이미 돌이 있을때
                print('그곳에는 둘 수 없습니다')

            else:
                self.board[step_x-1,step_y-1] = turn
                self.board_history[step_x-1,step_y-1] = self.cnt
                self.cnt = self.cnt + 1
                
                x_step = step_size*step_x-round(step_size/2) + off_set
                y_step = step_size*step_y-round(step_size/2) + off_set
                
                board_img[x_step:x_step+ball_size,y_step:y_step+ball_size] = ball

                # 게임 결과 확인
                if self.game_rule(self.board, turn):
                    self.game_end = 1

                    print('게임이 끝났습니다.')

                    board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
                    

                    print('축하합니다 당신이 승리 하였습니다')


                                  
        return board_img

    def sigmoid(self, x):
        return 1 / (1 +np.exp(-x))

    def find_max(self, result_mat):

        max = 0
        max_x = 0 
        max_y = 0

        for i in range(self.board_size):
            for j in range(self.board_size):
                if result_mat[i,j] > max:
                    max = result_mat[i,j]
                    max_x = i
                    max_y = j

        result_mat[max_x,max_y] = 0
        return result_mat, max_x, max_y       



    def mousePressEvent(self, e):

        x = e.x()
        y = e.y()
        if self.game_end == 0:

            # 흑돌 사람 게임 플레이

            self.board_cv2 = self.game_play(self.board_cv2, self.black_ball, y, x, 1)
           
            save_name =  'result/' + str(self.cnt) + "board_black.png"
            save_name_w = 'result/' + str(self.cnt) + "board_white.png"
            save_name_pred = 'result/' + str(self.cnt) + "board_pred.png"

            

            # 백돌 인공지능 게임 플레이
            input_X = self.board.flatten()/2
            result = self.sess.run(self.logits, feed_dict={self.X: input_X[None,:]})
            result_mat = self.sigmoid(result).reshape([self.board_size,self.board_size])

            heat_map = cv2.resize(result_mat*255, (500, 500))            
            result_mat, max_x, max_y = self.find_max(result_mat)

            save_image = cv2.resize(self.board_cv2, (500, 500), interpolation=cv2.INTER_CUBIC)
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            #save_image[:,:,0] = save_image[:,:,0] + heat_map

            cv2.imwrite(save_name, save_image)
            cv2.imwrite(save_name_pred, heat_map)


            # 인공지능이 판단한 최적의 위치가 만약 둘수 없는 곳이면 다시 계산
            while self.board[max_x,max_y] !=0:
                print('AI 거긴 둘 수 없다')
                result_mat, max_x, max_y = self.find_max(result_mat)


            self.board[max_x,max_y] = 2 # 인공지능은 항상 2값 / 사람은 1값으로 표현
            self.board_history[max_x,max_y] = self.cnt
            self.cnt = self.cnt + 1

            ball_size = self.white_ball.shape[0]
            step_size = 56
            off_set = 10
            x_step = step_size*(max_x+1)-round(step_size/2) + off_set
            y_step = step_size*(max_y+1)-round(step_size/2) + off_set
            self.board_cv2[x_step:x_step+ball_size,y_step:y_step+ball_size] = self.white_ball

             
            save_image = cv2.resize(self.board_cv2, (500, 500), interpolation=cv2.INTER_CUBIC)
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)   
            cv2.imwrite(save_name_w, save_image)
            #self.board_history[step_x-1,step_y-1] = self.cnt
            
            #self.board_cv2 = self.game_play(self.board_cv2, self.white_ball, y, x, 2)
                
           
            height, width, channel = self.board_cv2.shape
            bytesPerLine = 3 * width
            qImg_board = QImage(self.board_cv2.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.lbl_img.setPixmap(QPixmap(qImg_board))



    def game_rule(self, board, player): # 추후 오목 국룰 (렌주룰) 도입 예정
    
        game_result = 0
        diag_line = np.zeros(5)
        
        # ●●●●● 가로 5줄 
        for i_idx in range(len(board)):
            for j_idx in range(len(board)-4):
                p1 = (board[i_idx,j_idx:j_idx+5] == player)
                
                if p1.sum() == 5:
                    #print('player ', player, ' win')
                    game_result = 1
                    return game_result
                    #print(board)

        
        # 세로 5줄 
        for i_idx in range(len(board)-4):
            for j_idx in range(len(board)):
                p1 = (board[i_idx:i_idx+5,j_idx] ==player)
                
                if p1.sum() == 5:
                    #print('player ', player, ' win')
                    game_result = 1
                    return game_result
                    #print(board)
        
        # 대각 5줄 - 1
        for i_idx in range(len(board)-4):
            for j_idx in range(len(board)-4):
                diag_line[0] = board[i_idx+0,j_idx+0]
                diag_line[1] = board[i_idx+1,j_idx+1]
                diag_line[2] = board[i_idx+2,j_idx+2]
                diag_line[3] = board[i_idx+3,j_idx+3]
                diag_line[4] = board[i_idx+4,j_idx+4]    
          
                p1 = (diag_line == player)
                
                if p1.sum() == 5:
                    #print('player ', player, ' win')
                    game_result = 1
                    return game_result
                    #print(board)

        # 대각 5줄 - 반대
        for i_idx in range(len(board)-4):
            for j_idx in range(len(board)-4):
                diag_line[0] = board[i_idx+4,j_idx+0]
                diag_line[1] = board[i_idx+3,j_idx+1]
                diag_line[2] = board[i_idx+2,j_idx+2]
                diag_line[3] = board[i_idx+1,j_idx+3]
                diag_line[4] = board[i_idx+0,j_idx+4]    
          
                p1 = (diag_line == player)
                
                if p1.sum() == 5:
                    
                    game_result = 1
                    return game_result
            
        return game_result

    def save_history(self):


        

        result=np.array(self.board_history).flatten()
        f = open(self.save_name, 'w')


        f.write(np.array2string(result))
        f.close()

   
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())