import sys
import numpy as np
import cv2
import time
import os

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        # 바둑판 크기
        board_size = 15
        self.game_end = 0
        self.board_size = board_size
        self.board = np.zeros([board_size,board_size])
        self.board_history = np.zeros([board_size,board_size])
        self.cnt = 1


        root_path = 'C:/Users/bin/Desktop/omok/omok_MLP_5_100/training_data'
        time_now = time.gmtime(time.time())
        self.save_name = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.txt'
        self.save_name_png = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.png'

        self.save_name = os.path.join(root_path, self.save_name)
        self.save_name_png = os.path.join(root_path, self.save_name_png)
        
        # read image in numpy array (using cv2)
        board_cv2 = cv2.imread('C:/Users/bin/Desktop/omok/omok_MLP_5_100/source/board_1515.png')
        self.board_cv2 = cv2.cvtColor(board_cv2, cv2.COLOR_BGR2RGB)

        white_ball = cv2.imread('C:/Users/bin/Desktop/omok/omok_MLP_5_100/source/white.png')
        self.white_ball = cv2.cvtColor(white_ball, cv2.COLOR_BGR2RGB)

        black_ball = cv2.imread('C:/Users/bin/Desktop/omok/omok_MLP_5_100/source/black.png')
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

        self.setWindowTitle('오목 시뮬레이션')
        self.move(100, 100)
        self.resize(500,500)
        self.show()


    def game_play(self, board_img, ball, pos_x, pos_y, turn):

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
                
                y_step = step_size*step_x-round(step_size/2) + off_set
                x_step = step_size*step_y-round(step_size/2) + off_set
                
                board_img[y_step:y_step+ball_size,x_step:x_step+ball_size] = ball

                # 게임 결과 확인
                if self.game_rule(self.board, turn):
                    self.game_end = 1

                    print('게임이 끝났습니다.')

                    board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.save_name_png, board_img)

                    if turn == 1: # 흑백 턴 교차
                        print('흑 승리')
                        self.save_history()

                    else:
                        print('백 승리')
                        self.save_history()

                else: # 게임이 끝나지 않았다면, 턴을 바꾸어 주고 계속 진행

                    if turn == 1: # 흑백 턴 교차
                        self.player = 2
                    else:
                        self.player = 1
                                  
        return board_img


    def mousePressEvent(self, e):

        x = e.x()
        y = e.y()
        if self.game_end == 0:
            if self.player == 1: # 흑돌 차례

                self.board_cv2 = self.game_play(self.board_cv2, self.black_ball, y, x, 1)
            else: # 백돌 차례
                self.board_cv2 = self.game_play(self.board_cv2, self.white_ball, y, x, 2)
                
           
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