from __future__ import print_function
WIDTH = 540
HEIGHT = 540
MARGIN = 22
GRID = (WIDTH - 2 * MARGIN) / (15 - 1)
PIECE = 34
EMPTY = 0
BLACK = 1
WHITE = 2
SCALE = 5
width = 8
height = 8
n_in_row = 5
model_file = 'model/8_8_5_best_policy_.model'
use_gpu = False
n_playout = 800
use_rws = False 
from game import *
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  
import sys
import os
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from mcts_alphaZero_BTMM import MCTSPlayerBTMM
import threading

global AIChess


class cycleGroup(tuple):
    def __init__(self, parent):
        self.elements = parent
        self.order = len(parent)
        self.point = 0
    def pointTurnRight(self):
        self.point = (self.point + 1) % self.order
    def pointTurnLeft(self):
        self.point = (self.point + self.order - 1) % self.order
    def element(self):
        return self.elements[self.point]

class chessDetail(object):
    def __init__(self, i=0, j=0, x=0, y=0, chess=0, chessPic=None):
        self.gridCoordinate_i = i
        self.gridCoordinate_j = j
        self.pixelCoordinate_x = x
        self.pixelCoordinate_y = y
        self.chessType = chess
        self.chessPicture = chessPic


class ChessBoard(QWidget):
    signalClicked = pyqtSignal()
    signalAIFirst = pyqtSignal(bool)
    signalHumanDraw_ChessCoordinates = pyqtSignal(int, int)
    signalDraw_Finished = pyqtSignal(bool)
    def __init__(self):
        super(ChessBoard, self).__init__()
        
    def initialize(self, scale):
        self.graphicsParameterSet(scale)
        self.graphicsUIInterfaceSet()
        self.boardRunningLogicSet()
        
    def graphicsParameterSet(self, scale):
        self.WIDTH = 540
        self.HEIGHT = 540
        self.MARGIN = 22
        self.GRID = (self.WIDTH - 2 * self.MARGIN) / (15 - 1)
        self.PIECE = 34
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2
        self.SCALE = scale
        self.effectiveWIDTH = self.GRID * (self.SCALE - 1) + 2 * self.MARGIN
        self.effectiveHEIGHT = self.effectiveWIDTH
    
    def graphicsUIInterfaceSet(self):
        self.graphicsElementSet()
        self.graphicsUserSelfDefine()
        self.graphicsUserChoosingChessType()
        self.graphicsUserChoosingFirstHand()
        self.graphicsChessBoard()
    
    def graphicsElementSet(self):
        self.black = QPixmap('black.png')
        self.white = QPixmap('white.png')
        
    def graphicsUserSelfDefine(self):
        pass
    
    def graphicsUserChoosingChessType(self):
        message = QMessageBox()
        message.setIconPixmap(self.black)
        message.setWindowTitle("选择颜色")
        message.setText("玩家选择棋子颜色")
        message.addButton(QPushButton("黑子"), QMessageBox.YesRole)
        message.addButton(QPushButton("白子"), QMessageBox.NoRole)
        
        self.humanChessType = None
        answer = message.exec()
        global AIChess
        if answer == 0:
            self.humanChessType = BLACK
            self.AIChessType = WHITE
            AIChess = WHITE
        else:
            self.humanChessType = WHITE
            self.AIChessType = BLACK
            AIChess = BLACK
        self.dictionaryFromNameToElement = {'HUMAN':self.humanChessType, 'AI':self.AIChessType}
        
    def graphicsUserChoosingFirstHand(self):
        message = QMessageBox()
        message.setIconPixmap(self.white)
        message.setWindowTitle("选择先手")
        message.setText("玩家决定先手顺序")
        message.addButton(QPushButton("AI先手"), QMessageBox.YesRole)
        message.addButton(QPushButton("玩家先手"), QMessageBox.NoRole)
      
        answer = message.exec()
        if answer == 0:
            ai_first = True
        else:
            ai_first = False
            
        self.signalAIFirst.emit(ai_first)
        
    def graphicsChessBoard(self):
        
        
        palette1 = QPalette()  
        palette1.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap('chessboard.jpg')))
        self.setPalette(palette1)
        
        self.setCursor(Qt.PointingHandCursor)

        self.setMaximumSize(QtCore.QSize(WIDTH, HEIGHT))
        self.setWindowTitle("NINAROW")
        self.setWindowIcon(QIcon('black.png'))
        self.resize(int(self.effectiveWIDTH), int(self.effectiveHEIGHT))

        self.mouse_point = QLabel()
        self.mouse_point.setScaledContents(True)
        self.mouse_point.setPixmap(self.black)  
        self.mouse_point.setGeometry(270, 270, PIECE, PIECE)
        self.mouse_point.raise_()  
        self.setMouseTracking(True)      
        
        
    def graphicsGameOver(self, winner):
        self.gameOverMessage = QMessageBox()
        reply = self.gameOverMessage.information(self, '游戏结束', winner, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.close()
        else:
            self.close()
        sys.exit()
        
    def boardRunningLogicSet(self):
        self.isAIAlreadyDrawn = 1
        if self.isAIAlreadyDrawn is True:
            self.humanAvailable = 0
        else:
            self.humanAvailable = 1
        self.dictConstantToPic = {BLACK:self.black, WHITE:self.white}
        self.chessGrid = [[chessDetail(i, j, self.coordinate_transform_map2pixel(i, j)[0],
                                       self.coordinate_transform_map2pixel(i, j)[1], 0,
                                       QLabel(self)) for i in range(1+self.SCALE)] for j in range(1+self.SCALE)]
        
    def aiHasDrawn(self, AIHasDrawn):
        if AIHasDrawn is True:
            self.humanAvailable = True
        
        
    def mouseMoveEvent(self, e):
        self.mouse_point.move(e.x() - 16, e.y() - 16)
        
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.humanAvailable:
            x, y = e.x(), e.y()
            i, j = self.coordinate_transform_pixel2map(x, y)
            if i < self.SCALE and j < self.SCALE:
                self.draw(i, j, 'HUMAN')
                self.signalHumanDraw_ChessCoordinates.emit(i, j)
                self.signalClicked.emit()
    
    def closeEvent(self, event):
        event.accept()
        sys.exit()
    
    def draw(self, i, j, Player):
        chessToDraw = self.dictionaryFromNameToElement[Player]
        x, y = self.coordinate_transform_map2pixel(i, j)
        self.chessGrid[i][j].chessType = chessToDraw
        self.chessGrid[i][j].chessPicture.setMouseTracking(True)
        self.chessGrid[i][j].chessPicture.setVisible(True)
        self.chessGrid[i][j].chessPicture.setPixmap(self.dictConstantToPic[chessToDraw])
        self.chessGrid[i][j].chessPicture.setGeometry(int(x), int(y), int(self.PIECE), int(self.PIECE))

        self.update()
        QApplication.processEvents()
        self.signalDraw_Finished.emit(True)
    
    def coordinate_transform_map2pixel(self, i, j):
        return self.MARGIN + j * self.GRID - self.PIECE / 2, self.MARGIN + i * self.GRID - self.PIECE / 2

    def coordinate_transform_pixel2map(self, x, y):
        i, j = int(round((y - self.MARGIN) / self.GRID)), int(round((x - self.MARGIN) / self.GRID))
        if i < 0 or i >= 15 or j < 0 or j >= 15:
            return None, None
        else:
            return i, j    

class HumanAgent(object):
    """
    human player
    """
    
    def __init__(self, interface):
        self.player = None
        self.interface = interface
        self.interface.signalHumanDraw_ChessCoordinates.connect(self.get_location)
    def set_player_ind(self, p):
        """设置人类玩家的编号，黑：1，白：2"""
        self.player = p

    def get_action(self, board):
        """根据棋盘返回动作"""
        location = self.get_location_from_window()
        if isinstance(location, str): 
            location = [int(n, 10) for n in location.split(",")] 
        move = board.location_to_move(location) 
        if move not in board.availables():
            move = self.get_action(board)
        return move

    
    def get_location(self, i, j):
        self.currentLocation = [i, j]
        return
        
    def get_location_from_window(self, timeout = 10000):
        loop = QEventLoop()
        self.interface.signalClicked.connect(loop.quit)
        loop.exec_()
        return self.currentLocation

    def __str__(self):
        return "Human {}".format(self.player)    
#        try:
#            location = self.get_location_from_window() # 从键盘上读入位置，eg. "0,2"代表第0行第2列
#            print('location = ', location)
#            if isinstance(location, str):  # 如果location确实是字符串
#                location = [int(n, 10) for n in location.split(",")] # 将location转换为对应的坐标点
#            move = board.location_to_move(location) # 坐标点转换为一个一维的值move,介于[0,width*height)
#        except Exception as e: #异常情况下
#            move = -1
#        if move == -1 or move not in board.availables: # 如果move值不合法
#            print("invalid move")
#            move = self.get_action(board) # 重新等待输入
#        return move
    

class UserInterface_GO_Human_vs_Human(QWidget):
    signalOfDrawnChess = pyqtSignal(int, int, str)
    signalOfWinner = pyqtSignal(str)
    def __init__(self,board_logic,width,height):
        super().__init__()
        self.board = board_logic
        self.interface = ChessBoard()
        self.human = HumanAgent(self.interface)
        
        
class UserInterface_GO_Human_vs_AI(QWidget):
    signalOfDrawnChess = pyqtSignal(int, int, str)
    signalOfWinner = pyqtSignal(str)
    def __init__(self, AIPlayer, board_logic, width, height):
        super().__init__()
        self.AI = AIPlayer
        self.board = board_logic
        self.interface = ChessBoard()
        self.human = HumanAgent(self.interface)
        self.logicProcess()
        
        if width == height:
            self.scale = width 
        
        self.width = width
        self.height = height
            
    def run(self):
        self.interface.show()
        
    def test(self):
        self.interface.signalAIFirst.connect(self.cycleInitialize)
        self.signalOfDrawnChess.connect(self.interface.draw)
        self.signalOfWinner.connect(self.interface.graphicsGameOver)
        self.interface.initialize(self.scale)
        self.interface.show()
        self.playChess()


    def playChess(self):

        end, winner = self.board.game_end()
        while end is False:
            currentPlayer = self.dictionary[self.chesses.element()]
            playerName = self.chesses.element()
            nextMove = currentPlayer.get_action(self.board)
            self.board.do_move(nextMove)
            nextLocation = self.board.move_to_location(nextMove)
            i, j = nextLocation[0], nextLocation[1]
            self.signalOfDrawnChess.emit(i, j, playerName)
            self.chesses.pointTurnRight()
            end, winner = self.board.game_end()
        try:
            save_board_bitboard_and_2d(self.board, "ketqua_ui")
        except Exception as e:
            print("Lỗi khi lưu trạng thái:", e)

        if winner == 1:
            print("玩家胜利")
            strWinner = '玩家胜利'
        else:
            print("AI胜利")
            strWinner = 'AI胜利'
        self.signalOfWinner.emit(strWinner)
        
    def logicProcess(self):
        self.dictionary = {'HUMAN':self.human, 'AI':self.AI}
        cycle = ('HUMAN', 'HUMAN', 'AI', 'AI')
        self.chesses = cycleGroup(cycle)

    def cycleInitialize(self, aiFirst):
        if aiFirst is True:
            self.chesses.point = 3
            self.board.init_board(1)
        else:
            self.chesses.point = 1
            self.board.init_board(0)
    
def run(n_in_row, width, height, 
        model_file, ai_first,
        n_playout, use_gpu,
        use_rws=False,  time_limit=None):  
    try:
        board = Board(width=width, height=height, n_in_row=n_in_row) 

        best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=use_gpu) 
        mcts_player = MCTSPlayerBTMM(best_policy.policy_value_fn, c_puct=5, n_playout=n_playout, use_rws=True, time_limit=time_limit)
        main = UserInterface_GO_Human_vs_AI(mcts_player, board, width, height,)
        main.test()
    except KeyboardInterrupt:
        print('\n\rquit')

def usage():
    print("-s 设置棋盘大小，默认为6")
    print("-r 设置是几子棋，默认为4")
    print("-m 设置每步棋执行MCTS模拟的次数，默认为400")
    print("-i ai使用哪个文件中的模型，默认为model/6_6_4_best_policy.model")
    print("--use_gpu 使用GPU进行运算")
    print("--human_first 让人类先下")
    print("--rws      AI sẽ dùng Roulette Wheel Selection (ngẫu nhiên theo xác suất)")

def save_board_bitboard_and_2d(board, filename_prefix="ketqua_ui"):
    width = board.width
    height = board.height

    if hasattr(board, "bitboards"):
        with open(f"{filename_prefix}_bitboard.txt", "w") as f:
            f.write(str(board.bitboards[1]) + "\n")
            f.write(str(board.bitboards[2]) + "\n")
    else:
        print("Không tìm thấy thuộc tính bitboards trên board!")

    import numpy as np
    arr = np.zeros((height, width), dtype=int)
    for pos in range(width * height):
        player = board.get_player_at(pos) if hasattr(board, 'get_player_at') else 0
        i, j = pos // width, pos % width
        arr[i, j] = player
    np.savetxt(f"{filename_prefix}_2d.txt", arr, fmt='%d', delimiter=' ')
    print("Đã lưu trạng thái bàn cờ dưới dạng bitboard và 2D (txt)")


if __name__ == '__main__':
    import sys, getopt
    from PyQt5 import QtWidgets
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    height = 19
    width = 19
    n_in_row = 6
    use_gpu = False
    n_playout = 800
    model_file = "model/19_19_6_best_policy_hi_bitboard.model"
    ai_first = True
    use_rws = False 
    time_limit = None
    opts, args = getopt.getopt(sys.argv[1:], "hs:r:m:i:", ["use_gpu", "graphics", "human_first", "rws", "time_limit="])


    for op, value in opts:
        if op == "-h":
            usage()
            sys.exit()
        elif op == "-s":
            height = width = int(value)
        elif op == "-r":
            n_in_row = int(value)
        elif op == "--use_gpu":
            use_gpu = True
        elif op == "-m":
            n_playout = int(value)
        elif op == "-i":
            model_file = value
        elif op == "--human_first":
            ai_first = False
        elif op == "--rws":    
            use_rws = True
        elif op == "--time_limit":
            time_limit = float(value)


    run(height=height, width=width, n_in_row=n_in_row, use_gpu=use_gpu, n_playout=n_playout,
        model_file=model_file, ai_first=ai_first, use_rws=use_rws, time_limit=time_limit)

