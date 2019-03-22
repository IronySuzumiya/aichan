import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple

# 子力基础分值
piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }

# 位置修正分值
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# 初始棋盘状态，棋盘周边做padding，用于快速检测棋子是否跑出棋盘
board_initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# 棋盘四角索引，方便计算棋子相对位置
A1, H1, A8, H8 = 91, 98, 21, 28

# 由于棋盘周边做了padding，故也对pst周边做padding，并结合piece表，计算出最终的子力位置分值
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

# 棋子移动方向枚举，直接对应其在棋盘上的索引值的变化
N, E, S, W = -10, 1, 10, -1

# 棋子移动方向，不区分爬行（Crawling）和滑动（Sliding）
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# 置换表（Transposition Table）大小
TABLE_SIZE = 1e8

# 静态搜索（Quiescence Search）终止阈值
QS_LIMIT = 150

# MTD(f)二分搜索上下界区间模糊度
EVAL_ROUGHNESS = 20

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ 局面状态
    board -- 棋盘状态
    score -- 估分
    wc -- 王车易位权, [长, 短]
    bc -- 对手的王车易位权, [长, 短]
    ep - 吃过路兵标记
    kp - 王车易位通路标记
    """

    def gen_moves(self):
        ''' 生成所有合法着法 '''
        # 根据棋子所有可能的移动方向以生成射线，保留合法项
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # 棋盘边界与友方棋子阻挡
                    if q.isspace() or q.isupper(): break
                    # 兵挺进、double move、吃子（含吃过路兵）
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp): break
                    # 生成着法
                    yield (i, j)
                    # 若为Crawling移动方式的棋子（兵、马、王），或被友方棋子阻挡，则射线停止延长
                    if p in 'PNK' or q.islower(): break
                    # 王车易位
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' 旋转棋盘 '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' 弃权（空着）并旋转棋盘 '''
        # 和直接旋转棋盘的区别在于清空ep和kp
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        ''' 移动棋子 '''
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # 一旦移动车，则失去相应的王车易位权，同理，一旦吃掉对手的车，对手失去相应的王车易位权
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # 王车易位
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # 兵的特殊情况
        if p == 'P':
            # 兵的升变
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            # double move时留下吃过路兵标记
            if j - i == 2*N:
                ep = i + N
            # 吃过路兵
            if j - i in (N+W, N+E) and q == '.':
                board = put(board, j+S, '.')
        # 旋转棋盘以交给对手，利用对称性简化逻辑
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        ''' 对着法估值 '''
        i, j = move
        p, q = self.board[i], self.board[j]
        # 子力位置变化分值
        score = pst[p][j] - pst[p][i]
        # 吃子分值
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        # 王车易位时还需要移动车，计算车的位置变化分值
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # 兵的特殊情况
        if p == 'P':
            # 兵的升变分值
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            # 吃过路兵分值
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

# 最佳着法所在分值区间置换表项
# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class LRUCache:
    """ 置换表，用于支持MTD(f)算法 """
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try: self.od.move_to_end(key)
        except KeyError: return default
        return self.od[key]

    def __setitem__(self, key, value):
        try: del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value

class Searcher:
    """ 着法搜索器 """
    def __init__(self):
        self.tp_score = LRUCache(TABLE_SIZE)
        self.tp_move = LRUCache(TABLE_SIZE)
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        ''' returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)
        '''
        self.nodes += 1

        # Depth <= 0 时为静态搜索，此时搜索需要尽可能深以避免水平线效应，存入置换表时无须关心深度
        depth = max(depth, 0)

        # 王是否还存活
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # 利用置换表快速读取已评估过的局面最佳着法所在分值区间
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper
            
        def moves():
            ''' 着法生成与局面评估器 '''
            # 空着裁剪，R=2，若己方已无兵和王以外的棋子，则不考虑空着
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)
            # 静态搜索先给出空着，用于搜索终止时的估分
            if depth == 0:
                yield None, pos.score
            # 杀手着法启发
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            # 其他着法
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # 搜索
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # 保存着法，用于PV Construction和杀手着法启发
                self.tp_move[pos] = move
                break

        # 保存局面最佳着法所在分值区间
        if best >= gamma:
            self.tp_score[(pos, depth, root)] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[(pos, depth, root)] = Entry(entry.lower, best)

        return best

    def _search(self, pos):
        ''' 迭代加深的MTD(f)二分搜索 '''
        self.nodes = 0

        # 迭代加深
        for depth in range(1, 1000):
            self.depth = depth
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower + upper + 1) // 2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            # 确保至少生成一个着法
            score = self.bound(pos, lower, depth)

            yield

    def search(self, pos, secs):
        ''' 搜索接口 '''
        start = time.time()
        for _ in self._search(pos):
            if time.time() - start > secs:
                break
        return self.tp_move.get(pos), self.tp_score.get((pos, self.depth, True)).lower

def parse(c):
    ''' 用户输入解析，将代数表示法转换为棋盘索引 '''
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank

def render(i):
    ''' parse的逆过程，将棋盘索引转换为代数表示法 '''
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)

def print_pos(pos):
    ''' 打印棋盘（局面） '''
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')

def main():
    pos = Position(board_initial, 0, (True,True), (True,True), 0, 0)
    searcher = Searcher()
    #searcher1 = Searcher()
    while True:
        print_pos(pos)

        if pos.score <= -MATE_LOWER:
            print("You lost")
            break

        move = None
        while move not in pos.gen_moves():
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                print("Please enter a move like d2d4")

        # Player
        pos = pos.move(move)

        #move, score = searcher1.search(pos, secs=5)
        #print("Your move:", render(move[0]), render(move[1]))
        #pos = pos.move(move)

        # 由于执行pos.move后会旋转棋盘，故再做一次旋转
        print_pos(pos.rotate())

        if pos.score <= -MATE_LOWER:
            print("You won")
            break

        # AI
        move, score = searcher.search(pos, secs=2)

        if score == MATE_UPPER:
            print("Checkmate!")

        # 由于AI使用的棋盘是旋转过的，故棋子的坐标是实际坐标的镜像，需要做倒映操作
        print("My move:", render(119-move[0]) + render(119-move[1]))
        pos = pos.move(move)

if __name__ == '__main__':
    main()
