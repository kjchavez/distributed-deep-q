from collections import deque
from random import choice
import numpy as np


# constants
_NX = 10
_NY = 10
_NORTH = "wW"
_SOUTH = "sS"
_EAST = "dD"
_WEST = "aA"
_DIRECTIONS = "wasdWASD"
_NULL_DIR = "n"
_NULL_POS = (-1, -1)
_EMPTY_CELL = -1
_APPLE = -2
GRID = {(i, j) for i in range(_NX) for j in range(_NY)}
SCORE_GROW = 1
SCORE_MOVE = 0
SCORE_GAME_OVER = -_NX * _NY


# game objects
class Apples(object):
    def __init__(self):
        self.set = set()

    def set_apple(self, apple):
        self.set.add(apple)

    def set_apples(self, apples):
        for apple in apples:
            self.set.add(apple)

    def add_apple(self, emptygrid):
        self.set.add(choice(list(emptygrid)))

    def contains(self, pos):
        return pos in self.set

    def eat(self, apple):
        self.set.remove(apple)

    def clear(self):
        self.set.clear()


class Snake(object):
    def __init__(self, x, y, direction):
        self.deque = deque()
        self.deque.append((x, y))
        self.direction = direction

    def __len__(self):
        return len(self.deque)
        
    def set_snake(self, snake_ordered_list):
        for snake_cell in snake_ordered_list:
            self.deque.append(snake_cell)
        self.direction = self.deduce_direction()

    def contains(self, pos):
        return pos in self.deque

    def clear(self):
        self.deque.clear()
        self.direction = _NULL_DIR

    def legalize_direction(self, direction):
        if (direction in _NORTH and self.direction in _SOUTH) or \
           (direction in _SOUTH and self.direction in _NORTH) or \
           (direction in _EAST and self.direction in _WEST) or \
           (direction in _WEST and self.direction in _EAST):
            return self.direction[0]
        else:
            return direction

    def deduce_direction(self):
        head = self.deque[0]
        neck = self.deque[1]
        
        dx = head[0] - neck[0]
        dy = head[1] - neck[1]
        
        if dx == 1:
            return _NORTH
        elif dx == -1:
            return _SOUTH
        elif dy == 1:
            return _EAST
        elif dy == -1:
            return _WEST
        else:
            raise ValueError("Invalid direction")

    def new_head(self, direction):
        self.direction = self.legalize_direction(direction)

        head = list(self.deque[-1])
        if self.direction in _NORTH:
            head[1] += 1
            head[1] %= _NY
        elif self.direction in _SOUTH:
            head[1] -= 1
            head[1] %= _NY
        elif self.direction in _EAST:
            head[0] += 1
            head[0] %= _NX
        elif self.direction in _WEST:
            head[0] -= 1
            head[0] %= _NX
        else:
            raise ValueError("Direction input error")
            
        return tuple(head)

    def grow(self, direction):
        self.deque.append(self.new_head(direction))

    def move(self, direction):
        self.grow(direction)
        self.deque.popleft()


# game emulator
class SnakeGame(object):
    def __init__(self):
        self.apples = Apples()
        self.snake = Snake(_NX // 2, _NY // 2, _EAST) # grid center
        self.score = 0
        self.gameover = False
        self.generate_apple()
        
    def empty_grid(self):
        return GRID - (set(self.snake.deque) | self.apples.set)

    def generate_apple(self):
        self.apples.add_apple(self.empty_grid())

    def __str__(self):
        string = []
        for y in list(range(_NY))[::-1]:
            string.append(str(y) + ": | ")
            for x in range(_NX):
                cell = (x, y)
                if self.snake.contains(cell):
                    string.append("o | ")
                elif self.apples.contains(cell):
                    string.append("@ | ")
                else:
                    string.append("  | ")
            string.append("\n")
        
        string.append("   | ")
        for x in range(_NX):
            string.append(str(x) + " | ")
        string.append("\n")
        return "".join(string)
        
    def encode_state(self):
        snake_list = list(self.snake.deque)
        state_array = _EMPTY_CELL * np.ones((_NX, _NY))
        for y in range(_NY):
            for x in range(_NX):
                cell = (x, y)
                if cell in snake_list:
                    state_array[x, y] = snake_list.index(cell)
                elif self.apples.contains(cell):
                    state_array[x, y] = _APPLE
        return state_array

    def decode_state(self, state_array):
        snake = [_NULL_POS] * _NX * _NY
        apples = []
        for y in range(_NY):
            for x in range(_NX):
                if state_array[x, y] == _APPLE:
                    apples.append((x, y))
                elif state_array[x, y] != _EMPTY_CELL: # is snake
                    snake[state_array[x, y]] = (x, y)
        return (snake, apples)

    def set_state(self, state_array):
        if a.shape != (_NX, _NY):
            raise ValueError("State array input is incorrect")
        
        self.apples.clear()
        self.snake.clear()
        snake, apples = self.decode_state(state_array)
        self.snake.set_snake(snake)
        self.apples.set_apples(apples)

    def update_game(self, direction):
        ''' Updates game with direction and returns additional score. '''
        new_head = self.snake.new_head(direction)
        if self.snake.contains(new_head):
            self.gameover = True
            print "Game over"
            return SCORE_GAME_OVER
        elif self.apples.contains(new_head):
            self.apples.eat(new_head)
            self.snake.grow(direction)
            self.generate_apple()
            self.score += 1
            assert len(set(self.snake.deque) & self.apples.set) == 0
            return SCORE_GROW
        else:
            self.snake.move(direction)
            assert len(set(self.snake.deque) & self.apples.set) == 0
            return SCORE_MOVE

    def cpu_play(self, state_array, direction):
        self.set_state(state_array)
        score_update = self.update_game(direction)
        return self.encode_state(), score_update

    def human_play(self, direction):
        score_update = self.update_game(direction)
        print(self)
        print "reward: " + str(score_update)
        print "total score: " + str(self.score)

    def human_game(self):
        self.__init__()
        print "Welcome to a game of Snake!"
        print(self)
        while not self.gameover:
            direction = raw_input("Use WASD for direction control: ")
            while direction not in _DIRECTIONS:
                direction = raw_input("Invalid input.\n Use WASD for direction control: ")
            self.human_play(direction)