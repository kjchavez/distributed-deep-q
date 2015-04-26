from random import choice
from collections import deque

_NX = 10
_NY = 10
NORTH = "nj"
SOUTH = "sk"
EAST = "el"
WEST = "wh"
NULLPOS = (-1, -1)

class Apples(object):
    def __init__(self):
        self.apples_set = set()

    def add(self, emptygrid):
        self.apples_set.add(choice(list(emptygrid)))

    def contains(self, pos):
        return pos in self.apples_set

    def eat(self, applepos):
        self.apples_set.remove(applepos)

class Snake(object):
    def __init__(self, x, y, direction):
        self.head = [x, y]
        self.direction = direction
        self.snake_deque = deque()
        self.snake_deque.append((self.head[0], self.head[1]))
        self.snake_set = set()
        self.snake_set.add((self.head[0], self.head[1]))

    def contains(self, pos):
        return pos in self.snake_set

    def move(self, direction):
        last_cell = self.snake_deque.popleft()
        self.snake_set.remove(last_cell)

        if (direction in NORTH and self.direction in SOUTH) or \
           (direction in SOUTH and self.direction in NORTH) or \
           (direction in EAST and self.direction in WEST) or \
           (direction in WEST and self.direction in EAST):
            direction = self.direction[0]

        if direction in NORTH:
            self.head[1] += 1
            self.head[1] %= _NY
        elif direction in SOUTH:
            self.head[1] -= 1
            self.head[1] %= _NY
        elif direction in EAST:
            self.head[0] += 1
            self.head[0] %= _NX
        elif direction in WEST:
            self.head[0] -= 1
            self.head[0] %= _NX
        else:
            raise ValueError("Direction input error")

        self.direction = direction
        self.snake_deque.append((self.head[0], self.head[1]))
        self.snake_set.add((self.head[0], self.head[1]))

    def grow(self, direction):

        if (direction in NORTH and self.direction in SOUTH) or \
           (direction in SOUTH and self.direction in NORTH) or \
           (direction in EAST and self.direction in WEST) or \
           (direction in WEST and self.direction in EAST):
            direction = self.direction[0]

        if direction in NORTH:
            self.head[1] += 1
        elif direction in SOUTH:
            self.head[1] -= 1
        elif direction in EAST:
            self.head[0] += 1
        elif direction in WEST:
            self.head[0] -= 1
        else:
            raise ValueError("Direction input error")

        self.snake_deque.append((self.head[0], self.head[1]))
        self.snake_set.add((self.head[0], self.head[1]))                

class Game(object):
    def __init__(self):
        self.grid = {(i, j) for i in range(_NX) \
                            for j in range(_NY)}
        self.apples = Apples()
        self.snake = Snake(_NX // 2, _NY // 2, EAST)
        self.score = 0
        self.gameover = False
        self.add_apple()

    def emptygrid(self):
        return self.grid - (self.snake.snake_set | self.apples.apples_set)

    def add_apple(self):
        self.apples.add(self.emptygrid())

    def newpos(self, direction):
        if (direction in NORTH and self.snake.direction in SOUTH) or \
           (direction in SOUTH and self.snake.direction in NORTH) or \
           (direction in EAST and self.snake.direction in WEST) or \
           (direction in WEST and self.snake.direction in EAST):
            direction = self.snake.direction[0]

        if direction in NORTH:
            return (self.snake.head[0], (self.snake.head[1] + 1) % _NY)
        elif direction in SOUTH:
            return (self.snake.head[0], (self.snake.head[1] - 1) % _NY)
        elif direction in EAST:
            return (self.snake.head[0] + 1, (self.snake.head[1]) % _NX)
        elif direction in WEST:
            return (self.snake.head[0] - 1, (self.snake.head[1]) % _NX)
        else:
            raise ValueError("Direction input error")

    def viz(self):
        self.vizgrid(self.snake.snake_set, self.apples.apples_set)

    def vizgrid(self, snake, apples):
        print "\nGrid:\n"
        for y in list(range(_NY))[::-1]:
            print str(y) + ": |",
            for x in range(_NX):
                cell = (x, y)
                if cell in snake:
                    print "o |",
                elif cell in apples:
                    print "@ |",
                else:
                    print "  |",
            print "\n",
        
        print "   |",
        for x in range(_NX):
            print str(x) + " |",
        print "\n"

    def play(self, direction):
        newpos = self.newpos(direction)
        if self.snake.contains(newpos):
            self.gameover = True
            print "Game over!"
        elif self.apples.contains(newpos):
            self.apples.eat(newpos)
            self.snake.grow(direction)
            self.add_apple()
            self.score += 1
        else:
            self.snake.move(direction)
        assert len(self.snake.snake_set & self.apples.apples_set) == 0

    def newgame(self):
        self.__init__()
        self.vizgrid(self.snake.snake_set, self.apples.apples_set)
        while not self.gameover:
            direction = raw_input("Input the next command: ")
            while direction not in NORTH and direction not in SOUTH and \
                  direction not in EAST and direction not in WEST:
                print "Input based on WASD commands"
                direction = raw_input("Input the next command: ")
            self.play(direction)
            self.vizgrid(self.snake.snake_set, self.apples.apples_set)
        print "Game over!"