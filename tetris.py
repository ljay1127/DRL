import numpy as np
import pygame
import sys
from collections import deque
import random
import os.path

from PIL import Image

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Conv2D, MaxPool2D, Input
from tensorflow.keras.optimizers import Adam

# TETRIS ENVIRONMENT

class Tetris:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((24,10))

        # SAR
        self.state = np.zeros((84, 84, 3))
        self.action = 0
        self.reward = 0
        self.done = False

        self.BLOCK_SIZE = 30

        # PIECE INFO
        self.piece = 0
        self.piece_pos = 0

        # GAME INFO
        self.n_holes = 0
        self.curr_height = 23

        return self.action, self.reward, self.done
    
    def render(self):
        for i in range(20):
            for j in range(10):
                if self.grid[i + 4, j] == 1:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 255), tmp_rec)
                elif self.grid[i + 4, j] == 2:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 255, 0), tmp_rec)
                else:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 0), tmp_rec)

    def step(self, action):
        self.action = action
        self.reward = 0
        
        if self.action == 0:
            self.drop_down()
        elif self.action == 1:
            self.move_left()
        elif self.action == 2:
            self.move_right()
        elif self.action == 3:
            self.drop_down()
        elif self.action == 4:
            self.rotate_piece()

        self.done = self.check_game_over()

        return self.action, self.reward, self.done

    def create_piece(self):
        r_num = random.randint(0, 6)

        ##
        ##
        if r_num == 0:
            self.grid[3:5, 4:6] = 2
         #
        ###
        elif r_num == 1:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3] = 0
            self.grid[3, 5] = 0
        ####
        elif r_num == 2:
            self.grid[3, 3:7] = 2 
         ##
        ##
        elif r_num == 3:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3] = 0
            self.grid[4, 5] = 0
        ##
         ##
        elif r_num == 4:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 5] = 0
            self.grid[4, 3] = 0
          #
        ###
        elif r_num == 5:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3:5] = 0
        #
        ###
        elif r_num == 6:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 4:6] = 0

        self.piece = r_num
        self.piece_pos = 0

    def get_active_piece(self):
        active_piece = []
        for i in range(24):
            for j in range(10):
                if self.grid[i,j] == 2:
                    active_piece.append([i,j])
        return active_piece

    def check_collision(self, new_location):
        for i in new_location:
            if i[0] > 23:
                return True
            elif i[0] < 0:
                return True
            elif i[1] > 9:
                return True
            elif i[1] < 0:
                return True
            elif self.grid[i[0], i[1]] == 1:
                return True
        return False

    def update_piece_location(self, active_piece, new_location):
        for i in active_piece:
            self.grid[i[0], i[1]] = 0
        for i in new_location:
            self.grid[i[0], i[1]] = 2

    def change_piece_status(self, active_piece):
        for i in active_piece:
            self.grid[i[0], i[1]] = 0
        for i in active_piece:
            self.grid[i[0], i[1]] = 1

    def check_game_over(self):
        for i in range(24):
            for j in range(10):
                if self.grid[i,j] == 1 and i < 4:
                    self.reward = -10
                    return True
        return False

    def drop_down(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0] + 1, i[1]])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)
        else:
            self.change_piece_status(active_piece)
            self.check_grid()

    def move_left(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0], i[1] - 1])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)

    def move_right(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0], i[1] + 1])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)

    def rotate_piece(self):
        active_piece = self.get_active_piece()
        new_location = []
         #
        ###
        if self.piece == 1:
             #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            ##
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row - 1, col + 2])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
             #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 1, col - 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
             #
            ##
             #
            elif self.piece_pos == 3:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row - 1, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        ####
        elif self.piece == 2:
            ####
            if self.piece_pos == 0:
                row, col = active_piece[0]
                new_location.append([row - 1, col + 1])
                new_location.append([row - 2, col + 1])
                new_location.append([row - 3, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            #
            #
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
         ##
        ##
        elif self.piece == 3:
             ##
            ##
            if self.piece_pos == 0:
                row, col = active_piece[-2]
                new_location.append([row - 2, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            ##
             #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        ##
         ##
        elif self.piece == 4:
            ##
             ##
            if self.piece_pos == 0:
                row, col = active_piece[-2]
                new_location.append([row, col - 1])
                new_location.append([row - 1, col - 1])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
             #
            ##
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
          #
        ###
        elif self.piece == 5:
              #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col - 2])
                new_location.append([row - 1, col - 2])
                new_location.append([row - 2, col - 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            #
            ##
            elif self.piece_pos == 1:
                row, col = active_piece[-2]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row - 1, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
            #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row - 2, col])
                new_location.append([row - 2, col + 1])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
            ##
             #
             #
            elif self.piece_pos == 3:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        #
        ###
        elif self.piece == 6:
            #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 2, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            ##
            #
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col + 2])
                new_location.append([row-1, col])
                new_location.append([row-1, col + 1])
                new_location.append([row-1, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
              #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row, col - 2])
                new_location.append([row, col - 1])
                new_location.append([row - 1, col - 1])
                new_location.append([row - 2, col - 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
             #
             #
            ##
            elif self.piece_pos == 3:
                row, col = active_piece[-2]
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                new_location.append([row - 1, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0

    def check_grid(self):
        # CHECK FOR HOLES ON THE GAME
        hole_found = 0
        height = 0
        for i in range(23):
            for j in range(10):
                if self.grid[i, j] == 1:
                    if height == 0:
                        height = i
                    if self.grid[i + 1, j] == 0:
                        hole_found += 1
        if hole_found > self.n_holes:
            self.reward = -1
        elif hole_found < self.n_holes:
            self.reward = 1
        elif hole_found == self.n_holes:
            test = self.get_active_piece()
            if len(test) == 0:
                self.reward = 1
        self.n_holes = hole_found

        # CHECK FOR ROW TO REMOVE
        delete_this_row = []
        for i in range(24):
            delete_row = True
            for j in range(10):
                if self.grid[i, j] == 0:
                    delete_row = False
                    break
            if delete_row:
                delete_this_row.append(i)
        
        if len(delete_this_row) > 0:
            self.curr_height += len(delete_this_row)
            self.reward = 10 * len(delete_this_row)

            new_grid = self.grid.copy()
            new_grid = np.delete(new_grid, delete_this_row, axis=0)
            self.grid = np.zeros((24, 10), dtype=int)
            self.grid[24 - new_grid.shape[0]: ,:] = new_grid

    def convert_image(self, data):
        data = np.array(data)
        img = Image.fromarray(data).convert("L")
        img = img.resize((84,84))
        return np.reshape(np.array(img), (84, 84, 1))

    def add_image(self, observation_):
        observation_ = observation_ / 255
        self.state[:,:,0] = self.state[:,:,1]
        self.state[:,:,1] = self.state[:,:,2]
        self.state[:,:,2] = observation_[:,:,0]

# DQN AGENT
class DQN_Agent:
    def __init__(self, input_shape, n_actions):
        self.rng = 1
        self.rng_min = 0.1
        self.rng_decay = 0.999
        self.discount = 0.95
        self.weights='weights'
        self.decay_ctr = 0
        self.transfer_weight_ctr = 0
        self.frames_ctr = 0

        self.memory = deque(maxlen=20_000)

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.transfer_weights()

    def create_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same', activation='relu')(input)
        x = Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=(2,2), strides=(1,1), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        V = Dense(1, activation='linear')(x)
        A = Dense(self.n_actions, activation='linear')(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
    
        model = Model(inputs=input, outputs=Q)
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return model

    def remember(self, state, action, reward, state_, done):
        self.memory.append([state, action, reward, state_, done])

    def save(self):
        self.model.save_weights(self.weights)

    def load(self):
        self.model.load_weights(self.weights)
        self.transfer_weights()

    def action(self, state):
        if random.random() < self.rng:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.predict(state))

    def predict(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))

    def train(self):
        if len(self.memory) < 5_000:
            return

        self.decay_ctr += 1
        self.transfer_weight_ctr += 1

        mini_batch = random.sample(self.memory, 32)
        states = np.array([memory[0] for memory in mini_batch])
        states_ = np.array([memory[3] for memory in mini_batch])
        qs = self.model.predict(states)
        qs_ = self.target_model.predict(states_)

        X = states
        y = []

        for i, memory in enumerate(mini_batch):
            action = memory[1]
            reward = memory[2]
            done = memory[4]

            if done:
                q = reward
            else:
                q = reward + self.discount * np.max(qs_[i])

            qs[i][action] = q
            y.append(qs)
    
        self.model.fit(X, np.array(y), verbose=0, shuffle=False)

        if self.decay_ctr > 10:
            self.decay_rng()
            self.decay_ctr = 0

        if self.transfer_weight_ctr > 50:
            self.transfer_weights()
            self.transfer_weight_ctr = 0

        self.frames_ctr += 32

    def decay_rng(self):
        self.rng = self.rng * self.rng_decay
        if self.rng < self.rng_min:
            self.rng = self.rng_min
        print(self.rng)
        print(self.frames_ctr)

    def transfer_weights(self):
        self.target_model.set_weights(self.model.get_weights())


# MAIN GAME LOOP

# SCREEN
pygame.init()
SCREEN_SIZE = width, height = 300, 600
screen = pygame.display.set_mode(SCREEN_SIZE)

# DELTA TIME
current_time = pygame.time.get_ticks()
previous_time = current_time
delta_time = current_time - previous_time
drop_time = 0
control_time = 0

# INITIALIZE
game = Tetris()
agent = DQN_Agent((84,84,2), 5)
if os.path.exists('weights.data-00000-of-00001'):
    agent.load()
    print('agent loaded!')
else:
    print('no saved agent found!')
action, reward, done = game.reset()
ai_player = True
paused = False
train_ctr = 0

while True:
    # UPDATE DELTA TIME
    current_time = pygame.time.get_ticks()
    delta_time = current_time - previous_time
    previous_time = current_time

    train_ctr += 1

    if paused:
        delta_time = 0

    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            agent.save()
            sys.exit()
        # KEYBOARD EVENTS
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 4
            elif event.key == pygame.K_ESCAPE:
                agent.save()
                sys.exit()
            elif event.key == pygame.K_SPACE:
                if ai_player == True:
                    ai_player = False
                else:
                    ai_player = True
            elif event.key == pygame.K_p:
                if paused:
                    paused = False
                else:
                    paused = True

    # KEYBOARD LONG PRESS
    keys = pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
        action = 3
    elif keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2

    # NO MORE ACTIVE PIECE
    active_piece = game.get_active_piece()
    if len(active_piece) == 0:
        game.create_piece()

    if ai_player == True:
        # CONTROLS
        control_time += delta_time
        if control_time > 10:
            # AI
            action, reward, done = game.step(agent.action(game.state[:,:,1:]))
            control_time = 0

        # GRAVITY DROP
        drop_time += delta_time
        if drop_time > 10:
            action, reward, done = game.step(0)
            drop_time = 0
    else:
        # CONTROLS
        control_time += delta_time
        if control_time > 75 and action != 0:
            action, reward, done = game.step(action)
            control_time = 0

        # GRAVITY DROP
        drop_time += delta_time
        if drop_time > 500:
            action, reward, done = game.step(0)
            drop_time = 0

    # SCREEN DRAW
    screen.fill((0, 0, 0))
    game.render()
    pygame.display.flip()

    pygame.image.save(screen, 'tetris.png')
    img = Image.open('tetris.png')
    game.add_image(game.convert_image(img))

    agent.remember(game.state[:,:,0:2], action, reward, game.state[:,:,1:], done)
    if train_ctr > 100:
        agent.train()
        train_ctr = 0

    # GAME OVER
    if done:
        action, reward, done = game.reset()
