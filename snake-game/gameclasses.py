import pygame
import numpy as np
from collections import deque


def to_code(direction):
    code = None
    if direction == 'U':
        code = 0
    elif direction == 'L':
        code = 1
    elif direction == 'D':
        code = 2
    elif direction == 'R':
        code = 3
    return code


class Snake:
    def __init__(self, position, board_size, head_img, body_img, tail_img, img_unit, init_length=3):
        self.direction = np.random.choice(['U', 'L', 'D', 'R'])

        dx = -1 if self.direction == 'R' else 1 if self.direction == 'L' else 0
        dy = -1 if self.direction == 'D' else 1 if self.direction == 'U' else 0
        self.body = deque()
        # head
        self.body.append(position)
        # body, tail
        for i in range(1, init_length):
            self.body.append((position[0] + i * dx, position[1] + i * dy))

        self.board_size = board_size
        self.head_img = head_img
        self.body_img = body_img
        self.tail_img = tail_img
        self.img_unit = img_unit

    def get_body(self):
        return self.body

    def get_direction(self):
        return self.direction

    def move(self, action, feed_position):
        dir_code = to_code(self.direction)
        if dir_code is None:
            return
        act_code = to_code(action)
        if act_code is None:
            return

        # 가던 방향 그대로
        if dir_code == act_code or (dir_code + 2) % 4 == act_code:
            act_code = dir_code

        old_head = self.body[0]
        if act_code == 0:  # Up
            new_head = (old_head[0], old_head[1] - 1)
            self.direction = 'U'
        elif act_code == 1:  # Left
            new_head = (old_head[0] - 1, old_head[1])
            self.direction = 'L'
        elif act_code == 2:  # Down
            new_head = (old_head[0], old_head[1] + 1)
            self.direction = 'D'
        else:  # act_code == 3 -> Right
            new_head = (old_head[0] + 1, old_head[1])
            self.direction = 'R'

        # when new_head collides with body or wall
        if new_head in self.body or not self.validate(self.body[0]):
            return -1

        self.body.appendleft(new_head)

        # when snake takes feed
        if new_head == feed_position:
            return 1
        else:
            self.body.pop()

        return 0

    def validate(self, pos):
        return 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size

    # LU = Left and Up (보드판의 맨 왼쪽 위 좌표)
    def render(self, screen, LU):
        L = LU[0]
        U = LU[1]
        # head
        screen.blit(pygame.transform.rotate(self.head_img, to_code(self.direction) * 90),
                    (L + self.body[0][0] * self.img_unit, U + self.body[0][1] * self.img_unit))
        # body
        for i in range(1, len(self.body) - 1):
            screen.blit(self.body_img, (L + self.body[i][0] * self.img_unit, U + self.body[i][1] * self.img_unit))
        # tail
        screen.blit(pygame.transform.rotate(self.tail_img, to_code(self.calc_tail_dir()) * 90),
                    (L + self.body[-1][0] * self.img_unit, U + self.body[-1][1] * self.img_unit))

    def calc_tail_dir(self):
        dx = self.body[-1][0] - self.body[-2][0]
        dy = self.body[-1][1] - self.body[-2][1]

        if dx == 1:
            return 'R'
        elif dx == -1:
            return 'L'
        elif dy == 1:
            return 'D'
        elif dy == -1:
            return 'U'


class Button:
    def __init__(self, x, y, width, height, button_text, font, rounded_edge=True):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.fill_colors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }
        self.txt = font.render(button_text, True, (20, 20, 20))
        self.rect = self.txt.get_rect(center=(x, y))
        self.clicked = False
        self.rounded_edge = rounded_edge

    def draw(self, screen):
        pos = pygame.mouse.get_pos()

        state = 'normal'

        action = False
        if self.x - self.width // 2 <= pos[0] < self.x + self.width // 2\
                and self.y - self.height // 2 <= pos[1] < self.y + self.height // 2:
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                state = 'pressed'
                action = True
            else:
                state = 'hover'

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        border_radius = self.height // 3 if self.rounded_edge else 0
        pygame.draw.rect(screen, self.fill_colors[state],
                         [self.x - self.width // 2, self.y - self.height // 2, self.width, self.height],
                         border_radius=border_radius)
        screen.blit(self.txt, self.rect)

        return action
