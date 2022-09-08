import os
import numpy as np
import time
import pygame

from gameclasses import Snake, Button


def make_feed(snake_body, board_size):
    x = -1
    y = -1
    while not (0 <= x < board_size and 0 <= y < board_size) or (x, y) in snake_body:
        x = np.random.randint(0, board_size)
        y = np.random.randint(0, board_size)
    return x, y


def render_score(score, font):
    txt_score = font.render(score, True, (0, 0, 0))
    rect_score = txt_score.get_rect(center=(screen_width // 2, (screen_height - screen_width) // 2))
    screen.blit(txt_score, rect_score)


def mode_select():
    global PLAYER

    btn_width = 300
    btn_height = 100
    player_button = Button(screen_width // 2, 2 * screen_height // 5, btn_width, btn_height, '직접 플레이(enter)', btn_font)
    ai_button = Button(screen_width // 2, 3 * screen_height // 5, btn_width, btn_height, 'AI(미구현)', btn_font)

    running = True
    while running:
        if player_button.draw(screen):
            PLAYER = 'human'
            running = False
        if ai_button.draw(screen):
            print('미구현')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    PLAYER = 'human'
                    running = False

        pygame.display.update()


def play():
    # 뱀 최초 위치 랜덤 생성(벽에 너무 가까이 있으면 시작하자마자 죽을 수 있으므로 벽과 적당히 떨어져있도록 설정)
    snake_init_pos = (
        np.random.randint(board_size // 4, 3 * board_size // 4),
        np.random.randint(board_size // 4, 3 * board_size // 4))
    snake = Snake(position=snake_init_pos, board_size=board_size,
                  head_img=snake_head, body_img=snake_body, tail_img=snake_tail, img_unit=board_unit_size)

    # 먹이 위치
    feed_pos = make_feed(snake.get_body(), board_size)

    running = True
    LU = (0, screen_height - screen_width)
    score = 0
    pygame.mixer.music.play(-1)
    while running:
        clock.tick(fps)

        next_action = snake.get_direction()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    next_action = 'U'
                elif event.key == pygame.K_LEFT:
                    next_action = 'L'
                elif event.key == pygame.K_DOWN:
                    next_action = 'D'
                elif event.key == pygame.K_RIGHT:
                    next_action = 'R'

        reward = snake.move(next_action, feed_pos)
        # 먹이 먹음
        if reward == 1:
            score += 1
            feed_pos = make_feed(snake.get_body(), board_size)
        # 죽음
        if reward == -1:
            pygame.mixer.music.stop()
            break

        screen.blit(background, (0, 0))
        snake.render(screen, LU)
        screen.blit(feed, (LU[0] + feed_pos[0] * board_unit_size, LU[1] + feed_pos[1] * board_unit_size))
        render_score(str(score), game_font)
        pygame.display.update()


def replay():
    global end
    pygame.display.update()

    btn_width = 300
    btn_height = 100
    replay_button = Button(screen_width // 2, 2 * screen_height // 5, btn_width, btn_height, '다시하기(enter)', btn_font)
    end_button = Button(screen_width // 2, 3 * screen_height // 5, btn_width, btn_height, '끝내기(esc)', btn_font)

    running = True
    while running:
        if replay_button.draw(screen):
            running = False
        if end_button.draw(screen):
            end = True
            running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
                if event.key == pygame.K_ESCAPE:
                    end = True
                    running = False

        pygame.display.update()


# player or ai
PLAYER = None

pygame.init()

# 보드(정사각형) 한 변을 이루는 칸의 개수
board_size = 20
# 보드 한 칸의 한 변 길이
board_unit_size = 30

screen_width = board_size * board_unit_size
# 점수 등을 표시할 위쪽 여백 100 확보
screen_height = screen_width * 1.25

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('RL Snake')
clock = pygame.time.Clock()
# 초당 움직이는 칸 개수
fps = 7

# 이미지 load
current_path = os.path.dirname(os.path.realpath(__file__))
essets_path = f'{current_path}/essets/'
background = pygame.image.load(os.path.join(essets_path, 'background.jpg'))
snake_head = pygame.image.load(os.path.join(essets_path, 'snake_head.png')).convert_alpha()
snake_body = pygame.image.load(os.path.join(essets_path, 'snake_body.png'))
snake_tail = pygame.image.load(os.path.join(essets_path, 'snake_tail.png')).convert_alpha()
feed = pygame.image.load(os.path.join(essets_path, 'feed.png'))

background = pygame.transform.scale(background, (screen_width, screen_height))
snake_head = pygame.transform.scale(snake_head, (board_unit_size, board_unit_size))
snake_body = pygame.transform.scale(snake_body, (board_unit_size, board_unit_size))
snake_tail = pygame.transform.scale(snake_tail, (board_unit_size, board_unit_size))
feed = pygame.transform.scale(feed, (board_unit_size, board_unit_size))

# bgm load
pygame.mixer.music.load(os.path.join(essets_path + 'bgm.wav'))

btn_font = pygame.font.SysFont('applegothic', 30)
game_font = pygame.font.SysFont('arialrounded', 40)

mode_select()
end = False
while not end:
    play()
    replay()
