import os
import numpy as np
import pygame

from gameclasses import Snake, Button


def make_feed(snake_body, board_size):
    x = -1
    y = -1
    while not (0 <= x < board_size and 0 <= y < board_size) or (x, y) in snake_body:
        x = np.random.randint(0, board_size)
        y = np.random.randint(0, board_size)
    return x, y


def render_score(score, font, screen, screen_width, screen_height):
    txt_score = font.render(score, True, (0, 0, 0))
    rect_score = txt_score.get_rect(center=(screen_width // 2, (screen_height - screen_width) // 2))
    screen.blit(txt_score, rect_score)


def init():
    pygame.init()

    # 보드(정사각형) 한 변을 이루는 칸의 개수
    board_size = 20
    # 보드 한 칸의 한 변 길이
    board_unit_size = 30

    screen_width = board_size * board_unit_size
    # 점수 등을 표시할 위쪽 여백 확보
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

    # 이미지를 화면 크기에 맞게 scaling
    background = pygame.transform.scale(background, (screen_width, screen_height))
    snake_head = pygame.transform.scale(snake_head, (board_unit_size, board_unit_size))
    snake_body = pygame.transform.scale(snake_body, (board_unit_size, board_unit_size))
    snake_tail = pygame.transform.scale(snake_tail, (board_unit_size, board_unit_size))
    feed = pygame.transform.scale(feed, (board_unit_size, board_unit_size))

    # bgm load
    pygame.mixer.music.load(os.path.join(essets_path + 'bgm.wav'))

    # font
    btn_font = pygame.font.SysFont('applegothic', 30)
    game_font = pygame.font.SysFont('arialrounded', 40)

    game_env = {}
    game_env['player'] = None
    game_env['board_size'] = board_size
    game_env['board_unit_size'] = board_unit_size
    game_env['screen_width'] = screen_width
    game_env['screen_height'] = screen_height
    game_env['screen'] = screen
    game_env['clock'] = clock
    game_env['fps'] = fps
    game_env['image'] = {
        'background': background,
        'snake_head': snake_head,
        'snake_body': snake_body,
        'snake_tail': snake_tail,
        'feed': feed
    }
    game_env['font'] = {
        'btn_font': btn_font,
        'game_font': game_font
    }
    game_env['end'] = False

    return game_env


def mode_select(game_env):
    HUMAN = 'human'
    AI = 'ai'

    btn_font = game_env['font']['btn_font']
    screen_width = game_env['screen_width']
    screen_height = game_env['screen_height']
    screen = game_env['screen']

    btn_width = 300
    btn_height = 100
    player_button = Button(screen_width // 2, 2 * screen_height // 5, btn_width, btn_height, '직접 플레이(enter)', btn_font)
    ai_button = Button(screen_width // 2, 3 * screen_height // 5, btn_width, btn_height, 'AI(미구현)', btn_font)

    running = True
    while running:
        if player_button.draw(screen):
            game_env['player'] = HUMAN
            running = False
        if ai_button.draw(screen):
            # game_env['player'] = AI
            print('미구현')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    game_env['player'] = HUMAN
                    running = False

        pygame.display.update()


def play(game_env):
    board_size = game_env['board_size']
    board_unit_size = game_env['board_unit_size']
    screen_height = game_env['screen_height']
    screen_width = game_env['screen_width']
    clock = game_env['clock']
    fps = game_env['fps']
    screen = game_env['screen']

    snake_head = game_env['image']['snake_head']
    snake_body = game_env['image']['snake_body']
    snake_tail = game_env['image']['snake_tail']
    background = game_env['image']['background']
    feed = game_env['image']['feed']

    game_font = game_env['font']['game_font']

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
        render_score(str(score), game_font, screen, screen_width, screen_height)
        pygame.display.update()


def replay(game_env):
    screen_width = game_env['screen_width']
    screen_height = game_env['screen_height']
    btn_font = game_env['font']['btn_font']
    screen = game_env['screen']

    btn_width = 300
    btn_height = 100
    replay_button = Button(screen_width // 2, 2 * screen_height // 5, btn_width, btn_height, '다시하기(enter)', btn_font)
    end_button = Button(screen_width // 2, 3 * screen_height // 5, btn_width, btn_height, '끝내기(esc)', btn_font)

    running = True
    while running:
        if replay_button.draw(screen):
            running = False
        if end_button.draw(screen):
            game_env['end'] = True
            running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
                if event.key == pygame.K_ESCAPE:
                    game_env['end'] = True
                    running = False

        pygame.display.update()


if __name__ == '__main__':
    game_env = init()
    mode_select(game_env)
    while not game_env['end']:
        play(game_env)
        replay(game_env)
