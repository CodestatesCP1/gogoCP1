import os
import numpy as np
import pygame

from gameclasses import Snake, Button, VolumeControlBar, to_direction
from agent import Agent

HUMAN = 'human'
AI = 'ai'


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
    board_size = 30
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
    volume_on_img = pygame.image.load(os.path.join(essets_path, 'volume_on.png')).convert_alpha()
    volume_off_img = pygame.image.load(os.path.join(essets_path, 'volume_off.png')).convert_alpha()

    # 이미지를 화면 크기에 맞게 scaling
    background = pygame.transform.scale(background, (screen_width, screen_height))
    snake_head = pygame.transform.scale(snake_head, (board_unit_size, board_unit_size))
    snake_body = pygame.transform.scale(snake_body, (board_unit_size, board_unit_size))
    snake_tail = pygame.transform.scale(snake_tail, (board_unit_size, board_unit_size))
    feed = pygame.transform.scale(feed, (board_unit_size, board_unit_size))
    volume_on_img = pygame.transform.scale(volume_on_img, (1.5 * board_unit_size, 1.5 * board_unit_size))
    volume_off_img = pygame.transform.scale(volume_off_img, (1.5 * board_unit_size, 1.5 * board_unit_size))
    volume_off_img.set_alpha(196)

    # bgm load
    pygame.mixer.music.load(os.path.join(essets_path + 'bgm.wav'))

    volume_control_bar = VolumeControlBar(3 * screen_width // 4, 17 * (screen_height - screen_width) // 25,
                                          volume_on_img, volume_off_img, 2 * screen_width // 15, screen_width // 100)

    # font
    btn_font = pygame.font.SysFont('applegothic', 30)
    game_font = pygame.font.SysFont('arialrounded', 40)

    game_env = {}
    game_env['player'] = None
    game_env['agent'] = None
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
        'feed': feed,
    }
    game_env['font'] = {
        'btn_font': btn_font,
        'game_font': game_font
    }
    game_env['end'] = False
    game_env['volume_control_bar'] = volume_control_bar

    return game_env


def mode_select(game_env):
    btn_font = game_env['font']['btn_font']
    screen_width = game_env['screen_width']
    screen_height = game_env['screen_height']
    screen = game_env['screen']

    btn_width = 300
    btn_height = 100
    player_button = Button(screen_width // 2, 2 * screen_height // 5, btn_width, btn_height, '직접 플레이(↑)', btn_font)
    ai_button = Button(screen_width // 2, 3 * screen_height // 5, btn_width, btn_height, 'AI(↓)', btn_font)

    def select_human():
        game_env['player'] = HUMAN

    def select_ai():
        game_env['player'] = AI
        game_env['agent'] = Agent()

    running = True
    while running:
        action_player_btn, state_player_btn = player_button.catch_mouse_action()
        action_ai_btn, state_ai_btn = ai_button.catch_mouse_action()
        if action_player_btn:
            select_human()
            running = False
        if action_ai_btn:
            select_ai()
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    select_human()
                    running = False
                elif event.key == pygame.K_DOWN:
                    select_ai()
                    running = False

        player_button.draw(screen, state_player_btn)
        ai_button.draw(screen, state_ai_btn)
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
    volume_control_bar = game_env['volume_control_bar']

    game_font = game_env['font']['game_font']

    PLAYER = game_env['player']
    agent = game_env['agent']

    # 뱀 최초 위치 랜덤 생성(벽에 너무 가까이 있으면 시작하자마자 죽을 수 있으므로 벽과 적당히 떨어져있도록 설정)
    snake_init_pos = (
        np.random.randint(board_size // 4, 3 * board_size // 4),
        np.random.randint(board_size // 4, 3 * board_size // 4))
    snake = Snake(position=snake_init_pos, board_size=board_size,
                  head_img=snake_head, body_img=snake_body, tail_img=snake_tail, img_unit=board_unit_size)

    # 먹이 위치
    feed_pos = make_feed(snake.get_body(), board_size)

    # 현재 상태 표현
    state = None
    if PLAYER == AI:
        state = np.zeros((board_size, board_size, 3), dtype=int)
        # 뱀 몸통 표시
        for x, y in snake.get_body():
            state[y][x][0] = 1
        # 뱀 머리 표시
        head_x, head_y = snake.get_body()[0]
        state[head_y][head_x][1] = 1
        # 먹이 표시
        state[feed_pos[1]][feed_pos[0]][2] = 1

    running = True
    # Left and Up (뱀이 움직일 수 있는 영역의 맨 왼쪽 위 좌표)
    LU = (0, screen_height - screen_width)
    score = 0
    pygame.mixer.music.play(-1)
    while running:
        clock.tick(fps)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

        next_action = snake.get_direction()
        # 플레이어가 사람인 경우 키보드 입력으로 다음 행동 결정
        if PLAYER == HUMAN:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        next_action = 'U'
                    elif event.key == pygame.K_LEFT:
                        next_action = 'L'
                    elif event.key == pygame.K_DOWN:
                        next_action = 'D'
                    elif event.key == pygame.K_RIGHT:
                        next_action = 'R'
        # 플레이어가 인공지능인 경우 agent 가 다음 행동 결정
        elif PLAYER == AI:
            next_action = to_direction(agent.policy(state))

        reward, new_head, old_head, popped = snake.move(next_action, feed_pos)

        old_feed_pos = None
        # 먹이 먹음
        if reward == 1:
            score += 1
            old_feed_pos = feed_pos
            feed_pos = make_feed(snake.get_body(), board_size)

        done = False
        # 죽음
        if reward == -1:
            pygame.mixer.music.stop()
            done = True

        # 상태 업데이트
        if PLAYER == AI and not done:
            state[new_head[1]][new_head[0]] = np.array([1, 1, 0])
            state[old_head[1]][old_head[0]][1] = 0
            if popped:
                state[popped[1]][popped[0]][0] = 0
            if old_feed_pos:
                state[old_feed_pos[1]][old_feed_pos[0]][2] = 0
                state[feed_pos[1]][feed_pos[0]][2] = 1

        ### 여기서 AI 학습용으로 state 갖다 쓰면 됩니다...!

        if done:
            break

        volume_control_bar.catch_mouse_action()

        screen.blit(background, (0, 0))
        volume_control_bar.render(screen)
        snake.render(screen, LU)
        screen.blit(feed, (LU[0] + feed_pos[0] * board_unit_size, LU[1] + feed_pos[1] * board_unit_size))
        render_score(str(score), game_font, screen, screen_width, screen_height)
        pygame.display.update()


def replay(game_env):
    PLAYER = game_env['player']
    if PLAYER == AI:
        return
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
        action_replay_btn, state_replay_btn = replay_button.catch_mouse_action()
        action_end_btn, state_end_btn = end_button.catch_mouse_action()
        if action_replay_btn:
            running = False
        if action_end_btn:
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

        replay_button.draw(screen, state_replay_btn)
        end_button.draw(screen, state_end_btn)
        pygame.display.update()


if __name__ == '__main__':
    game_env = init()
    mode_select(game_env)
    while not game_env['end']:
        play(game_env)
        replay(game_env)
