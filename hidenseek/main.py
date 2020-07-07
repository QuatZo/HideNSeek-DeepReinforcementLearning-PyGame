import pygame
import sys

from ext.engine import HideNSeek

if __name__ == "__main__":
    fps = 60
    pygame.init()
    players_speed_ratio = {
        'p_hide': .9,
        'p_seek': .91,
    }

    game = HideNSeek(width=512, height=512, fps=fps, speed_ratio=players_speed_ratio, speed_multiplier=1000)
    game.setup()
    game.init()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or game.game_over():
                pygame.quit()
                sys.exit()
        ##### temporarily, before agents
        if(game.game_over()):
            break
        #####
        dt = game.clock.tick_busy_loop(fps)
        game.step(dt)
        pygame.display.update()
