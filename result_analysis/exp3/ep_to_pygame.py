import pygame, sys
from pygame.locals import *

block_size = 20
length = 21

black = (0, 0, 0)
white = (255, 255, 255)

height = block_size + 2
width = (block_size+1) * length

pygame.init()
window = pygame.display.set_mode((width, height))

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    window.fill(black)

    # Draw rail.
    for y in range(21):
        rect = pygame.Rect(y*(block_size+1), 1, block_size, block_size)
        pygame.draw.rect(window, white, rect)

    pygame.display.flip()


