import pygame
import sys

# Initialize
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Xbox Controller Test")
font = pygame.font.Font(None, 36)

# Initialize joystick
if pygame.joystick.get_count() == 0:
    print("No joystick found.")
    pygame.quit()
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Controller:", joystick.get_name())

running = True
clock = pygame.time.Clock()

while running:
    pygame.event.pump()
    screen.fill((20, 20, 20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Axes
    for i in range(joystick.get_numaxes()):
        val = joystick.get_axis(i)
        txt = font.render(f"Axis {i}: {val:.2f}", True, (255, 255, 255))
        screen.blit(txt, (50, 40 + i * 30))

    # Buttons
    for i in range(joystick.get_numbuttons()):
        val = joystick.get_button(i)
        txt = font.render(f"Button {i}: {val}", True, (255, 255, 255))
        screen.blit(txt, (400, 40 + i * 30))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
