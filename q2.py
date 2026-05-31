import pygame
import math
import sys

# Initialize pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Table and Chair Animation")

# Colors
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
GREEN = (100, 200, 100)
BLUE = (135, 206, 235)

# Ground position
GROUND_Y = 500

# Table dimensions
TABLE_TOP = pygame.Rect(400, 350, 200, 20)
TABLE_LEGS = [
    pygame.Rect(410, 350, 10, 150),
    pygame.Rect(580, 350, 10, 150),
    pygame.Rect(410, 480, 170, 10)
]

# Chair dimensions
CHAIR_SEAT = [200, GROUND_Y - 20, 60, 20]
CHAIR_BACK = [200, GROUND_Y - 70, 20, 50]
pivot_x, pivot_y = 200, GROUND_Y  # Pivot at bottom left


# Rotation function
def rotate_point(x, y, pivot_x, pivot_y, angle):
    s, c = math.sin(angle), math.cos(angle)
    x -= pivot_x
    y -= pivot_y
    x_new = x * c - y * s
    y_new = x * s + y * c
    return x_new + pivot_x, y_new + pivot_y


# Main animation
angle = 0
falling = True
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear screen
    screen.fill(BLUE)

    # Draw ground
    pygame.draw.rect(screen, GREEN, (0, GROUND_Y, width, height - GROUND_Y))

    # Draw table
    pygame.draw.rect(screen, BROWN, TABLE_TOP)
    for leg in TABLE_LEGS:
        pygame.draw.rect(screen, BROWN, leg)

    # Rotate chair if falling - FIXED SECTION
    if falling:
        angle += 0.02
        if angle >= math.pi / 2.5:  # ~70 degrees
            falling = False
            angle = math.pi / 2.5

    # Calculate rotated chair points
    points = []
    for dx, dy in [(0, 0), (60, 0), (60, -50), (0, -50)]:
        x, y = pivot_x + dx, pivot_y + dy - 20
        points.append(rotate_point(x, y, pivot_x, pivot_y, angle))

    # Draw chair
    pygame.draw.polygon(screen, LIGHT_BROWN, points)  # Seat
    back_points = [
        rotate_point(pivot_x + 40, pivot_y - 20, pivot_x, pivot_y, angle),
        rotate_point(pivot_x + 60, pivot_y - 20, pivot_x, pivot_y, angle),
        rotate_point(pivot_x + 60, pivot_y - 70, pivot_x, pivot_y, angle),
        rotate_point(pivot_x + 40, pivot_y - 70, pivot_x, pivot_y, angle)
    ]
    pygame.draw.polygon(screen, BROWN, back_points)  # Backrest

    # Update display
    pygame.display.flip()
    clock.tick(60)