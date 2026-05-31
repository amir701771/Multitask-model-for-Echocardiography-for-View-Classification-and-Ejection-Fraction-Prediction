import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def dda_line(x1, y1, x2, y2):
    # DDA algorithm implementation
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    x_increment = dx / steps
    y_increment = dy / steps

    line_pixels = []
    x, y = x1, y1
    for _ in range(steps):
        line_pixels.append((int(round(x)), int(round(y))))
        x += x_increment
        y += y_increment

    return np.array(line_pixels)


def bresenham_line(x1, y1, x2, y2):
    # Bresenham algorithm implementation
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = y2 - y1
    p = 2 * dy - dx
    y = y1

    line_pixels = []
    for x in range(x1, x2 + 1):
        if steep:
            line_pixels.append((y, x))
        else:
            line_pixels.append((x, y))

        if p >= 0:
            y += 1 if y1 < y2 else -1
            p -= 2 * dx
        p += 2 * dy

    return np.array(line_pixels)


def yu_line(x1, y1, x2, y2):
    # Yu's algorithm implementation
    line_pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        line_pixels.append((x1, y1))
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    line_pixels.append((x2, y2))
    return np.array(line_pixels)


def gaussian_aliasing(line_pixels):
    line_pixels = np.array(line_pixels)
    x, y = line_pixels[:, 0], line_pixels[:, 1]
    aliased_x = gaussian_filter(x, sigma=1)
    aliased_y = gaussian_filter(y, sigma=1)
    aliased_line_pixels = np.column_stack((aliased_x, aliased_y))
    return aliased_line_pixels


def conical_aliasing(line_pixels):
    conical_line = []
    for x, y in line_pixels:
        conical_line.extend([(x - 1, y - 1), (x, y), (x + 1, y - 1), (x, y - 1), (x - 1, y)])
    return np.array(conical_line)


def broadline_aliasing(line_pixels):
    broad_line = []
    for x, y in line_pixels:
        broad_line.extend([(x - 2, y), (x - 1, y), (x, y), (x + 1, y), (x + 2, y)])
    return np.array(broad_line)


def weighted_line_aliasing(line_pixels):
    weighted_line = []
    for x, y in line_pixels:
        weighted_line.extend([(x - 1, y), (x, y), (x, y), (x + 1, y)])
    return np.array(weighted_line)


def line_drawing_with_aliasing(x1, y1, x2, y2, line_algo_index, aliasing_index):
    if line_algo_index == 1:
        line_pixels = dda_line(x1, y1, x2, y2)
    elif line_algo_index == 2:
        line_pixels = bresenham_line(x1, y1, x2, y2)
    elif line_algo_index == 3:
        line_pixels = yu_line(x1, y1, x2, y2)
    else:
        print("Invalid line algorithm index")

    aliased_line_pixels = line_pixels

    if aliasing_index == 1:  # Gaussian aliasing
        aliased_line_pixels = gaussian_aliasing(line_pixels)
    elif aliasing_index == 2:  # Conical aliasing
        aliased_line_pixels = conical_aliasing(line_pixels)
    elif aliasing_index == 3:  # Broadline aliasing
        aliased_line_pixels = broadline_aliasing(line_pixels)
    elif aliasing_index == 4:  # Weighted line aliasing
        aliased_line_pixels = weighted_line_aliasing(line_pixels)
    else:
        print("Invalid aliasing index")

    return aliased_line_pixels


def main():
    x1, y1 = 10, 20
    x2, y2 = 150, 100
    line_algo_index = 3  # Breshenham algorithm
    aliasing_index = 4  # Conical aliasing

    aliased_line_pixels = line_drawing_with_aliasing(x1, y1, x2, y2, line_algo_index, aliasing_index)

    # Display the line with aliasing
    plt.plot(aliased_line_pixels[:, 0], aliased_line_pixels[:, 1], marker='o')
    plt.show()


if __name__ == "__main__":
    main()
