import cv2 as cv
import numpy as np
from random import randint
from math import sin, cos


class MaskGenerators:
    def __init__(
        self,
        num=0,
        height=0,
        width=0,
        channels=3,
    ) -> None:
        self._height = height
        self._width = width
        self._num = num
        self._channels = channels

    def continuous_mask(self, max_angle, max_length, maxBrushWidth):
        """Generates a continuous mask with lines, circles and elipses"""

        img = np.zeros((self._height, self._width, self._channels), np.uint8)

        for j in range(1):
            startX = randint(0, self._width)
            startY = randint(0, self._height)
            for i in range(0, randint(1, self._num)):
                angle = randint(0, max_angle)
                if i % 2 == 0:
                    angle = 360 - angle
                length = randint(1, max_length)
                brushWidth = randint(1, maxBrushWidth)
                endX = startX + int(length * sin(angle))
                endY = startY + int(length * cos(angle))
                if endX > 255:
                    endX = 255
                if endX < 0:
                    endX = 0
                if endY > 255:
                    endY = 255
                if endY < 0:
                    endY = 0
                cv.line(
                    img, (startX, startY), (endX, endY), (255, 255, 255), brushWidth
                )
                cv.circle(img, (endX, endY), brushWidth // 2, (255, 255, 255), -1)
                startY = endY
                startX = endX

        img2 = np.zeros((self._height, self._width, 1))
        img2[:, :, 0] = img[:, :, 0]
        img2[img2 > 1] = 1

        return 1 - img2

    def discontinuous_mask(self, low, high):
        """Generates a discontinuous mask with lines, circles and elipses
        When we were training, we generated more elipses
        """
        img = np.zeros((self._height, self._width, self._channels), np.uint8)

        # Set size scale
        size = int((self._width + self._height) * 0.1)
        if self._width < 64 or self._height < 64:
            raise Exception("Width and height of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, self._num)):
            x1, x2 = randint(1, self._width), randint(1, self._width)
            y1, y2 = randint(1, self._height), randint(1, self._height)
            thickness = randint(3, size)
            cv.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random circles
        for _ in range(randint(1, self._num)):
            x1, y1 = randint(1, self._width), randint(1, self._height)
            radius = randint(3, size)
            cv.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        # Draw randow rectangle
        for _ in range(randint(1, self._num)):
            x1, y1 = randint(1, self._width), randint(1, self._height)
            x2, y2 = randint(1, self._width), randint(1, self._height)
            cv.rectangle(img, (x1, y1), (x2, y2), (1, 1, 1), -1)

        # Draw random ellipses
        for _ in range(randint(1, self._num)):
            x1, y1 = randint(1, self._width), randint(1, self._height)
            s1, s2 = randint(low, high), randint(low, high)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        img2 = np.zeros((self._height, self._width, 1))
        img2[:, :, 0] = img[:, :, 0]

        return 1 - img2
