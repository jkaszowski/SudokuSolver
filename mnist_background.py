from PIL import Image, ImageDraw
import random
import os

def generateBackground(N):
    # Save the image as a JPG file
    for i in range(N):
        img = Image.new("RGB", (28, 28), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # line_pos = random.randint(20,28)
        # shape = [(line_pos,0), (line_pos,28)]
        # drawer.line(shape, fill="black", width=random.randint(2,3))
        # right line
        # if random.randint(0, 1):
        #     shape = [(26, 0), (26, 28)]
        #     draw.line(shape, fill="black", width=random.randint(2, 3))
        #     # top line
        # if random.randint(0, 1):
        #     shape = [(0, 2), (28, 2)]
        #     draw.line(shape, fill="black", width=random.randint(2, 3))
        # # bottom line
        # if random.randint(0, 1):
        #     shape = [(0, 28), (28, 28)]
        #     draw.line(shape, fill="black", width=random.randint(2, 3))
        img.save(f"images/background/background_{i}.jpg")

if __name__ == "__main__":
    os.makedirs(f"images/background", exist_ok=True)
    generateBackground(100)