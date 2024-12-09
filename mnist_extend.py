from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random

fonts = ["arial.ttf", "Comfortaa-Bold.ttf", "Helvetica.ttf", "hline.ttf", "NotoSans-Regular.ttf", "Trebuchet_MS.ttf", "VeraMoBd.ttf", "Vera.ttf"]
# fonts = ["arial.ttf", "Comfortaa-Bold.ttf"]

def generateArialNumberAndSave(number, n):
    os.makedirs(f"images/{number}", exist_ok=True)
    for font_idx, font_name in enumerate(fonts):
        for idx in range(n):
            image = Image.new("RGB", (28, 28), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            fontsize = 24
            x_position = 6
            y_position = 2
            if idx != 0:
                fontsize = random.randint(22,25)
                x_position = random.randint(5, 8)
                y_position = random.randint(2, 3)
            font = ImageFont.truetype(font_name, fontsize)  # Adjust font size if needed
            bbox = draw.textbbox((0, 0), number, font=font)


            # right line
            # if random.randint(0,1):
            #     shape = [(28,0), (28,28)]
            #     draw.line(shape, fill="black", width=random.randint(2,4))
            #     # top line
            # if random.randint(0, 1):
            #     shape = [(0, 0), (28, 0)]
            #     draw.line(shape, fill="black", width=random.randint(2, 4))
            # # bottom line
            # if random.randint(0, 1):
            #     shape = [(0, 28), (28, 28)]
            #     draw.line(shape, fill="black", width=random.randint(2, 4))
            #     # bottom line
            # if random.randint(0, 1):
            #     shape = [(0, 0), (0, 28)]
            #     draw.line(shape, fill="black", width=random.randint(2, 4))
            draw.text((x_position, y_position), number, font=font, fill=(0, 0, 0))  # Black text
            # if idx > n//2:
            image = image.filter(ImageFilter.GaussianBlur(radius = 0.8))
            image.save(f"images/{number}/{number}_{font_idx}_{idx}.jpg")


N = 50
for i in ["0","1","2","3","4","5","6","7","8","9"]:
    generateArialNumberAndSave(i,N)