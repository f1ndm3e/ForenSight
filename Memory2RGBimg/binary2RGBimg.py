#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Transform binary file to a RGB PNG image by directory path, the image has
 the same name with the binary file.
"""

import time
import math
import os
import sys
from numba import jit

# pip install Pillow
from PIL import Image, ImageDraw

class FileData:
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

@jit(nopython=True, cache=True)
def determine_size(data):
    # size = int(math.sqrt(len(data)) + 1)
    num_bytes = len(data)
    num_pixels = int(math.ceil(float(num_bytes) / 3.0))
    sqrt = math.sqrt(num_pixels)
    size = int(math.ceil(sqrt))
    return size,size

@jit(nopython=True, cache=True)
def calccolor(byteval):
    return (
        ((byteval & 0o300) >> 6) * 64,
        ((byteval & 0o070) >> 3) * 32,
        (byteval & 0o007) * 32,
    )


def bin2img(data):
    colorfunc =  calccolor
    xsize, ysize = size = determine_size(data)
    print("size is :"+ str(xsize)+", "+str(ysize))
    img = Image.new("RGB", size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    print("Draw file begin!")
    try:
        i = 0
        for y in range(ysize):
            for x in range(xsize):
                draw.point((x, y), fill=colorfunc(data[i]))
                i += 1

    except IndexError:
        pass
    print("Draw file end!")
    return img


def error(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

def generate_image(infile):
    with open(infile, "rb") as f:
        data = f.read()
        print("read file success!")
    return bin2img(data)


if __name__ == "__main__":

    dumpath = r"Your bin file <directory name>"
    imagepath = r"Your RGB img file <directory name>"

    dumps = [f for f in os.listdir(dumpath) if f.endswith(".core")]

    for dumpfile in dumps:
        start = time.time()
        print("\n\nPROCESSING DUMP:", dumpfile)

        sourcepath = os.path.join(dumpath, dumpfile)

        targetname = os.path.splitext(dumpfile)[0] + ".png"
        targetpath = os.path.join(imagepath, targetname)

        img = generate_image(sourcepath)
        print(f'Image generated from "{sourcepath}"')

        img.save(targetpath, "PNG", compress_level=9)
        print(f'Image stored at "{targetpath}"')

        end = time.time()
        print("Time:", end - start)