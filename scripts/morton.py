#!/usr/bin/env python
# Taken from https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints/18528775#18528775

import random

def prettyBinString(x, d=32, steps=4, sep=".", emptyChar="0"):
    b = bin(x)[2:]
    zeros = d - len(b)

    if zeros <= 0:
        zeros = 0
        k = steps - (len(b) % steps)
    else:
        k = steps - (d % steps)

    s = ""
    # print("zeros" , zeros)
    # print("k" , k)
    for i in range(zeros):
        # print("k:",k)
        if k % steps == 0 and i != 0:
            s += sep
        s += emptyChar
        k += 1

    for i in range(len(b)):
        if (k % steps == 0 and i != 0 and zeros == 0) or (
            k % steps == 0 and zeros != 0
        ):
            s += sep
        s += b[i]
        k += 1
    return s


def binStr(x):
    return prettyBinString(x, 32, 4, " ", "0")


def computeBitMaskPatternAndCode(numberOfBits, numberOfEmptyBits):
    bitDistances = [i * numberOfEmptyBits for i in range(numberOfBits)]
    bitDistancesB = [bin(dist)[2:] for dist in bitDistances]
    moveBits = ([])

    maxLength = len(max(bitDistancesB, key=len))
    for i in range(maxLength):
        moveBits.append([])
        for idx, bits in enumerate(bitDistancesB):
            if not len(bits) - 1 < i:
                if bits[len(bits) - i - 1] == "1":
                    moveBits[i].append(idx)

    bitPositions = list(range(numberOfBits))
    maskOld = (1 << numberOfBits) - 1

    codeString = "x &= " + hex(maskOld) + "\n"
    for idx in range(len(moveBits) - 1, -1, -1):
        if len(moveBits[idx]):

            shifted = 0
            for bitIdxToMove in moveBits[idx]:
                shifted |= 1 << bitPositions[bitIdxToMove]
                bitPositions[bitIdxToMove] += 2 ** idx
                # keep track where the actual bit stands! might get moved
                # several times

            # Get the non shifted part!
            nonshifted = ~shifted & maskOld

            shifted = shifted << 2 ** idx
            maskNew = shifted | nonshifted

            codeString += (
                "x = (x | x << " + str(2 ** idx) + ") & " + hex(maskNew) + "\n"
            )
            maskOld = maskNew
    return codeString


numberOfBits = 21
numberOfEmptyBits = 2
codeString = computeBitMaskPatternAndCode(numberOfBits, numberOfEmptyBits)
print(codeString)
