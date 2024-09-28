import numpy as np
import random


def cut(deck, portion, stdev):
    rng = np.random.default_rng()
    point = int(rng.normal(portion, stdev) * len(deck))
    return split(deck, point)


# returns two decks split at exact point
# the number split indicates the index of the first card in the resulting bottom deck
def split(deck, point):
    if point <= 0:
        top = []
        bottom = deck
    elif point > len(deck) - 1:
        top = deck
        bottom = []
    else:
        top = deck[:point]
        bottom = deck[point:]

    return top, bottom


# combines two decks into one, by placing the first deck ontop of the second
def stack(top, bottom):
    deck = np.append(top, bottom)
    return deck


# two decks to riffle
# stickiness defines the likeliness of the next card coming from the same
# deck as the card preceding it - stickiness of 0 = perfect riffle, stickiness of 1 = stack.
def riffle(left, right, stickiness):
    deck = []
    l = True
    if random.random() > 0.5:
        l = False
    left_pointer = 0
    right_pointer = 0

    while left_pointer < len(left) and right_pointer < len(right):
        if l:
            deck.append(left[left_pointer])
            left_pointer += 1
        else:
            deck.append(right[right_pointer])
            right_pointer += 1
        if random.random() > stickiness:
            l = not l

    if left_pointer < len(left):
        deck.extend(left[left_pointer:])
    elif right_pointer < len(right):
        deck.extend(right[right_pointer:])

    return np.array(deck)


# strip shuffle - cut the deck into n roughly equal piles and
# then stack in reverse order
def box(deck, strips=6, stdev=0.05):
    portion = len(deck) / strips
    pile = []

    for i in range(strips - 1):
        if not len(deck) == 0:
            a, deck = cut(deck, portion / len(deck), stdev)
            pile = stack(a, pile)
    pile = stack(deck, pile)
    pile = [int(x) for x in pile]

    return pile


def div(a, b):
    return a / b if b else 1


def identity(deck):
    return deck


# custom shuffle for 8 decks
def custom(deck, stickiness=0.3):
    a, b = cut(deck, 0.5, 0)
    pile = []
    portion = deck.size / 16

    for j in range(2):
        for i in range(7):
            c, a = cut(a, div(portion, len(a)), 0.025)
            d, b = cut(b, div(portion, len(b)), 0.025)
            pile = stack(pile, riffle(c, d, stickiness))

        pile = stack(pile, riffle(a, b, stickiness))

        a, b = cut(pile, 0.5, 0.025)
        pile = []

    for i in range(7):
        c, a = cut(a, div(portion, len(a)), 0.025)
        d, b = cut(b, div(portion, len(b)), 0.025)

        temp = riffle(c, d, stickiness)
        temp = box(temp, 6, 0.025)
        c, d = cut(temp, 0.5, 0.025)
        pile = stack(pile, riffle(c, d, stickiness))

    pile = stack(pile, riffle(a, b, stickiness))

    pile = [int(x) for x in pile]

    return pile
