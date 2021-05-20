from random import randint


def run():
    with open('data.txt', 'w+') as f:
        for i in range(10000):
            ege = randint(180, 300)
            eng = randint(0, 100)
            lang = 'java' if randint(0, 1) == 0 else '#C'
            course = randint(1, 5)
            f.write('%d %d %d %s %d\n' % (i + 1, ege, eng, lang, course))


if __name__ == '__main__':
    run()
