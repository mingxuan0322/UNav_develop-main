import numpy as np
import pygsheets

def load_data(start, end):
    gc = pygsheets.authorize(service_file='/home/endeleze/Desktop/UNav/Localization/unav-342020-d0b2954b5825.json')
    sh = gc.open('UNav_test')
    wks = sh[1]
    return np.array(wks.get_values_batch([start + ':' + end])).squeeze().astype(float)

def calculate(A):
    n,m=A.shape
    sum=0
    num=0
    for i in A:
        for j in range(m):
            for k in range(j+1,m):
                if i[j]<=i[k]:
                    sum+=1
                num+=1
    return sum/num

def main():
    data=load_data('O47','T63')
    v=calculate(data)
    print('probability of denser map that is equal or better than sparse map (rotation downsampling):\t%04f'%v)
    data = load_data('O26', 'X42')
    v = calculate(data)
    print('probability of denser map that is equal or better than sparse map (frame downsampling):\t%04f' % v)
    data = load_data('Z26', 'AH42')
    v = calculate(data)
    print('probability of denser map that is equal or better than sparse map (frame downsampling):\t%04f' % v)
if __name__ == '__main__':
    main()