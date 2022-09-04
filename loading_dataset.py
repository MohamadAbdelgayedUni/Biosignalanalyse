from glob import glob

path = "./Source_Data/training-a"  # expanding ./Source_Data/training-a.zip
filesPaths = glob(path + '/*.wav')
filesPaths.sort()

if __name__ == '__main__':
    for filePath in filesPaths:
        fileName = filePath.split('/')[-1]
        print('filePath: ', filePath)
        print('fileName: ', fileName, '\n')

    print('The folder contains', len(filesPaths), 'wav files.')
