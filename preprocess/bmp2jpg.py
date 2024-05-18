import os.path
from PIL import Image
from io import BytesIO


def scanDir(dirPath: str) -> list:
    """扫描目录"""
    fileList = []
    for top, dirs, nondirs in os.walk(dirPath):
        for item in nondirs:
            if item.split('.')[-1] == 'bmp':
                fileList.append(os.path.join(top, item))
    return fileList


def bmp2jpg(filePath):
    fileLst = scanDir(filePath)
    print(filePath)
    for oneFile in fileLst:
        bmp = Image.open(oneFile)
        output_buffer = BytesIO()
        bmp.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        a = os.path.splitext(oneFile)[0]
        a = a.replace(filePath, jpg_path)
        tmpUrl = f'{a}.jpg'
        print(tmpUrl)
        try:
            with open(tmpUrl, 'wb') as f:
                f.write(byte_data)
        except Exception as e:
            pass


if __name__ == '__main__':
    # bmp图片的地址
    filePath = r'E:\train_image'
    # jpg图片的存放地址
    jpg_path = r'E:\train_image\jpg'
    bmp2jpg(filePath)
