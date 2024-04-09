import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont, QTransform, QPixmap, QPainter, QImage
import cv2
import numpy as np
from skimage import io as io2
from PyQt5.uic import loadUi
import threading
import time
#import resources
import sys
from PIL import Image
import asyncio
import io
from winrt.windows.media.control import \
    GlobalSystemMediaTransportControlsSessionManager as MediaManager

from winrt.windows.storage.streams import \
    DataReader, Buffer, InputStreamOptions


async def read_stream_into_buffer(stream_ref, buffer):
    readable_stream = await stream_ref.open_read_async()
    readable_stream.read_async(buffer, buffer.capacity, InputStreamOptions.READ_AHEAD)

async def get_media_info():
    sessions = await MediaManager.request_async()

    # This source_app_user_model_id check and if statement is optional
    # Use it if you want to only get a certain player/program's media
    # (e.g. only chrome.exe's media not any other program's).

    # To get the ID, use a breakpoint() to run sessions.get_current_session()
    # while the media you want to get is playing.
    # Then set TARGET_ID to the string this call returns.

    current_session = sessions.get_current_session()
    if current_session:  # there needs to be a media session running

            info = await current_session.try_get_media_properties_async()

            # song_attr[0] != '_' ignores system attributes
            info_dict = {song_attr: info.__getattribute__(song_attr) for song_attr in dir(info) if song_attr[0] != '_'}

            # converts winrt vector to list
            info_dict['genres'] = list(info_dict['genres'])
            infob = current_session.get_timeline_properties()
            info_dict2 = {song_attr: infob.__getattribute__(song_attr) for song_attr in dir(infob) if
                         song_attr[0] != '_'}
            endt = {song_attr: info_dict2["end_time"].__getattribute__(song_attr) for song_attr in
                    dir(info_dict2["end_time"]) if
                    song_attr[0] != '_'}
            start = {song_attr: info_dict2["position"].__getattribute__(song_attr) for song_attr in
                     dir(info_dict2["position"]) if
                     song_attr[0] != '_'}

            total_seconds = start["duration"] * 100 / 1e9

            # Extract minutes and seconds
            minutesa = int(total_seconds // 60)
            secondsa = int(total_seconds % 60)
            total_seconds = endt["duration"] * 100 / 1e9

            # Extract minutes and seconds
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            info_dict["currenttime"] = f"{minutesa}:{secondsa:02d}/{minutes}:{seconds:02d}"
            return info_dict

    # It could be possible to select a program from a list of current
    # available ones. I just haven't implemented this here for my use case.
    # See references for more information.
    raise Exception('TARGET_PROGRAM is not the current media session')
# Open original image and extract the alpha channel



def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)



sys.excepthook = except_hook
def spinthread(data):
    print(data)
    xa=0

    thumb_read_buffer = Buffer(5000000)
    while True:
        xa=xa+1
        time.sleep(2)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            current_media_info = loop.run_until_complete(get_media_info())
            loop.close()
        except:
            print(":(")

        data.findChild(QLabel, "label").setText(current_media_info["title"])
        data.findChild(QLabel, "label_4").setText(current_media_info["artist"])
        thumb_stream_ref = current_media_info['thumbnail']

        # 5MB (5 million byte) buffer - thumbnail unlikely to be larger

        try:
            asyncio.run(read_stream_into_buffer(thumb_stream_ref, thumb_read_buffer))
            buffer_reader = DataReader.from_buffer(thumb_read_buffer)
            byte_buffer = buffer_reader.read_bytes(thumb_read_buffer.length)



            #time.sleep(0.1)
            img = io2.imread(io.BytesIO(bytearray(byte_buffer)))[:, :, :-1]
            average = img.mean(axis=0).mean(axis=0)
            pixels = np.float32(img.reshape(-1, 3))

            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS

            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts)]
            colorpallete = {}
            x=0
            for i in palette:
                colorpallete[counts[x]] = palette[x]
                x=x+1
            dominant = colorpallete[sorted(colorpallete)[::-1][0]]
            if dominant[0]+dominant[1]+dominant[2] < 100:
                dominant = colorpallete[sorted(colorpallete)[::-1][1]]
            im = Image.open('white.png')
            imb = Image.open('black.png').convert("RGBA")
            cover = Image.open(io.BytesIO(bytearray(byte_buffer))).convert("RGBA")
            #cover = cover.rotate(xa, Image.NEAREST, expand = 0)
            coverhoilder = Image.new('RGBA', imb.size, color=(0, 0, 0, 0))
            shrinkx = 00
            shrinky = 00
            cover = cover.resize((imb.size[0] - shrinkx, imb.size[0] - shrinky), 1)
            coverhoilder.paste(cover, (int(shrinkx / 2), int(shrinky / 2)))
            # cover.resize(imb.size, Image.Resampling.LANCZOS)
            alpha = im.getchannel('A')
            alphab = imb.getchannel('A')
            coverhoilder.putalpha(alphab)
            # Create red image the same size and copy alpha channel across
            red = Image.new('RGBA', im.size, color=(int(dominant[0]),int(dominant[1]),int(dominant[2])))

            red.putalpha(alpha)
            red.paste(coverhoilder, (0, 0), coverhoilder)
            im = red.convert("RGBA")

            datab = im.tobytes("raw", "RGBA")
            qim = QImage(datab, im.size[0], im.size[1], QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qim).scaled(231, 201)
            data.findChild(QLabel, "label_3").setPixmap(pix)  # image for your label
            data.findChild(QLabel, "label_5").setText(current_media_info["currenttime"])



        except:
            print(":(")
        #print(1)
        '''
        qim = ImageQt(im)
        pix = QtGui.QPixmap.fromImage(qim)
        data.pixmap = pix  # image for your label
        painter = QPainter(data.pixmap)
        pixmap_rotated = data.pixmap.transformed(QTransform().rotate(x))
        pixmap_rotated = pixmap_rotated.scaled(231, 201)


        painter.setPixmap(pixmap_rotated)
        rect = pixmap_rotated.rect()
        painter.end()
        data.setFixedSize(rect.size())
        '''

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        loadUi(r'main.ui', self)
        self.findChild(QLabel, "label_3")

        # Connect signals to slots or do any other initialization if needed
        my_thread = threading.Thread(target=spinthread, args=(self,))

        # Start the thread
        my_thread.start()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())