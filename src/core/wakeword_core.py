import os
import struct
import numpy as np
import sounddevice as sd
import pvporcupine
from dotenv import load_dotenv  # 导入库
# WAKEWORD = ''
# SAMPLE_RATE = 0             #16k
# Access_Key = ""
class WakewordCore:
    def __init__(self):
        load_dotenv()
        self.Access_Key = os.getenv('PVPORCUPINE_ACCESS_KEY')
        self.WAKEWORD = os.getenv('WAKEWORD')
        self.SAMPLE_RATE = int(os.getenv('SAMPLE_RATE'))

        print(f'accesskey:{self.Access_Key},Wakeword:{self.WAKEWORD},Sample_rate:{self.SAMPLE_RATE}')

    def start(self):
        print(f'start dev')
        pvp = pvporcupine.create(access_key=self.Access_Key, keywords=[self.WAKEWORD])
        frame_length = pvp.frame_length
        print(f"wakeword:{self.WAKEWORD},frame_length:{frame_length}")

        def callback(indata,frames,time,status):
            if status:
                print(f'status:{status},frames:{frames},time:{time}')

            pmc = (indata[:,0]*32767).astype(np.int16)
            keyword_index = pvp.process(pmc)
            if keyword_index>0:
                print(f'wakeword detected!')

        with sd.InputStream(
            channels=1,
            samplerate=self.SAMPLE_RATE,
            blocksize=frame_length,
            dtype="float32",
            callback=callback,
        ):
            while True:
                sd.sleep(1000)
