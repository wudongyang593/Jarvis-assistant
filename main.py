import os
import struct
import numpy as np
import sounddevice as sd
import pvporcupine
from dotenv import load_dotenv  # 导入库
from src.core.wakeword_core import WakewordCore
WAKEWORD = 'jarvis'
SAMPLE_RATE = 16000             #16k
# Access_Key = 'xZ3FCBaG7tqxZamqjwR2oVio7gFsZuog4a3t63jNd3bWRZeeYA46Hw=='

load_dotenv()

Access_Key = os.getenv('PVPORCUPINE_ACCESS_KEY')

def main():
    print("可用的音频输入设备：")
    print(sd.query_devices())
    # 获取默认输入设备的信息
    default_input_device = sd.default.device[0]
    device_info = sd.query_devices(default_input_device, 'input')
    print(f"\n默认输入设备支持的采样率：{device_info['default_samplerate']}")
    # print(f"设备支持的所有采样率范围：{device_info['samplerates']}")

    wwc = WakewordCore()
    wwc.start()

if __name__ == "__main__":
    main()
