import os
import struct
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from src.core.wakeword_core import WakewordCore
from src.core.chatAI import ChatAI

load_dotenv()

def main():
    print("可用的音频输入设备：")
    print(sd.query_devices())
    
    default_input_device = sd.default.device[0]
    device_info = sd.query_devices(default_input_device, 'input')
    print(f"\n默认输入设备支持的采样率：{device_info['default_samplerate']}")

    # 初始化模块
    chat = ChatAI()
    
    # 定义唤醒回调：当检测到唤醒词时，停止监听，进入对话主流程
    def on_wake():
        # 设置标志位为 False，使 WakewordCore.start() 中的循环结束并返回
        if wwc:
            wwc.is_running = False

    wwc = WakewordCore(on_wake_callback=on_wake)

    try:
        while True:
            # 1. 启动唤醒词监听 (阻塞直到唤醒词出现)
            wwc.start()
            
            # 2. 唤醒后进入对话模式
            # start() 返回主要意味着被唤醒了 (或者被中断)
            print(">>> System Woken Up! Starting Dialogue...")
            history = chat.start_dialogue()
            
            # 3. 对话结束，处理历史记录 (如果有需求)
            if history:
                print(f"[Main] Session finished. Captured {len(history)} messages.")
            
            # 循环回到步骤 1，重新监听唤醒词
            
    except KeyboardInterrupt:
        print("\nProgram exited by user.")

if __name__ == "__main__":
    main()
