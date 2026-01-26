import os
import pvporcupine
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from typing import Callable, Optional

class WakewordCore:
    """
    语音唤醒检测核心模块
    负责监听麦克风音频流，利用 pvporcupine 检测唤醒词
    """
    def __init__(self, on_wake_callback: Optional[Callable] = None):
        """
        初始化唤醒词检测器
        
        Args:
            on_wake_callback: 当检测到唤醒词时触发的回调函数
        """
        # 加载环境变量
        load_dotenv()
        
        self.access_key = os.getenv('PVPORCUPINE_ACCESS_KEY')
        if not self.access_key:
            raise ValueError("Missing 'PVPORCUPINE_ACCESS_KEY' in environment variables.")

        # 获取唤醒词，默认为 'jarvis'
        # 注意：自定义唤醒词需要对应的模型文件，内置关键词如 'jarvis', 'porcupine' 可直接使用
        self.wakeword = os.getenv('WAKEWORD', 'jarvis')
        
        self.on_wake_callback = on_wake_callback
        self.porcupine = None
        self.is_running = False

    def start(self):
        """
        启动唤醒词监听服务 (阻塞运行)
        """
        try:
            # 初始化 Porcupine 实例
            # keywords 参数接受列表，这里暂时只监听一个
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.wakeword]
            )
            
            print(f"\n[WakewordCore] System initialized.")
            print(f"[WakewordCore] Listenning for wake word: '{self.wakeword}'")
            print(f"[WakewordCore] Sample Rate: {self.porcupine.sample_rate}")
            print(f"[WakewordCore] Status: Ready. Say '{self.wakeword}' to wake me up.\n")

            # 配置音频流参数
            # Porcupine 要求 16-bit PCM 音频，单声道
            # 使用 sounddevice 的 callback 模式处理音频流
            with sd.InputStream(
                channels=1,
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype='int16',  # 直接以 16-bit int 读取，避免手动转换
                callback=self._audio_callback
            ):
                self.is_running = True
                while self.is_running:
                    sd.sleep(100) # 保持主线程活跃，每100ms检查一次
                    
        except KeyboardInterrupt:
            print("\n[WakewordCore] Stopping by user request...")
        except Exception as e:
            print(f"\n[WakewordCore] Error occurred: {e}")
        finally:
            self._cleanup()

    def _audio_callback(self, indata, frames, time, status):
        """
        Sounddevice 音频流回调函数
        """
        if status:
            print(f"[Audio Error] {status}")

        if self.porcupine is None:
            return

        # indata 是 (frames, channels) 的 numpy array
        # Porcupine process 需要一维 flattened array
        frame = indata[:, 0]

        try:
            # 执行检测
            keyword_index = self.porcupine.process(frame)
            
            # process 返回检测到的关键词索引，-1 表示未检测到
            # 只要 index >= 0 即表示检测成功
            if keyword_index >= 0:
                print(f"✨ Wake word detected! (Index: {keyword_index})")
                
                # 触发唤醒回调
                if self.on_wake_callback:
                    self.on_wake_callback()
                    
        except Exception as e:
            print(f"[WakewordCore] Processing error: {e}")

    def _cleanup(self):
        """释放资源"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            print("[WakewordCore] Resources released.")

# -------------------------------------------------------------------------
# 这里是为了单独运行该文件进行测试的代码
# -------------------------------------------------------------------------
# if __name__ == '__main__':
#     def test_wake_action():
#         print(">>> 🤖 [System] : I am awake! Starting conversation logic... <<<")
#         # 在实际项目中，这里会调用对话系统的启动函数
    
#     wakeword_engine = WakewordCore(on_wake_callback=test_wake_action)
#     wakeword_engine.start()
