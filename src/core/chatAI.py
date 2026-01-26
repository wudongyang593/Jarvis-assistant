import time
import os
import collections
import queue
import wave
import struct
import numpy as np
import sounddevice as sd
import webrtcvad
from typing import List, Optional, Generator

class ChatAI:
    """
    负责处理唤醒后的对话逻辑
    集成 VAD (Voice Activity Detection) 实现自动听写切分
    """
    def __init__(self):
        # 退出对话的关键词列表
        self.exit_keywords = ["谢谢", "再见", "结束", "退下", "exit", "quit", "bye"]
        # 对话超时时间（秒）
        self.timeout_seconds = 20 # 增加超时时间，适应语音交互
        # 连续无效输入的允许次数
        self.max_invalid_inputs = 3
        
        # Audio / VAD 配置
        self.sample_rate = 16000
        self.frame_duration_ms = 30  # 10, 20, or 30ms
        self.vad_aggressiveness = 3  # 0-3, 3 is most aggressive in filtering non-speech
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # VAD 算法参数
        self.padding_duration_ms = 300  # 语音开始/结束前后的缓冲时长
        self.frame_prop_duration_ms = self.frame_duration_ms 
        
        # 录音参数
        self.channels = 1
        self.dtype = 'int16'
        self.block_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
    def start_dialogue(self) -> List[dict]:
        """
        开始对话流程
        """
        print("\n" + "="*30)
        print("🤖 Jarvis: I'm listening... (Speak to microphone)")
        print("="*30 + "\n")

        conversation_history = []
        last_input_time = time.time()
        invalid_input_count = 0

        while True:
            try:
                # 1. 检查超时
                if time.time() - last_input_time > self.timeout_seconds:
                    print(f"\n[System] Timeout: No interaction for {self.timeout_seconds}s. Going back to sleep.")
                    break

                # 2. 监听并获取文本 (VAD -> Speech -> ASR -> Text)
                text_input = self._listen_and_transcribe()

                # 3. 处理无效输入 (未检测到语音 或 ASR为空)
                if not text_input:
                    # 如果只是没听清，暂时不计入严格的无效次数，或者可以宽松处理
                    # 这里简单的逻辑：如果连续多次啥都没听到，可能用户走了
                    invalid_input_count += 1
                    if invalid_input_count >= self.max_invalid_inputs:
                        print("\n[System] Too many failed attempts. Ending conversation.")
                        break
                    continue
                
                # 有效输入，重置计数器
                last_input_time = time.time()
                invalid_input_count = 0
                
                print(f"User: {text_input}")

                # 4. 检查是否包含结束词
                if self._check_exit_intent(text_input):
                    print("🤖 Jarvis: Goodbye!")
                    break

                # 5. 生成回复 (Mock LLM)
                ai_response = self._process_response(text_input)
                print(f"🤖 Jarvis: {ai_response}")

                # 记录对话
                conversation_history.append({"role": "user", "content": text_input})
                conversation_history.append({"role": "ai", "content": ai_response})
                
                # TODO: 这里添加 TTS (Text-to-Speech) 播放回复
                # self._play_audio(ai_response)

            except KeyboardInterrupt:
                print("\n[System] Interrupted by user.")
                break

        print("\n" + "="*30)
        print("😴 Jarvis: Entering sleep mode...")
        print("="*30 + "\n")
        
        return conversation_history

    def _listen_and_transcribe(self) -> str:
        """
        核心流程：监听麦克风 -> VAD切分语音 -> ASR识别 -> 返回文本
        """
        print(">> Listening...", end="", flush=True)
        
        # 1. 录制语音片段 (阻塞直到说话结束)
        audio_data = self._record_speech_segment()
        
        if not audio_data:
            print(" [Silence detected]")
            return ""
            
        print(f" [Captured {len(audio_data)} bytes audio]")
        
        # 2. 语音转文字 (ASR)
        text = self._asr_engine(audio_data)
        return text

    def _record_speech_segment(self) -> bytes:
        """
        使用 VAD 录制一段有效的语音
        逻辑：
        - 持续读取音频流
        - 维护一个环形缓冲区 (RingBuffer) 存储最近的音频帧
        - 当检测到触发状态 (Triggered) 时，开始录制
        - 当连续静音超过一定时长，停止录制
        """
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        
        triggered = False
        voiced_frames = []
        
        # 沉默帧计数器，用于判断语音结束
        silent_frame_count = 0
        max_silent_frames = int(800 / self.frame_duration_ms) # 停止前的最大静音时长 (例如 800ms)

        # 使用 RawInputStream 读取原始字节流
        with sd.RawInputStream(
            samplerate=self.sample_rate, 
            blocksize=self.block_size, 
            dtype=self.dtype, 
            channels=self.channels
        ) as stream:
            
            # 最大录音时长保护 (例如 15秒)
            max_frames = int(15000 / self.frame_duration_ms)
            frame_count = 0
            
            while True:
                # 读取音频块
                data, overflow = stream.read(self.block_size)
                if overflow:
                    pass # 忽略溢出警告

                # VAD 检测
                is_speech = self.vad.is_speech(data, self.sample_rate)

                if not triggered:
                    ring_buffer.append((data, is_speech))
                    
                    # 触发逻辑：如果环形缓冲区中超过 90% 的帧是语音，则触发开始
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        print("\n[Speech Detected] Recording...", end="", flush=True)
                        # 将缓冲区的内容加入录制列表
                        for f, s in ring_buffer:
                            voiced_frames.append(f)
                        ring_buffer.clear()
                else:
                    # 已触发，正在录制
                    voiced_frames.append(data)
                    frame_count += 1
                    
                    if is_speech:
                        silent_frame_count = 0 
                    else:
                        silent_frame_count += 1
                    
                    # 结束逻辑 1: 连续静音足够长
                    if silent_frame_count > max_silent_frames:
                        print(" [End of speech]")
                        break
                    
                    # 结束逻辑 2: 达到最大时长
                    if frame_count > max_frames:
                        print(" [Max duration reached]")
                        break
        
        # 如果录制的帧数太少（例如只是一个噪音），则忽略
        if len(voiced_frames) < 10:
            return b""
            
        return b''.join(voiced_frames)

    def _asr_engine(self, audio_data: bytes) -> str:
        """
        Mock ASR 引擎
        TODO: 在这里集成实际的 ASR 模型 (如 OpenAI Whisper, Google Speech Recognition 等)
        """
        # 为了演示，我们可以把音频保存下来，方便调试
        self._save_wav(audio_data, "last_speech.wav")
        
        # 返回模拟文本
        # 实际项目中，在这里调用： return whisper_model.transcribe("last_speech.wav")['text']
        print("\n[ASR] (Simulating recognition...)")
        
        # 暂时返回固定文本用于测试多轮对话
        # 你可以在这里让它稍微随机一点，或者根据录音长度变化
        if len(audio_data) < 32000: # 很短的声音
            return ""
            
        return "你好 Jarvis，这是一个测试对话。"

    def _save_wav(self, audio_data: bytes, filename: str):
        """保存音频数据到 wav 文件"""
        path = os.path.join(os.path.dirname(__file__), "recordings", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2) # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        # print(f"[Debug] Audio saved to {path}")

    def _check_exit_intent(self, text: str) -> bool:
        """检查是否有退出意图"""
        text_lower = text.lower()
        for kw in self.exit_keywords:
            if kw in text_lower:
                return True
        return False

    def _process_response(self, text: str) -> str:
        """
        处理用户输入并生成回复
        """
        if "你好" in text:
            return "你好！很高兴为你服务。"
        if "几点" in text or "时间" in text:
            return f"现在是 {time.strftime('%H:%M')}。"
        return f"我听到了：{text}，但我还不知道怎么回答。"
