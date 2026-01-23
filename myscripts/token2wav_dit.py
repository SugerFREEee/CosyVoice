# CosyVoice2 Token2Wav
# This script implements a non-streaming token-to-waveform conversion using CosyVoice2 models.
# It converts generated speech tokens to audio waveforms using the cosyvoice library.
""" Example Usage
    /home/wjs/workspace/miniconda3/envs/cosyvoice/bin/python myscripts/token2wav_dit.py
"""
import torch
import sys
import os
# 添加项目根目录到Python搜索路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# 添加Matcha-TTS到Python搜索路径
sys.path.append('/home/wjs/workspace/CosyVoice/third_party/Matcha-TTS')


from torch.utils.data import DataLoader, Dataset
import torchaudio
import argparse
import time
from tqdm import tqdm
from datetime import datetime, timedelta


class CosyVoice2_Token2Wav(torch.nn.Module):
    """拆分模块的 Token2Wav 模型，参考 runtime/triton_trtllm/token2wav_dit.py 的结构"""
    def __init__(self, model_dir: str, device_id: int = 0, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.dtype = dtype

        # 加载 cosyvoice 各个模块
        from cosyvoice.cli.cosyvoice import AutoModel
        cosyvoice = AutoModel(model_dir=model_dir)

        # 拆分各个模块
        self.frontend = cosyvoice.frontend
        self.flow = cosyvoice.model.flow
        self.hift = cosyvoice.model.hift
        self.token_mel_ratio = cosyvoice.model.flow.token_mel_ratio

    def extract_prompt_features(self, prompt_wav_path: str):
        """提取 prompt 音频的所有特征

        Returns:
            prompt_token: prompt 音频的 speech token
            prompt_feat: prompt 音频的 mel 特征
            embedding: 说话人嵌入向量
        """
        prompt_token, _ = self.frontend._extract_speech_token(prompt_wav_path)
        prompt_feat, _ = self.frontend._extract_speech_feat(prompt_wav_path)
        embedding = self.frontend._extract_spk_embedding(prompt_wav_path)

        return prompt_token, prompt_feat, embedding

    def forward_flow(self, token: torch.Tensor, prompt_token: torch.Tensor,
                     prompt_feat: torch.Tensor, embedding: torch.Tensor):
        """Flow 模型推理，生成 mel 频谱

        Args:
            token: 生成的 speech token [B, T]
            prompt_token: prompt 音频的 token [B, T']
            prompt_feat: prompt 音频的 mel 特征 [B, T'', D]
            embedding: 说话人嵌入 [B, D]

        Returns:
            tts_mel: 生成的 mel 频谱 [B, D, T]
        """
        tts_mel, _ = self.flow.inference(
            token=token.to(self.device, dtype=torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device),
            streaming=False,
            finalize=True
        )
        return tts_mel

    def forward_hift(self, tts_mel: torch.Tensor):
        """HiFT 声码器推理，将 mel 频谱转换为音频波形

        Args:
            tts_mel: mel 频谱 [B, D, T]

        Returns:
            tts_speech: 生成的音频波形 [B, T]
        """
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech

    @torch.inference_mode()
    def forward(
        self, generated_speech_tokens_list: list[list[int]], prompt_audios_list: list[str], prompt_audios_sample_rate: list[int]
    ):
        assert all(sample_rate == 16000 for sample_rate in prompt_audios_sample_rate)
        generated_wavs = []

        for generated_speech_tokens, prompt_wav_path in zip(generated_speech_tokens_list, prompt_audios_list):
            # 转换为 tensor 并确保 batch 维度
            token = torch.tensor([generated_speech_tokens], dtype=torch.int32)

            # 步骤1: 提取 prompt 音频特征
            prompt_token, prompt_feat, embedding = self.extract_prompt_features(prompt_wav_path)

            # 步骤2: Flow 模型推理生成 mel 频谱
            tts_mel = self.forward_flow(token, prompt_token, prompt_feat, embedding)

            # 步骤3: HiFT 声码器推理生成音频波形
            tts_speech = self.forward_hift(tts_mel)

            generated_wavs.append(tts_speech)

        return generated_wavs


def load_tokens_from_file(token_file):
    """从文件加载预保存的 tokens

    Args:
        token_file: token 文件路径

    Returns:
        list[int]: token 列表
    """
    with open(token_file, 'r') as f:
        tokens_str = f.read().strip()

    # 解析 tokens (空格分隔的整数)
    tokens = [int(t) for t in tokens_str.split()]

    return tokens


def load_prompt_audio(prompt_wav_path, target_sample_rate=16000):
    """返回prompt音频文件路径（不实际加载音频，由frontend内部处理）

    Args:
        prompt_wav_path: prompt音频文件路径
        target_sample_rate: 目标采样率（这里仅作为兼容性参数）

    Returns:
        str: 音频文件路径
        int: 采样率（固定为16000，与frontend期望一致）
    """
    return prompt_wav_path, 16000


class LocalToken2WavDataset(Dataset):
    """本地文件的Token2Wav数据集类"""
    def __init__(self, input_lst, tokens_dir, sample_rate=16000):
        self.tokens_dir = tokens_dir
        self.sample_rate = sample_rate
        self.data_items = []
        
        # 读取并解析输入列表
        with open(input_lst, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析不同格式
            parts = line.split('|')
            
            if len(parts) == 5:
                utt, prompt_text, prompt_wav, infer_text, infer_wav = parts
            elif len(parts) == 4:
                utt, prompt_text, prompt_wav, infer_text = parts
            elif len(parts) == 3:
                utt, infer_text, prompt_wav = parts
                prompt_text = ""
            elif len(parts) == 2:
                utt, infer_text = parts
                prompt_text = ""
                prompt_wav = ""
            else:
                print(f"Warning: Skipping line with unexpected format: {line}")
                continue
            
            # 处理 utt（去掉 .wav 后缀）
            if utt.endswith('.wav'):
                utt = utt[:-4]
            
            # 检查 token 文件是否存在
            token_file = os.path.join(tokens_dir, f"{utt}.txt")
            if not os.path.exists(token_file):
                print(f"Warning: Token file not found for {utt}: {token_file}")
                continue
            
            # 处理 prompt_wav 路径（将相对路径转换为绝对路径）
            if prompt_wav and not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(input_lst), prompt_wav)
            
            # 检查 prompt_wav 是否存在
            if not prompt_wav or not os.path.exists(prompt_wav):
                print(f"Warning: Prompt audio not found for {utt}: {prompt_wav}")
                continue
            
            self.data_items.append({
                'utt': utt,
                'prompt_wav': prompt_wav,
                'token_file': token_file
            })
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # 加载 tokens
        target_audio_cosy2_tokens = load_tokens_from_file(item['token_file'])
        
        # 加载 prompt 音频（现在返回的是路径字符串）
        prompt_audio_path, sampling_rate = load_prompt_audio(item['prompt_wav'], self.sample_rate)
        
        return {
            'id': item['utt'],
            'target_audio_cosy2_tokens': target_audio_cosy2_tokens,
            'prompt_audio': {'path': prompt_audio_path, 'sampling_rate': sampling_rate}
        }


def collate_fn(batch):
    ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = [], [], [], []
    for item in batch:
        generated_speech_tokens_list.append(item['target_audio_cosy2_tokens'])
        prompt_audios_list.append(item['prompt_audio']['path'])
        prompt_audios_sample_rate.append(item['prompt_audio']['sampling_rate'])
        ids.append(item['id'])

    return ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="/home/wjs/workspace/data/new_seedtts_output_tokens")
    parser.add_argument("--input-lst", type=str, default="/home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst", help="Input meta list file path (format: utt|prompt_text|prompt_wav|infer_text)")
    parser.add_argument("--tokens-dir", type=str, default="/home/wjs/workspace/data/seedtts_tokens", help="Directory containing pre-saved token files")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs, performance statistics will only be collected from the last epoch")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for input prompt audio (default: 22050)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("="*80)
    print("CosyVoice2 Token2Wav Inference")
    print("="*80)
    print(f"Model directory:     {args.model_dir}")
    print(f"Input list:          {args.input_lst}")
    print(f"Tokens directory:    {args.tokens_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Warmup epochs:       {args.warmup}")
    print("="*80)

    # 加载模型
    print("\nLoading model...")
    model = CosyVoice2_Token2Wav(model_dir=args.model_dir)
    print("Model loaded successfully!")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 创建本地数据集实例
    print("\nLoading dataset...")
    dataset = LocalToken2WavDataset(
        input_lst=args.input_lst,
        tokens_dir=args.tokens_dir,
        sample_rate=args.sample_rate
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"\nStarting inference with {args.warmup} warmup epoch(s)...")
    print("="*80)

    for epoch in range(args.warmup):
        print(f"\nEpoch {epoch + 1}/{args.warmup}")
        start_time = time.time()
        start_datetime = datetime.now()

        success_count = 0
        error_count = 0

        # 使用 tqdm 显示进度条
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}", unit="batch"):
            try:
                ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = batch

                generated_wavs = model(generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate)

                # for id, wav in zip(ids, generated_wavs):
                #     torchaudio.save(f"{args.output_dir}/{id}.wav", wav.cpu(), 24000)
                #     success_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError processing batch: {str(e)}")
                continue

        end_time = time.time()
        end_datetime = datetime.now()
        epoch_time = end_time - start_time

        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Duration:          {timedelta(seconds=int(epoch_time))}")
        print(f"  Successful:        {success_count}/{success_count + error_count}")
        if success_count > 0:
            print(f"  Avg time/sample:   {epoch_time/success_count:.3f}s")
            print(f"  Throughput:        {success_count/epoch_time:.2f} samples/s")

    print("\n" + "="*80)
    print("INFERENCE COMPLETED!")
    print("="*80)
    print(f"Output directory:    {args.output_dir}")
    print(f"Total samples:       {len(dataset)}")
    print("="*80)
