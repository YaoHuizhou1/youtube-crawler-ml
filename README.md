# YouTube 双人对话视频抓取系统 - ML 服务

Python 实现的机器学习分析服务，提供人脸检测和说话人分离功能。

## 技术栈

- **语言**: Python 3.11
- **RPC 框架**: gRPC
- **人脸检测**: MediaPipe
- **说话人分离**: pyannote.audio
- **视频处理**: OpenCV, FFmpeg

## 项目结构

```
youtube-crawler-ml/
├── src/
│   ├── main.py              # gRPC 服务入口
│   ├── face_detector.py     # 人脸检测模块
│   ├── speaker_diarizer.py  # 说话人分离模块
│   └── video_processor.py   # 视频处理工具
├── proto/
│   └── ml_service.proto     # gRPC 服务定义
├── requirements.txt         # Python 依赖
└── Dockerfile
```

## 环境要求

- Python 3.11+
- FFmpeg
- CUDA (可选，用于 GPU 加速)

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows
# 下载并添加到 PATH
```

### 4. 配置 HuggingFace Token (可选)

说话人分离需要 HuggingFace 访问令牌：

```bash
export HF_TOKEN=your_huggingface_token
```

### 5. 生成 gRPC 代码

```bash
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src \
    --grpc_python_out=./src \
    ./proto/ml_service.proto
```

### 6. 启动服务

```bash
python src/main.py
```

服务将在 `localhost:50051` 启动。

## gRPC 接口

### 服务定义

```protobuf
service MLService {
  // 人脸检测
  rpc DetectFaces(FaceDetectionRequest) returns (FaceDetectionResponse);

  // 说话人分离
  rpc AnalyzeSpeakers(SpeakerAnalysisRequest) returns (SpeakerAnalysisResponse);
}
```

### 人脸检测

**请求**:
```protobuf
message FaceDetectionRequest {
  string video_path = 1;   // 视频文件路径
  int32 sample_rate = 2;   // 采样间隔（秒）
}
```

**响应**:
```protobuf
message FaceDetectionResponse {
  double avg_face_count = 1;   // 平均人脸数
  int32 max_face_count = 2;    // 最大人脸数
  int32 min_face_count = 3;    // 最小人脸数
  repeated FrameFaceResult frame_results = 4;  // 每帧结果
}
```

### 说话人分离

**请求**:
```protobuf
message SpeakerAnalysisRequest {
  string video_path = 1;   // 视频文件路径
}
```

**响应**:
```protobuf
message SpeakerAnalysisResponse {
  int32 speaker_count = 1;     // 说话人数量
  double dialogue_ratio = 2;   // 对话比例
  repeated SpeakerSegment segments = 3;  // 时间段列表
}
```

## 模块说明

### FaceDetector (人脸检测)

使用 Google MediaPipe 进行人脸检测：

```python
class FaceDetector:
    def analyze(self, video_path: str, sample_rate: int = 5) -> dict:
        """
        分析视频中的人脸数量

        Args:
            video_path: 视频文件路径
            sample_rate: 每隔多少秒采样一帧

        Returns:
            {
                "avg_face_count": 2.1,
                "max_face_count": 3,
                "min_face_count": 1,
                "frame_results": [...]
            }
        """
```

**工作原理**:
1. 使用 OpenCV 读取视频
2. 按采样率提取关键帧
3. 使用 MediaPipe Face Detection 检测每帧中的人脸
4. 统计平均、最大、最小人脸数

### SpeakerDiarizer (说话人分离)

使用 pyannote.audio 进行说话人分离：

```python
class SpeakerDiarizer:
    def analyze(self, video_path: str) -> dict:
        """
        分析视频中的说话人

        Args:
            video_path: 视频文件路径

        Returns:
            {
                "speaker_count": 2,
                "dialogue_ratio": 0.85,
                "segments": [
                    {"start_ms": 0, "end_ms": 30000, "speaker_id": 0},
                    {"start_ms": 30000, "end_ms": 60000, "speaker_id": 1},
                    ...
                ]
            }
        """
```

**工作原理**:
1. 使用 FFmpeg 从视频提取音频 (16kHz WAV)
2. 使用 pyannote.audio 进行说话人分离
3. 统计说话人数量和对话比例
4. 返回时间段标注

### VideoProcessor (视频处理工具)

提供视频处理的通用工具函数：

```python
class VideoProcessor:
    @staticmethod
    def extract_frames(video_path, timestamps_ms) -> List[Tuple[int, ndarray]]

    @staticmethod
    def get_video_info(video_path) -> dict

    @staticmethod
    def extract_audio(video_path, output_path=None) -> str

    @staticmethod
    def create_thumbnail(video_path, timestamp_ms=0) -> str
```

## GPU 加速

### 检查 CUDA 是否可用

```python
import torch
print(torch.cuda.is_available())
```

### 配置 GPU

pyannote.audio 会自动检测并使用可用的 GPU：

```python
if torch.cuda.is_available():
    self.pipeline.to(torch.device("cuda"))
```

## Docker 部署

### 构建镜像

```bash
docker build -t youtube-crawler-ml .
```

### 运行容器 (CPU)

```bash
docker run -p 50051:50051 youtube-crawler-ml
```

### 运行容器 (GPU)

```bash
docker run --gpus all -p 50051:50051 youtube-crawler-ml
```

### Dockerfile 说明

```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 生成 gRPC 代码
RUN python -m grpc_tools.protoc ...

CMD ["python", "src/main.py"]
```

## 依赖说明

| 依赖 | 版本 | 用途 |
|-----|------|------|
| grpcio | 1.62.1 | gRPC 运行时 |
| grpcio-tools | 1.62.1 | gRPC 代码生成 |
| mediapipe | 0.10.11 | 人脸检测 |
| pyannote.audio | 3.1.1 | 说话人分离 |
| torch | >=2.0.0 | 深度学习框架 |
| opencv-python | 4.9.0.80 | 视频处理 |
| ffmpeg-python | 0.2.0 | FFmpeg 封装 |

## 性能优化

### 1. 采样率调优

对于人脸检测，默认每 5 秒采样一帧。可根据需要调整：

```python
# 更快但精度较低
result = detector.analyze(video_path, sample_rate=10)

# 更精确但更慢
result = detector.analyze(video_path, sample_rate=2)
```

### 2. 模型选择

MediaPipe 提供两种人脸检测模型：

```python
# model_selection=0: 近距离模型（2米内）
# model_selection=1: 全距离模型（5米内，默认）
self.face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
```

### 3. 批处理

对于大量视频，考虑使用批处理：

```python
# 并行处理多个视频
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(detector.analyze, video_paths)
```

## 故障排除

### FFmpeg 未找到

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

确保 FFmpeg 已安装并在 PATH 中。

### HuggingFace Token 问题

```
OSError: You are not authenticated to download this model
```

设置 `HF_TOKEN` 环境变量或使用 `huggingface-cli login`。

### CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

减小批处理大小或使用 CPU：

```python
self.pipeline.to(torch.device("cpu"))
```

## 相关项目

- [youtube-crawler-backend](../youtube-crawler-backend) - Go 后端服务
- [youtube-crawler-frontend](../youtube-crawler-frontend) - 前端 React 应用

## License

MIT
