import logging
from concurrent import futures
import grpc
import ml_service_pb2
import ml_service_pb2_grpc
from face_detector import FaceDetector
from speaker_diarizer import SpeakerDiarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.face_detector = FaceDetector()
        self.speaker_diarizer = SpeakerDiarizer()
        logger.info("ML services initialized")

    def DetectFaces(self, request, context):
        logger.info(f"DetectFaces called for: {request.video_path}")

        try:
            result = self.face_detector.analyze(
                request.video_path,
                sample_rate=request.sample_rate or 5
            )

            frame_results = [
                ml_service_pb2.FrameFaceResult(
                    timestamp_ms=fr["timestamp_ms"],
                    face_count=fr["face_count"]
                )
                for fr in result["frame_results"]
            ]

            return ml_service_pb2.FaceDetectionResponse(
                avg_face_count=result["avg_face_count"],
                max_face_count=result["max_face_count"],
                min_face_count=result["min_face_count"],
                frame_results=frame_results
            )
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.FaceDetectionResponse()

    def AnalyzeSpeakers(self, request, context):
        logger.info(f"AnalyzeSpeakers called for: {request.video_path}")

        try:
            result = self.speaker_diarizer.analyze(request.video_path)

            segments = [
                ml_service_pb2.SpeakerSegment(
                    start_ms=seg["start_ms"],
                    end_ms=seg["end_ms"],
                    speaker_id=seg["speaker_id"]
                )
                for seg in result["segments"]
            ]

            return ml_service_pb2.SpeakerAnalysisResponse(
                speaker_count=result["speaker_count"],
                dialogue_ratio=result["dialogue_ratio"],
                segments=segments
            )
        except Exception as e:
            logger.error(f"Speaker analysis error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.SpeakerAnalysisResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("ML Service started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
