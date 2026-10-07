import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import numpy as np
import time

@st.cache_resource
# 検出結果を元画像に重ねて描画する処理
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image) # 元画像のコピーを作成

  # 検出した顔一つひとつに対して描画処理を実行
  for idx in range(len(face_landmarks_list)): #len(face_landmarks_list) = 検出した人数を表す。
    face_landmarks = face_landmarks_list[idx]

    # 検出結果の描画

    # 顔全体に対して網目状のワイヤーフレームを描画(tesselation)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())

    # 目、眉、口、顔の輪郭を描画(contours)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

    # 左目の虹彩を四角形で囲む(left iris)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    # 右目の虹彩を四角形で囲む(right iris)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='./models/face_landmarker_v2_with_blendshapes.task') # Github上のモデルパスを指定
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)




class VideoProcessor:
    
    def __init__(self):
        self.start = time.time()
        self.is_warning = False
    
    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の読み込み
        # NumPy配列からMediaPipeのImageオブジェクトを作成する
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
        # AIによる推定の実行
        detection_result = detector.detect(image)

        # 推定結果を描画
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        # 左目の判定処理
        # 上端のy座標と、下端のy座標の差を計算
        # 画像内の下に位置するほど、y座標が大きいことに注意しましょう。
        left_y_th = detection_result.face_landmarks[0][374].y - detection_result.face_landmarks[0][386].y

        # 右端のx座標と、左端のx座標の差を計算
        # 画像内の左に位置するほど、x座標が大きいことに注意しましょう。
        left_x_th = detection_result.face_landmarks[0][263].x - detection_result.face_landmarks[0][362].x

        # 比率を計算
        left_ratio = left_y_th / left_x_th

        # 右目の判定処理
        # 上端のy座標と、下端のy座標の差を計算
        # 画像内の下に位置するほど、y座標が大きいことに注意しましょう。
        right_y_th = detection_result.face_landmarks[0][145].y - detection_result.face_landmarks[0][159].y

        # 右端のx座標と、左端のx座標の差を計算
        # 画像内の左に位置するほど、x座標が大きいことに注意しましょう。
        right_x_th = detection_result.face_landmarks[0][133].x - detection_result.face_landmarks[0][33].x

        # 比率を計算
        right_ratio = right_y_th / right_x_th

        now = time.time()
        open_eye_time = now - self.start # 目が開いている時間を計測
                    
        if open_eye_time > 6:
            text = "WARNING BLINK!"
            #cv2.putText(画像データ, 文字, 文字の位置, フォントの種類, 文字の倍率, 文字の色, 文字の太さ)
            cv2.putText(img, text, (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 4)

        if right_ratio < 0.5 and left_ratio < 0.5:
            self.start = time.time() # タイマーをリセット

        return av.VideoFrame.from_ndarray(img, format="bgr24")



def run_blink_detection():
    webrtc_streamer(
      key="example", #画面上のカメラ画面を一意に識別し、その接続や内部状態を保持するための固有ID
      video_processor_factory=VideoProcessor,
      media_stream_constraints={
            "video": {
                "width": {"ideal": 640}, # 画面の横幅
                "height": {"ideal": 360}, # 画面の高さ
                "frameRate": {"ideal": 30} # フレームレート
            },
            "audio": False
      },
      async_processing=True #非同期処理(リアルタイム性を確保するための設定)
    )