import dlib # 사람의 얼굴을 인식하기 위한 랜드마크를 찾아내는 라이브러리
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector() # 얼굴 앞면을 찾는 detector
shape = dlib.shape_predictor('./models/BeautyGAN/shape_predictor_68_face_landmarks.dat') # 숫자는 랜드마크(특징점) 갯수

# 이미지 확인 코드는 주석 처리했다##############################
# img = dlib.load_rgb_image('./imgs/08.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy()
# dets = detector(img, 1) # 얼굴을 찾으면 dets에 return
#
# if len(dets): # 만약 찾은 얼굴이 여러 가지라면 dets의 길이가 2 이상일 수 있다
#     fig, ax = plt.subplots(1, figsize=(10, 16))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height() # 얼굴의 좌표, 너비, 높이
#         rect = patches.Rectangle((x, y), w, h, linewidth=2,
#                                  edgecolor='r', facecolor='None') # 얼굴 영역은 지정한 서식의 직사각형으로 표시
#         ax.add_patch(rect) # 직사각형 추가
#     ax.imshow(img_result)
#     plt.show()
# else:
#     print('Not found face')
#
#
# # 사진에 얼굴이 여러 개 있을 경우
# # img = dlib.load_rgb_image('./imgs/02.jpg')
# # img_result = img.copy()
# # dets = detector(img, 1) # 얼굴을 찾으면 dets에 return
# #
# # if len(dets): # 만약 찾은 얼굴이 여러 가지라면 dets의 길이가 2 이상일 수 있다
# #     fig, ax = plt.subplots(1, figsize=(10, 16))
# #     for det in dets:
# #         x, y, w, h = det.left(), det.top(), det.width(), det.height() # 얼굴의 좌표, 너비, 높이
# #         rect = patches.Rectangle((x, y), w, h, linewidth=2,
# #                                  edgecolor='r', facecolor='None') # 얼굴 영역은 지정한 서식의 직사각형으로 표시
# #         ax.add_patch(rect)
# #     ax.imshow(img_result)
# #     plt.show()
# # else:
# #     print('Not found face')
#
# # 이미지 다시 불러 오기 (파일을 안 바꿀 거면 안 해도 됨) ========
# img = dlib.load_rgb_image('./imgs/04.jpg')
# img_result = img.copy()
# dets = detector(img, 1)
# # ============================================================
#
# fig, ax = plt.subplots(1, figsize=(16, 10))
# obj = dlib.full_object_detections()
#
# # 모델을 변경하고 싶으면 ============================
# shape = dlib.shape_predictor('./models/BeautyGAN/shape_predictor_68_face_landmarks.dat')
# # =================================================
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3,
#                                 edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#     ax.imshow(img)
# plt.show()
# ###########################################################

# 얼굴을 정렬(틀어진 얼굴을 올바른 각도로 전환하기 위해 이미지 자체를 회전시킨다든가) ######################

def align_faces(img): # 얼굴 세워서 저장하기
    dets = detector(img)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection) # 각 얼굴의 랜드마크 추출
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.5) # 얼굴을 256* 256 크기로 크롭하고 정렬, 얼굴 주변에 padding을 추가해서 더 자연스럽게 크롭
    return faces

# test_img = dlib.load_rgb_image('./imgs/01.jpg')
# test_faces = align_faces(test_img)
# fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(10, 8))
# axes[0].imshow(test_img)
# for i, face in enumerate(test_faces):
#     axes[i + 1].imshow(face)
# plt.show()
# #########################################################################################################

# 화장시키기
# 모델 불러 오기
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.import_meta_graph('./models/BeautyGAN/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/BeautyGAN'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0') # 소스 이미지 입력
Y = graph.get_tensor_by_name('Y:0') # 레퍼런스 이미지
Xs = graph.get_tensor_by_name('generator/Xs:0') # 생성된 결과 출력

def preprocess(img):
    return img / 127.5 - 1 # -1 < img < 1 / [0,255] → [-1,1] 범위로 정규화
def deprocess(img):
    return (img + 1) / 2 # 0 < img < 1 / [-1,1] → [0,1] 범위로 역정규화

# source 이미지
img1 = dlib.load_rgb_image('./imgs/no_makeup/vSYYZ306.png')
img1_faces = align_faces(img1)

# reference 이미지
img2 = dlib.load_rgb_image('./imgs/makeup/XMY-014.png')
img2_faces = align_faces(img2)

# 이미지 표시하기
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

src_img = img1_faces[0]
ref_img = img2_faces[0]

# 이미지 전처리 - shape 맞추기
X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0) # 배치 차원 추가

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0) # 배치 차원 추가

# BeautyGAN 실행
output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
axes[2].imshow(output_img)
plt.show()