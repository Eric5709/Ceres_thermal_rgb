# #!/usr/bin/python3
# # WARNING: SuperGlue is allowed to be used for non-commercial research purposes!!
# #        : You must carefully check and follow its licensing condition!!
# #        : https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE
# from email.mime import image
# import sys
# import cv2
# import math
# import json
# import torch
# import numpy as np
# import argparse
# import matplotlib
# import matplotlib.cm as cm
# from models.matching import Matching
# from models.utils import (make_matching_plot_fast, frame2tensor)

# def main():
#   print('\033[93m' + '****************************************************************************************************' + '\033[0m')
#   print('\033[93m' + '* WARNING: You are going to use SuperGlue that is not allowed to be used for commercial purposes!! *' + '\033[0m')
#   print('\033[93m' + '****************************************************************************************************' + '\033[0m')

#   parser = argparse.ArgumentParser(description='Initial guess estimation based on SuperGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#   parser.add_argument('--data_path', default ='/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo', help='Input data path')
#   parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
#   parser.add_argument('--max_keypoints', type=int, default=-1, help='Maximum number of keypoints detected by Superpoint' ' (\'-1\' keeps all keypoints)')
#   parser.add_argument('--keypoint_threshold', type=float, default=0.05, help='SuperPoint keypoint detector confidence threshold')
#   parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
#   parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
#   parser.add_argument('--match_threshold', type=float, default=0.02, help='SuperGlue match threshold')
#   parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
#   parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
#   parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (CW 0, 90, 180, or 270) (CW)')
#   parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')
#   parser.add_argument('--resize', type=int, nargs='+', default=-1, help='Resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize')
#   opt = parser.parse_args()
#   print(opt)

#   # if len(opt.resize) == 2 and opt.resize[1] == -1:
#   #     opt.resize = opt.resize[0:1]
#   # if len(opt.resize) == 2:
#   #     print('Will resize to {}x{} (WxH)'.format(
#   #         opt.resize[0], opt.resize[1]))
#   # elif len(opt.resize) == 1 and opt.resize[0] > 0:
#   #     print('Will resize max dimension to {}'.format(opt.resize[0]))
#   # elif len(opt.resize) == 1:
#   #     print('Will not resize images')
#   # else:
#   #     raise ValueError('Cannot specify more than two integers for --resize')
    
      
#   torch.set_grad_enabled(False)
#   device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'

#   print('Running inference on device \"{}\"'.format(device))
#   config = {
#     'superpoint': {
#       'nms_radius': opt.nms_radius,
#       'keypoint_threshold': opt.keypoint_threshold,
#       'max_keypoints': opt.max_keypoints
#     },
#     'superglue': {
#       'weights': opt.superglue,
#       'sinkhorn_iterations': opt.sinkhorn_iterations,
#       'match_threshold': opt.match_threshold,
#     }
#   }

#   def angle_to_rot(angle, image_shape):
#     width, height = image_shape[:2]

#     if angle == 90:
#       code = cv2.ROTATE_90_CLOCKWISE
#       func = lambda x: np.stack([x[:, 1], width - x[:, 0]], axis=1)
#     elif angle == 180:
#       code = cv2.ROTATE_180
#       func = lambda x: np.stack([height - x[:, 0], width - x[:, 1]], axis=1)
#     elif angle == 270:
#       code = cv2.ROTATE_90_COUNTERCLOCKWISE
#       func = lambda x: np.stack([height - x[:, 1], x[:, 0]], axis=1)
#     else:
#       print('error: unsupported rotation angle %d' % angle)
#       exit(1)

#     return code, func


#   data_path = opt.data_path
#   with open(data_path + '/calib.json', 'r') as f:
#     calib_config = json.load(f)

#   for iter in range(0, 4):
#     print('processing %s' % iter)

#     matching = Matching(config).eval().to(device)
#     keys = ['keypoints', 'scores', 'descriptors']
    
#     str_iter = str(iter)
#     if len(str_iter) == 1:
#       bag_name = "00000" + str_iter
#     elif len(str_iter) == 2:
#       bag_name = "0000" + str_iter
#     else:
#       bag_name = "000" + str_iter
    
#     camera_image = cv2.imread('%s/RGB/%s.png' % (data_path, bag_name), 0)
#     thermal_image = cv2.imread('%s/Thermal/%s.png' % (data_path, bag_name), 0)

#     if opt.rotate_camera:
#       code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
#       camera_image = cv2.rotate(camera_image, code)
#     if opt.rotate_lidar:
#       code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, thermal_image.shape)
#       thermal_image = cv2.rotate(thermal_image, code)

#     camera_image_tensor = frame2tensor(camera_image, device)
#     thermal_image_tensor = frame2tensor(thermal_image, device)

#     last_data = matching.superpoint({'image': camera_image_tensor})
#     last_data = {k+'0': last_data[k] for k in keys}
#     last_data['image0'] = camera_image_tensor

#     pred = matching({**last_data, 'image1': thermal_image_tensor})
#     kpts0 = last_data['keypoints0'][0].cpu().numpy()
#     kpts1 = pred['keypoints1'][0].cpu().numpy()
#     matches = pred['matches0'][0].cpu().numpy()
#     confidence = pred['matching_scores0'][0].cpu().numpy()
#     valid = matches > -1
#     mkpts0 = kpts0[valid]
#     mkpts1 = kpts1[matches[valid]]
#     color = cm.jet(confidence[valid])
#     kpts0_ = kpts0
#     kpts1_ = kpts1
#     text = [
#         'SuperGlue',
#         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
#         'Matches: {}'.format(len(mkpts0))
#     ]
#     k_thresh = matching.superpoint.config['keypoint_threshold']
#     m_thresh = matching.superglue.config['match_threshold']
#     small_text = [
#         'Keypoint Threshold: {:.4f}'.format(k_thresh),
#         'Match Threshold: {:.2f}'.format(m_thresh)
#     ]
#     if opt.rotate_camera:
#       kpts0_ = camera_R_inv(kpts0_)
#     if opt.rotate_lidar:
#       kpts1_ = lidar_R_inv(kpts1_)

#     result = { 'kpts0': kpts0_.flatten().tolist(), 'kpts1': kpts1_.flatten().tolist(), 'matches': matches.flatten().tolist(), 'confidence': confidence.flatten().tolist() }
#     with open('%s/Output/json/%s_matches.json' % (data_path, bag_name), 'w') as f:
#       json.dump(result, f)

#     # visualization
#     # camera_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
#     H0, W0 = camera_image.shape
#     H1, W1 = thermal_image.shape
#     H, W = max(H0, H1), W0 + W1 + 10
    
#     out = 255 * np.ones((H,W), np.uint8)
#     out[:H0, :W0] = camera_image
#     out[:H1, W0 + 10:] = thermal_image
#     out = np.stack([out]*3, -1)
    
#     kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
#     white = (255, 255, 255)
#     black = (0, 0, 0)
#     for x, y in kpts0:
#         cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
#     for x, y in kpts1:
#         cv2.circle(out, (x + 10 + W0, y), 2, black, -1,
#                     lineType=cv2.LINE_AA)
#         cv2.circle(out, (x + 10 + W0, y), 1, white, -1,
#                     lineType=cv2.LINE_AA)

#     mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
#     color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
#     for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
#         c = c.tolist()
#         cv2.line(out, (x0, y0), (x1 + 10 + W0, y1),
#                  color=c, thickness=1, lineType=cv2.LINE_AA)
#         # display line end-points as circles
#         cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x1 + 10 + W0, y1), 2, c, -1,
#                    lineType=cv2.LINE_AA)

#     sc = min(H / 640., 2.0)

#     Ht = int(30 * sc)  # text height
#     txt_color_fg = (255, 255, 255)
#     txt_color_bg = (0, 0, 0)
#     for i, t in enumerate(text):
#         cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
#                     1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
#         cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
#                     1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

#     # Small text.
#     Ht = int(18 * sc)  # text height
#     for i, t in enumerate(reversed(small_text)):
#         cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
#                     0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
#         cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
#                     0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
        
#     cv2.imshow('Ouput', out)
#     cv2.waitKey(2)
    

#     cv2.imwrite('%s/Output/image/%s.png' % (data_path, bag_name), out)

# if __name__ == '__main__':
#   main()




#!/usr/bin/python3
# WARNING: SuperGlue is allowed to be used for non-commercial research purposes!!
#        : You must carefully check and follow its licensing condition!!
#        : https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE
from email.mime import image
import sys
import cv2
import math
import json
import torch
import numpy as np
import argparse
import matplotlib
import matplotlib.cm as cm
from models.matching import Matching
from models.utils import (make_matching_plot_fast, frame2tensor)

def main():
  print('\033[93m' + '****************************************************************************************************' + '\033[0m')
  print('\033[93m' + '* WARNING: You are going to use SuperGlue that is not allowed to be used for commercial purposes!! *' + '\033[0m')
  print('\033[93m' + '****************************************************************************************************' + '\033[0m')

  parser = argparse.ArgumentParser(description='Initial guess estimation based on SuperGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_path', default ='/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Dataset/sThereo', help='Input data path')
  parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
  parser.add_argument('--max_keypoints', type=int, default=-1, help='Maximum number of keypoints detected by Superpoint' ' (\'-1\' keeps all keypoints)')
  parser.add_argument('--keypoint_threshold', type=float, default=0.05, help='SuperPoint keypoint detector confidence threshold')
  parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
  parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
  parser.add_argument('--match_threshold', type=float, default=0.02, help='SuperGlue match threshold')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
  parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (CW 0, 90, 180, or 270) (CW)')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')
  parser.add_argument('--resize', type=int, nargs='+', default=-1, help='Resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize')
  opt = parser.parse_args()
  print(opt)

  # if len(opt.resize) == 2 and opt.resize[1] == -1:
  #     opt.resize = opt.resize[0:1]
  # if len(opt.resize) == 2:
  #     print('Will resize to {}x{} (WxH)'.format(
  #         opt.resize[0], opt.resize[1]))
  # elif len(opt.resize) == 1 and opt.resize[0] > 0:
  #     print('Will resize max dimension to {}'.format(opt.resize[0]))
  # elif len(opt.resize) == 1:
  #     print('Will not resize images')
  # else:
  #     raise ValueError('Cannot specify more than two integers for --resize')
    
      
  torch.set_grad_enabled(False)
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'

  print('Running inference on device \"{}\"'.format(device))
  config = {
    'superpoint': {
      'nms_radius': opt.nms_radius,
      'keypoint_threshold': opt.keypoint_threshold,
      'max_keypoints': opt.max_keypoints
    },
    'superglue': {
      'weights': opt.superglue,
      'sinkhorn_iterations': opt.sinkhorn_iterations,
      'match_threshold': opt.match_threshold,
    }
  }

  def angle_to_rot(angle, image_shape):
    width, height = image_shape[:2]

    if angle == 90:
      code = cv2.ROTATE_90_CLOCKWISE
      func = lambda x: np.stack([x[:, 1], width - x[:, 0]], axis=1)
    elif angle == 180:
      code = cv2.ROTATE_180
      func = lambda x: np.stack([height - x[:, 0], width - x[:, 1]], axis=1)
    elif angle == 270:
      code = cv2.ROTATE_90_COUNTERCLOCKWISE
      func = lambda x: np.stack([height - x[:, 1], x[:, 0]], axis=1)
    else:
      print('error: unsupported rotation angle %d' % angle)
      exit(1)

    return code, func


  data_path = opt.data_path
  with open(data_path + '/calib.json', 'r') as f:
    calib_config = json.load(f)

  for iter in range(0, 4):
    print('processing %s' % iter)

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    
    str_iter = str(iter)
    if len(str_iter) == 1:
      bag_name = "00000" + str_iter
    elif len(str_iter) == 2:
      bag_name = "0000" + str_iter
    else:
      bag_name = "000" + str_iter
    
    camera_image = cv2.imread('%s/RGB_left/%s.png' % (data_path, bag_name), 0)
    thermal_image = cv2.imread('%s/RGB_right/%s.png' % (data_path, bag_name), 0)

    if opt.rotate_camera:
      code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
      camera_image = cv2.rotate(camera_image, code)
    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, thermal_image.shape)
      thermal_image = cv2.rotate(thermal_image, code)

    camera_image_tensor = frame2tensor(camera_image, device)
    thermal_image_tensor = frame2tensor(thermal_image, device)

    last_data = matching.superpoint({'image': camera_image_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = camera_image_tensor

    pred = matching({**last_data, 'image1': thermal_image_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    kpts0_ = kpts0
    kpts1_ = kpts1
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh)
    ]
    if opt.rotate_camera:
      kpts0_ = camera_R_inv(kpts0_)
    if opt.rotate_lidar:
      kpts1_ = lidar_R_inv(kpts1_)

    result = { 'kpts0': kpts0_.flatten().tolist(), 'kpts1': kpts1_.flatten().tolist(), 'matches': matches.flatten().tolist(), 'confidence': confidence.flatten().tolist() }
    with open('%s/Output/json/%s_matches.json' % (data_path, bag_name), 'w') as f:
      json.dump(result, f)

    # visualization
    # camera_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    H0, W0 = camera_image.shape
    H1, W1 = thermal_image.shape
    H, W = max(H0, H1), W0 + W1 + 10
    
    out = 255 * np.ones((H,W), np.uint8)
    out[:H0, :W0] = camera_image
    out[:H1, W0 + 10:] = thermal_image
    out = np.stack([out]*3, -1)
    
    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    for x, y in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1:
        cv2.circle(out, (x + 10 + W0, y), 2, black, -1,
                    lineType=cv2.LINE_AA)
        cv2.circle(out, (x + 10 + W0, y), 1, white, -1,
                    lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + 10 + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + 10 + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    sc = min(H / 640., 2.0)

    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
        
    cv2.imshow('Ouput', out)
    cv2.waitKey(2)
    

    cv2.imwrite('%s/Output/image/%s.png' % (data_path, bag_name), out)

if __name__ == '__main__':
  main()