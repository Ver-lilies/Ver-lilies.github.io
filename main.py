import cv2
import numpy as np
import sys

# 新增：定义生成三种扭曲映射表的函数
def init_distortion_maps(width, height):
    # 构建坐标网格
    map_y, map_x = np.indices((height, width), dtype=np.float32)
    
    # 1. 波浪扭曲 (Wave)
    # 通过对 y 坐标应用正弦函数来使 x 坐标产生波动
    map_x_wave = map_x + 20 * np.sin(map_y / 20.0)
    map_y_wave = map_y.copy()
    
    # 构建极坐标用于透镜特效
    cx, cy = width / 2.0, height / 2.0
    x = map_x - cx
    y = map_y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    R_max = np.sqrt(cx**2 + cy**2) + 1  # 避免除以零
    
    # 2. 凸透镜效果/凹陷 (Bulge)
    # 使中心区域膨胀扭曲
    r_bulge = r * (r / R_max)
    map_x_bulge = cx + r_bulge * np.cos(theta)
    map_y_bulge = cy + r_bulge * np.sin(theta)
    
    # 3. 凹透镜效果/缩小 (Pinch)
    # 使中心区域收缩扭曲
    r_pinch = np.sqrt(r * R_max)
    map_x_pinch = cx + r_pinch * np.cos(theta)
    map_y_pinch = cy + r_pinch * np.sin(theta)
    
    return (
        (map_x_wave.astype(np.float32), map_y_wave.astype(np.float32)),
        (map_x_bulge.astype(np.float32), map_y_bulge.astype(np.float32)),
        (map_x_pinch.astype(np.float32), map_y_pinch.astype(np.float32))
    )

def main():
    print("正在尝试打开摄像头...")
    # 0通常是默认的自带摄像头，如果是外接USB摄像头可能为1或2
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查权限或摄像头连接。")
        sys.exit(1)
        
    print("================== 图像扭曲菜单 ==================")
    print(" 按 '0' 键: 恢复正常画面")
    print(" 按 '1' 键: 波浪扭曲效果 (Wave)")
    print(" 按 '2' 键: 凸透镜效果   (Bulge)")
    print(" 按 '3' 键: 凹透镜效果   (Pinch)")
    print(" 按 'q' 键: 退出程序")
    print("==================================================")
    
    # 读取第一帧以获取图像分辨率，用于初始化扭曲映射表
    ret, frame = cap.read()
    if not ret:
        print("无法获取图像，退出...")
        cap.release()
        return
        
    height, width = frame.shape[:2]
    maps = init_distortion_maps(width, height)
    
    mode = 0  # 初始模式：正常画面
    mode_names = ["0: Normal", "1: Wave Distortion", "2: Bulge Distortion", "3: Pinch Distortion"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像，退出...")
            break
            
        # 根据当前模式采用对应的扭曲算法处理采集到的图像
        if mode == 0:
            distorted_frame = frame
        elif mode >= 1 and mode <= 3:
            map_x, map_y = maps[mode - 1]
            # 使用 cv2.remap 进行高效的图像坐标重映射（图像扭曲处理）
            distorted_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        # 在展示图像上方绘制当前所用算法和模式的提示文本
        cv2.putText(distorted_frame, mode_names[mode], (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # 展示扭曲后的图像
        cv2.imshow('Camera Feed - Press "q" to quit', distorted_frame)
        
        # 等待键盘事件交互，构建用户操作菜单响应
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            mode = 0
            print(">> 已切换到: 正常画面")
        elif key == ord('1'):
            mode = 1
            print(">> 已切换到: 波浪扭曲")
        elif key == ord('2'):
            mode = 2
            print(">> 已切换到: 凸透镜特效")
        elif key == ord('3'):
            mode = 3
            print(">> 已切换到: 凹透镜特效")
            
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
