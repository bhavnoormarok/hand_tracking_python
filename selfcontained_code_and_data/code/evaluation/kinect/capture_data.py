from utils.freq_imports import *
from utils.helper import create_dir
from utils.perspective_projection import uvd2xyz
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, CpuPacketPipeline, SyncMultiFrameListener, FrameType, Registration, Frame, createConsoleLogger, setGlobalLogger, LoggerLevel
from evaluation.kinect import preprocess
from utils.image import colormap_depth

def setup_kinectv2():
    # Ref: https://github.com/r9y9/pylibfreenect2/blob/master/examples/multiframe_listener.py

    logger = createConsoleLogger(LoggerLevel.Warning)
    setGlobalLogger(logger)

    freenect2 = Freenect2()
    if freenect2.enumerateDevices() == 0:
        print("no device connected")
        exit(1)

    pipeline = OpenGLPacketPipeline()
    serial = freenect2.getDeviceSerialNumber(0)
    device = freenect2.openDevice(serial, pipeline=pipeline)
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()

    # returning freenect2 is required so that the global object is not destructed
    return device, listener, freenect2

def register_color_and_undistort_depth(frames, registration, frame_depth_undistorted, frame_color_registered):
    registration.apply(frames["color"], frames["depth"], frame_depth_undistorted, frame_color_registered)

    color = frames["color"].asarray()   # (1080, 1920, 4)   BGRA uint8
    color_registered = frame_color_registered.asarray(np.uint8) # (424, 512, 4)   BGRA uint8
    depth = frames["depth"].asarray()   # (424, 512) float32 [0, 4500]
    depth_undistorted = frame_depth_undistorted.asarray(np.float32) # (424, 512) float32 [0, 4500]
    ir = frames["ir"].asarray()         # (424, 512) float32 [0, 65535]

    color_raw = color_registered[:, :, :3][:, :, ::-1]  # RGB
    depth_raw = depth_undistorted

    color_raw = np.fliplr(color_raw)
    depth_raw = np.fliplr(depth_raw)

    # TODO: depth has noise around finger boundary, erode or use median filter
    

    return color_raw, depth_raw

def get_kinectv2_intrinsics(device):
    fx = device.getIrCameraParams().fx
    fy = device.getIrCameraParams().fy
    cx = device.getIrCameraParams().cx
    cy = device.getIrCameraParams().cy

    # 366.085, 366.085, 259.229, 207.968
    return fx, fy, cx, cy

def get_kinecvtv2_stored_intrinsics():
    fx = 366.085
    fy = 366.085
    cx = 259.229
    cy = 207.968

    return fx, fy, cx, cy

def main():
    # -------------------------------------------------
    # NOTE: change this for each user and each sequence
    user = "rahul"
    sequence = 8
    # -------------------------------------------------
    
    # sequences:
    # 1: flexion/extension
    # 2: adduction/abduction
    # 3: close/open fist
    # 4: global transform
    # 5: American sign language (Ref: https://www.handspeak.com/learn/f/fingerspell/abcposter.pdf)
    # 6: random gestures

    print(f"User: {user}\nsequence: {sequence}\n")
    print("Press 'y' to confirm, or any other key to exit")
    if input() != "y":
        print("Exiting")
        exit()

    log_dir = f"./log/kinect/{user}/{sequence}"; create_dir(log_dir, True)
    
    out_dir = f"./data/kinect/{user}/{sequence}"; create_dir(out_dir, True)
    out_color_raw_dir = f"{out_dir}/color_raw"; create_dir(out_color_raw_dir, True)
    out_color_proc_dir = f"{out_dir}/color_proc"; create_dir(out_color_proc_dir, True)
    out_depth_raw_dir = f"{out_dir}/depth_raw"; create_dir(out_depth_raw_dir, True)
    out_depth_proc_dir = f"{out_dir}/depth_proc"; create_dir(out_depth_proc_dir, True)
    out_depth_raw_cm_dir = f"{out_dir}/depth_raw_cm"; create_dir(out_depth_raw_cm_dir, True)
    out_depth_proc_cm_dir = f"{out_dir}/depth_proc_cm"; create_dir(out_depth_proc_cm_dir, True)
    
    device, listener, freenect2 = setup_kinectv2()
    registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())
    frame_depth_undistorted = Frame(512, 424, 4)
    frame_color_registered = Frame(512, 424, 4)
    # 366.085, 366.085, 259.229, 207.968
    H, W = 424, 512
    fx, fy, cx, cy = get_kinectv2_intrinsics(device)
    # fx, fy, cx, cy = get_kinecvtv2_stored_intrinsics()

    time_frame_prev = 0
    time_frame_curr = 0
    fps_acc = 0
    fps_cnt = 0
    fps_avg = 0

    z_near, z_far = 0.6, 1.2   # m
    d_near = z_near * 1000; d_far = z_far * 1000    # depth uses mm as units

    xyz_crop_center = np.array([0, 0, 0.7])
    start_capture = False
    i_frame = 0
    while True:
        time_frame_curr = time.time()
        fps = int(1/(time_frame_curr - time_frame_prev))
        fps_acc += fps
        fps_cnt += 1
        fps_avg = int(fps_acc/fps_cnt)
        time_frame_prev = time_frame_curr
        # print(fps_avg)

        # read frame
        frames = listener.waitForNewFrame()
        color_raw, depth_raw = register_color_and_undistort_depth(frames, registration, frame_depth_undistorted, frame_color_registered)
        listener.release(frames)
        
        # crop hand
        color_proc, depth_proc, mask_sil, xyz_crop_center = preprocess.process_frame(color_raw, depth_raw, d_near, d_far, fx, fy, cx, cy, xyz_crop_center)

        # plot
        depth_raw_cm = colormap_depth(depth_raw, d_near, d_far, cv.COLORMAP_INFERNO)
        depth_proc_cm = colormap_depth(depth_proc, d_near, d_far, cv.COLORMAP_INFERNO)
        # img = np.hstack([color_raw, depth_raw_cm, color_proc, depth_proc_cm])
        img_color = np.hstack([color_raw, color_proc])
        img_depth = np.hstack([depth_raw_cm, depth_proc_cm])
        img = np.vstack([img_color, img_depth])
        cv.imshow("color_depth", img[:, :, ::-1])

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            start_capture = True
            print("Capture start")

        if start_capture:
            # save
            cv.imwrite(f"{log_dir}/{i_frame:05d}.png", img[:, :, ::-1])
            cv.imwrite(f"{out_color_raw_dir}/{i_frame:05d}.png", color_raw[:, :, ::-1])
            cv.imwrite(f"{out_color_proc_dir}/{i_frame:05d}.png", color_proc[:, :, ::-1])
            cv.imwrite(f"{out_depth_raw_cm_dir}/{i_frame:05d}.png", depth_raw_cm[:, :, ::-1])
            cv.imwrite(f"{out_depth_proc_cm_dir}/{i_frame:05d}.png", depth_proc_cm[:, :, ::-1])
            np.save(f"{out_depth_raw_dir}/{i_frame:05d}.npy", depth_raw)
            np.save(f"{out_depth_proc_dir}/{i_frame:05d}.npy", depth_proc)

            i_frame += 1
    device.stop()
    device.close()
    cv.destroyAllWindows()

    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_color_raw_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_color_proc_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_depth_raw_cm_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_depth_proc_cm_dir}"])
    subprocess.run(["./code/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{log_dir}"])
    

if __name__ == "__main__":
    main()