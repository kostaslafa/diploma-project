import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import PIL.Image
import PIL.ImageTk
import cv2
import time
from operator import itemgetter
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import numpy as np


def edgefilter(image, p=1, iters=1):
    # Edge preserving filter
    img = image.astype(np.float32)
    (h, w) = img.shape[:2]
    for i in range(iters):
        padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        tmp0 = abs(img - padded[1:-1, :-2, :])
        tmp1 = abs(img - padded[1:-1, 2:, :])
        tmp2 = abs(img - padded[2:, 1:-1, :])
        tmp3 = abs(img - padded[:-2, 1:-1, :])
        tmp4 = abs(img - padded[:-2, :-2, :])
        tmp5 = abs(img - padded[2:, :-2, :])
        tmp6 = abs(img - padded[2:, 2:, :])
        tmp7 = abs(img - padded[:-2, 2:, :])
        d0 = (tmp0[:, :, 0] + tmp0[:, :, 1] + tmp0[:, :, 2]) / 765
        c0 = (1 - d0) ** p
        d1 = (tmp1[:, :, 0] + tmp1[:, :, 1] + tmp1[:, :, 2]) / 765
        c1 = (1 - d1) ** p
        d2 = (tmp2[:, :, 0] + tmp2[:, :, 1] + tmp2[:, :, 2]) / 765
        c2 = (1 - d2) ** p
        d3 = (tmp3[:, :, 0] + tmp3[:, :, 1] + tmp3[:, :, 2]) / 765
        c3 = (1 - d3) ** p
        d4 = (tmp4[:, :, 0] + tmp4[:, :, 1] + tmp4[:, :, 2]) / 765
        c4 = (1 - d4) ** p
        d5 = (tmp5[:, :, 0] + tmp5[:, :, 1] + tmp5[:, :, 2]) / 765
        c5 = (1 - d5) ** p
        d6 = (tmp6[:, :, 0] + tmp6[:, :, 1] + tmp6[:, :, 2]) / 765
        c6 = (1 - d6) ** p
        d7 = (tmp7[:, :, 0] + tmp7[:, :, 1] + tmp7[:, :, 2]) / 765
        c7 = (1 - d7) ** p
        sum_c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
        j = 0
        while j < 3:
            img[:, :, j] = c0 * padded[1:-1, :-2, j]
            img[:, :, j] = img[:, :, j] + c1 * padded[1:-1, 2:, j]
            img[:, :, j] = img[:, :, j] + c2 * padded[2:, 1:-1, j]
            img[:, :, j] = img[:, :, j] + c3 * padded[:-2, 1:-1, j]
            img[:, :, j] = img[:, :, j] + c4 * padded[:-2, :-2, j]
            img[:, :, j] = img[:, :, j] + c5 * padded[2:, :-2, j]
            img[:, :, j] = img[:, :, j] + c6 * padded[2:, 2:, j]
            img[:, :, j] = img[:, :, j] + c7 * padded[:-2, 2:, j]
            img[:, :, j] = img[:, :, j] / sum_c
            j += 1

    img = img.astype(np.uint8)
    return img
    # End of edge preserving filter


def reducecolors(img, n_colors=32):
    (h, w) = img.shape[:2]
    # Sobel in order to use in Sample
    k_size = 3
    # sobel to blue channel
    sobelxb = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 1, 0, ksize=k_size)
    sobelyb = cv2.Sobel(img[:, :, 0], cv2.CV_64F, 0, 1, ksize=k_size)
    sb64 = (sobelxb ** 2 + sobelyb ** 2) ** (0.5)
    abs_sb = np.absolute(sb64)
    sb = np.uint8(abs_sb)
    #sobel to green channel
    sobelxg = cv2.Sobel(img[:, :, 1], cv2.CV_64F, 1, 0, ksize=k_size)
    sobelyg = cv2.Sobel(img[:, :, 1], cv2.CV_64F, 0, 1, ksize=k_size)
    sg64 = (sobelxg ** 2 + sobelyg ** 2) ** (0.5)
    abs_sg = np.absolute(sg64)
    sg = np.uint8(abs_sg)
    # sobel to red channel
    sobelxr = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=k_size)
    sobelyr = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=k_size)
    sr64 = (sobelxr ** 2 + sobelyr ** 2) ** (0.5)
    abs_sr = np.absolute(sr64)
    sr = np.uint8(abs_sr)

    # find max
    tmp = np.maximum(sb, sg)
    g = np.maximum(tmp, sr)

    # choose pixels-sampling
    inv = (255 - g).astype(np.float64)
    invp = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    ninv = inv * invp[1:-1, :-2] * invp[1:-1, 2:] * invp[2:, 1:-1] * invp[:-2, 1:-1] * invp[:-2, :-2] * invp[2:,
                                                                                                        :-2] * invp[
                                                                                                               2:,
                                                                                                               2:] * invp[
                                                                                                                     :-2,
                                                                                                                     2:]
    invf = ninv.reshape((h * w))
    pos = np.asarray(np.nonzero(invf))[0]
    pindex = np.random.randint(0, pos.shape[0], size=1000)

    # KMean
    img = np.array(img, dtype=np.float64) / 255
    image_array = np.reshape(img, (w * h, 3))

    image_array_sample = image_array[pos[pindex]][:]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    labels = kmeans.predict(image_array)

    quant = kmeans.cluster_centers_[labels]

    # MeanShift
    bandwidth = estimate_bandwidth(quant[pos[pindex]])
    ms = MeanShift(bandwidth=bandwidth / 2, bin_seeding=True).fit(quant[pos[pindex]])

    labels = ms.predict(quant)

    final = ms.cluster_centers_[labels]
    final = final.reshape((h, w, 3))

    # return final_image as unsigned int8, centers
    return (final * 255).astype(np.uint8), (255 * ms.cluster_centers_).astype(np.uint8)


def make_mask(h, w):
    h = int(h)
    w = int(w)
    mask = np.zeros((h, w), dtype=np.uint8)

    x1 = int(0.4 * w)
    x2 = int(0.6 * w)

    pts = np.array([[[0, h - 1], [x1, 0], [x2, 0], [w - 1, h - 1]]])
    cv2.fillPoly(mask, pts, (255, 255, 255))
    return mask


def white_index(img, roi, centers):
    temp = np.empty_like(img)

    for i in range(3):
        temp[:, :, i] = roi.astype(bool) * img[:, :, i]

    ind0 = np.argwhere(centers[:, 0] == np.max(temp[:, :, 0]))
    ind1 = np.argwhere(centers[:, 1] == np.max(temp[:, :, 1]))
    ind2 = np.argwhere(centers[:, 2] == np.max(temp[:, :, 2]))
    if (ind0 == ind1).any():
        index = ind0[np.argwhere(ind0 == ind1)[0, 0]]
    elif (ind0 == ind2).any():
        index = ind0[np.argwhere(ind0 == ind2)[0, 0]]
    elif (ind1 == ind2).any():
        index = ind1[np.argwhere(ind1 == ind2)[0, 0]]
    return centers[index]


def kmean(img):
    h, w = img.shape[:2]
    img = np.array(img, dtype=np.float64) / 255
    image_array = np.reshape(img, (w * h, 3))
    image_array_sample = np.random.permutation(image_array)[:1000]
    n_colors = 4
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    quant = kmeans.cluster_centers_[labels]
    final = quant.reshape((h, w, 3))
    return (255 * final).astype(np.uint8), (255 * kmeans.cluster_centers_).astype(np.uint8)


def find_edges(original, area_of_interest, first_frame=False):
    if first_frame == True:
        filter_img = edgefilter(original, 10, 1)
        reduced, centers = reducecolors(filter_img)
        colour = white_index(reduced, area_of_interest, centers)
        white = cv2.inRange(reduced, colour - 1, colour + 1)

        canny = cv2.Canny(filter_img, 85, 255, True)
        color_edges = cv2.bitwise_or(white, canny)
        edges = cv2.bitwise_and(area_of_interest, color_edges)
    else:
        if type(area_of_interest) == tuple:
            area_of_interest = cv2.bitwise_or(area_of_interest[0], area_of_interest[1])
        km_image, km_centers = kmean(original)
        colour = white_index(km_image, area_of_interest, km_centers)
        white = cv2.inRange(km_image, colour - 1, colour + 1)

        canny = cv2.Canny(original, 85, 255, True)
        color_edges = cv2.bitwise_or(white, canny)
        edges = cv2.bitwise_and(color_edges, area_of_interest)
    return edges


def make_coordinates(points, heights):
    if points.shape[0] != 0:
        z = np.polynomial.polynomial.polyfit(points[:, 1], points[:, 0], 1)
        points = np.empty((heights.shape[0], 2))
        points[:, 0] = np.polynomial.polynomial.polyval(heights, z, 1)
        points[:, 1] = heights
        return points.astype(int), z[1]
    else:
        return 0, 0


def find_lines(img_edges, heights):
    left_fit = []
    right_fit = []
    unknown = []
    line_width = False
    c_line = -1
    dynamic_offset = int(0.1*desired_width.get())
    for i in heights:
        out = np.argwhere(img_edges[i, :] == 255)
        if out.shape[0] != 0:
            if i == heights[-1]:
                if out[-1] - out[0] < dynamic_offset:
                    line_width = int(out[-1] - out[0])
                else:
                    data1 = out[out<out[0]+dynamic_offset]
                    data2 = out[out>out[-1]-dynamic_offset]
                    line_width = int((data1[-1]-data1[0] + data2[-1]-data2[0])/2)
            mx = np.max(out)
            mn = np.min(out)
            if mx - mn < 125:
                unknown.append((np.average(out), i))
            else:
                left_fit.append((np.average(out[out < mn + dynamic_offset]), i))
                right_fit.append((np.average(out[out > mx - dynamic_offset]), i))
        else:
            continue

    right_len = len(right_fit)
    left_len = len(left_fit)
    unknown_len = len(unknown)
    set1 = -1
    set2 = -1
    if right_len <= 2 and left_len <= 2:
        if unknown_len >= 2:
            median = int(np.median(unknown, axis=0)[0])
            unknown_array = np.asarray(unknown)
            tmp1 = unknown_array[:, 0] < median + 100
            tmp2 = unknown_array[:, 0] > median - 100
            ind = np.argwhere((tmp1 * tmp2) == True)
            unknown_array = unknown_array[ind].reshape((ind.shape[0], 2))
            set1 = make_coordinates(unknown_array, heights)
    else:
        if len(unknown) != 0:
            unknown_average = np.average(unknown, axis=0)[0]
            if abs(unknown_average - np.average(right_fit, axis=0)[0]) < abs(
                    unknown_average - np.average(left_fit, axis=0)[0]):
                right_fit = right_fit + unknown
                right_fit = sorted(right_fit, key=itemgetter(1))
            else:
                left_fit = left_fit + unknown
                left_fit = sorted(left_fit, key=itemgetter(1))

        if len(right_fit) >= len(left_fit):
            c_line = 'right'
        else:
            c_line = 'left'
        leftpoints = np.asarray(left_fit)
        rightpoints = np.asarray(right_fit)
        leftpoints = leftpoints[leftpoints[:, 0] < rightpoints[0, 0]]
        rightpoints = rightpoints[rightpoints[:, 0] > leftpoints[0, 0]]
        set1 = make_coordinates(leftpoints, heights)
        set2 = make_coordinates(rightpoints, heights)
    line_info = (c_line, line_width)
    return set1, set2, line_info

def checker(output, setLeft, setRight):
    h, w = output.shape[:2]
    nextRoi = False
    drawLeft = False
    drawRight = False
    drawMiddle = False
    if type(setLeft) == tuple and type(setRight) == tuple:
        leftPoints, leftSlope = setLeft
        rightPoints, rightSlope = setRight
        if leftSlope <= -0.25:
            drawLeft = True
            cv2.polylines(output, [leftPoints], False, (0, 0, 0), 3)
        if rightSlope >= 0.25:
            drawRight = True
            cv2.polylines(output, [rightPoints], False, (0, 0, 0), 3)
        if drawLeft and drawRight:
            offset1 = 5
            offset2 = int(0.05*desired_width.get())
            ll_pts = []
            lr_pts = []
            rl_pts = []
            rr_pts = []
            if leftPoints[0, 0] - offset1 >= 0:
                ll_pts.append((leftPoints[0, 0] - offset1, leftPoints[0, 1]))
            else:
                ll_pts.append((0, leftPoints[0, 1]))
            lr_pts.append((leftPoints[0, 0] + offset1, leftPoints[0, 1]))
            if leftPoints[-1, 0] - offset2 >= 0:
                ll_pts.append((leftPoints[-1, 0] - offset2, leftPoints[-1, 1]))
            else:
                ll_pts.append((0, leftPoints[-1, 1]))
            lr_pts.append((leftPoints[-1, 0] + offset2, leftPoints[-1, 1]))
            if rightPoints[0, 0] + offset1 <= w - 1:
                rr_pts.append((rightPoints[0][0] + offset1, rightPoints[0][1]))
            else:
                rr_pts.append((w - 1, rightPoints[0][1]))
            rl_pts.append((rightPoints[0, 0] - offset1, rightPoints[0][1]))
            if rightPoints[-1, 0] + offset2 <= w - 1:
                rr_pts.append((rightPoints[-1, 0] + offset2, rightPoints[-1, 1]))
            else:
                rr_pts.append((w - 1, rightPoints[-1, 1]))
            rl_pts.append((rightPoints[-1, 0] - offset2, rightPoints[-1, 1]))

            rr_pts.reverse()
            ra = rl_pts + rr_pts
            right_area_points = np.asarray(ra).astype(int)

            lr_pts.reverse()
            la = ll_pts + lr_pts
            left_area_points = np.asarray(la).astype(int)

            nextRoi = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(nextRoi, [left_area_points], 255)
            cv2.fillPoly(nextRoi, [right_area_points], 255)

    elif type(setLeft) == tuple and type(setRight) == int:
        drawMiddle = True
        cv2.polylines(output, [setLeft[0]], False, (0, 0, 0), 3)
    draw_set = (drawLeft, drawRight, drawMiddle)
    return draw_set, nextRoi


def calc_dist(LeftSet, RightSet, info_line, centre):
    c_line, line_w = info_line
    if type(LeftSet) == tuple and type(RightSet) == tuple:
        x2 = RightSet[0][-1, 0]
        x1 = LeftSet[0][-1, 0]
        # real width of line is approximately 12.5cm or 0.125m
        if line_w == 0:
            comm = 'Cannot calculate'
            dist = format(0, '.4f')
            Comment.set(comm)
            FinalDistance.set(dist)
            return dist, comm
        else:
            comm = 'Distance (m)'
            meters_per_pixel = 0.125 / line_w
            if c_line == 'right':
                diff = x2 - centre
                dist = format(diff*meters_per_pixel, '.4f')
            else:
                diff = (2*x2 - x1) - centre
                dist = format(diff*meters_per_pixel, '.4f')
            Comment.set(comm)
            FinalDistance.set(dist)
            return dist, comm
    else:
        dist = format(0, '.4f')
        comm = 'Changing lane'
        Comment.set(comm)
        FinalDistance.set(dist)
        return dist, comm


def video_func(frame, y, roi, const_roi, heights, cnt):
    final = np.copy(frame)
    (h, w) = frame.shape[:2]
    if type(roi) == bool:
        roi = const_roi
    if cnt == 0:
        edges = find_edges(frame[y:, :, :], roi, True)
    else:
        edges = find_edges(frame[y:, :, :], roi, False)

    set1, set2, width_line = find_lines(edges, heights)
    distance, comment = calc_dist(set1, set2, width_line, int(w/2))
    draw_tuple, roi = checker(frame[y:, :, :], set1, set2)
    return final, roi

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Lane Navigator')
        self.root.geometry('360x100+100+100')
        self.root.resizable(0, 0)

        # place open, exit and about buttons
        self.btn_open = tk.Button(self.root, text="Open file", command=self.openfile, fg='green', cursor='hand2', borderwidth=4)
        self.btn_open.place(width=120, height=60, x=40, y=20)
        self.btn_exit = tk.Button(self.root, text="Exit", command=self.myExit, fg='red', cursor='hand2', borderwidth=4)
        self.btn_exit.place(width=120, height=60, x=200, y=20)


        self.root.protocol('WM_DELETE_WINDOW', self.myExit)

    def myExit(self):
        self.myExit = tk.messagebox.askyesno("Confirm Exit", "Are you sure you want to exit Lane Navigator?")
        if self.myExit > 0:
            self.root.destroy()

    def donothing(self):
        filewin = tk.Toplevel(self.root)
        button = tk.Button(filewin, text="Do nothing button")
        button.pack()

    def openfile(self):
        self.video_path = tk.filedialog.askopenfilename(initialdir="/", title="Select file",
                                                        filetypes=(
                                                            ("Video files (mp4)", "*.mp4  "), ("all files", "*.*")))
        if self.video_path:
            new_window = tk.Toplevel(self.root)
            desired_height.set(int(4 * self.root.winfo_screenheight() / 5))
            desired_width.set(int(4 * self.root.winfo_screenwidth() / 5))
            new_window.geometry('%dx%d+0+0' % (desired_width.get(), desired_height.get() + 60))
            Player(new_window, self.video_path)


class Player:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title('Lane Navigator Player')

        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=desired_width.get(), height=desired_height.get())
        self.canvas.pack()

        # Create frame for the buttons
        self.l_frame1 = tk.LabelFrame(window, text='Controls')
        self.l_frame1.pack(side='left', expand=1, fill='x', padx=10, pady=2)
        # Button that lets the user take a snapshot
        self.btn_play = tk.Button(self.l_frame1, text="Play", command=self.vid.play, borderwidth=4)
        self.btn_play.pack(side='left', expand=1, fill='x', padx=10, pady=2)

        self.btn_pause = tk.Button(self.l_frame1, text="Pause", command=self.vid.pause, borderwidth=4)
        self.btn_pause.pack(side='left', expand=1, fill='x', padx=10, pady=2)

        self.btn_snapshot = tk.Button(self.l_frame1, text="Snapshot", command=self.snapshot, borderwidth=4)
        self.btn_snapshot.pack(side='left', expand=1, fill='x', padx=10, pady=2)

        self.btn_stop = tk.Button(self.l_frame1, text="Stop", command=self.window.destroy, borderwidth=4)
        self.btn_stop.pack(side='left', expand=1, fill='x', padx=10, pady=2)

        # Create frame for printing info
        l2_text = 'Distance from the right continuous line'
        self.l_frame2 = tk.LabelFrame(window, text=l2_text)
        self.l_frame2.pack(side='left', expand=1, fill='x', padx=10, pady=2)
        self.label1 = tk.Label(self.l_frame2, textvariable=Comment, borderwidth=0, relief="groove")
        self.label1.pack(side='left', expand=1, fill='x')
        self.label2 = tk.Label(self.l_frame2, textvariable=FinalDistance, bg='white', borderwidth=2, relief="groove")
        self.label2.pack(side='left', expand=1, fill='x', padx=10, pady=2)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def update(self):
        # Get a frame from the video source
        if not self.vid.pause_flag:
            ret, frame = self.vid.get_frame()

            if ret:
                fixed_img = PIL.Image.fromarray(frame).resize((desired_width.get(), desired_height.get()), PIL.Image.ANTIALIAS)
                self.photo = PIL.ImageTk.PhotoImage(image=fixed_img)
                self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = desired_width.get()
        self.height = desired_height.get()

        self.pause_flag = False

        self.y = int(0.68 * self.height)
        self.constant_roi = make_mask(self.height - self.y, self.width)
        self.heights = np.asarray([i for i in range(0, int(self.height) - int(self.y), 3)])
        self.cnt = 0
        self.roi = False

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR\
                frame = cv2.resize(frame, (desired_width.get(), desired_height.get()))
                final, self.roi = video_func(frame, self.y, self.roi, self.constant_roi, self.heights, self.cnt)
                self.cnt = 1
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def pause(self):
        if self.vid.isOpened():
            self.pause_flag = True

    def play(self):
        if self.vid.isOpened():
            self.pause_flag = False


root = tk.Tk()
desired_height = tk.IntVar()
desired_width = tk.IntVar()
Comment = tk.StringVar()
FinalDistance = tk.DoubleVar()
App(root)
root.mainloop()
