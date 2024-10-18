import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y)

# 计数器，用于控制动画结束
frame_count = 0
max_frames = 50

def update(frame):
    global frame_count
    if frame_count < max_frames:
        line.set_ydata(np.sin(x + frame / 10))  # 更新y数据
        frame_count += 1
    else:
        ani.event_source.stop()  # 停止动画
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

plt.show()
