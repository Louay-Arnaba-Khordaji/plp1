import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load your actual CSV file
data = np.genfromtxt("snapshots_8.csv", delimiter=",")

times = data[:, 0]
values = data[:, 1:]

fig, ax = plt.subplots()
line, = ax.plot(values[0])
ax.set_ylim(0, np.max(values) + 1)

def update(frame):
    line.set_ydata(values[frame])
    ax.set_title(f"Time step: {int(times[frame])}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(times), interval=200, repeat=False)
plt.show()