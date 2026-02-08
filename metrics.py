import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import pynvml as nvml


def start_gpu_monitor(interval=0.2):
    log = []
    stop_event = threading.Event()

    def monitor():
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)

        while not stop_event.is_set():
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            temp = nvml.nvmlDeviceGetTemperature(
                handle, nvml.NVML_TEMPERATURE_GPU
            )
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            log.append((
                time.time(),
                util.gpu,
                mem.used / 1024**2,
                power,
                temp
            ))
            time.sleep(interval)

        nvml.nvmlShutdown()

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event, log, t


def plot_gpu_log(log):
    t0 = log[0][0]
    t = [x[0] - t0 for x in log]
    util = [x[1] for x in log]
    mem = [x[2] for x in log]
    power = [x[3] for x in log]
    temp = [x[4] for x in log]

    fig, axs = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axs[0].plot(t, util, lw=2)
    axs[0].set_ylabel("GPU Util (%)")

    axs[1].plot(t, power, lw=2)
    axs[1].set_ylabel("Power (W)")

    axs[2].plot(t, mem, lw=2)
    axs[2].set_ylabel("VRAM (MB)")

    axs[3].plot(t, temp, lw=2)
    axs[3].set_ylabel("Temp (°C)")
    axs[3].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(alpha=0.3)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        "Tensor-Core Monte Carlo GPU Profile\n"
        f"Generated: {now}\n"
        "Aayushman Semwal",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()
