import time
import threading
import matplotlib.pyplot as plt
import pynvml


def start_gpu_monitor(interval=0.2):
    log = []
    stop_event = threading.Event()

    def monitor():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W

            log.append((
                time.time(),
                util.gpu,
                mem.used / 1024**2,
                power,
                temp
            ))

            time.sleep(interval)

        pynvml.nvmlShutdown()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    return stop_event, log, thread


def plot_gpu_log(log):
    times = [x[0] - log[0][0] for x in log]
    util  = [x[1] for x in log]
    mem   = [x[2] for x in log]
    power = [x[3] for x in log]
    temp  = [x[4] for x in log]

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(times, util)
    axs[0].set_ylabel("GPU Util (%)")
    axs[0].grid(True)

    axs[1].plot(times, power)
    axs[1].set_ylabel("Power (W)")
    axs[1].grid(True)

    axs[2].plot(times, mem)
    axs[2].set_ylabel("VRAM (MB)")
    axs[2].grid(True)

    axs[3].plot(times, temp)
    axs[3].set_ylabel("Temp (°C)")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid(True)

    plt.suptitle("RTX 4050 GPU Load Profile")
    plt.tight_layout()
    plt.show()
