"""Lower sustained duty cycle after a real temperature-guard stop; same caps."""
import time
import gpu_safety as safety


def initialize():
    safety.PAUSE=.15
    safety.initialize()
    while safety.telemetry[-1]['temperature_c']>65:
        print('Cooling before GPU work:',safety.telemetry[-1]['temperature_c'],'C',flush=True)
        time.sleep(5)
        safety.check(True)
