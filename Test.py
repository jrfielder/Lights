from gpiozero import PWMLED
from time import sleep

LED_PIN = 18  # GPIO pin connected to the LED

def main():
    led = PWMLED(LED_PIN)
    duty_cycle = 0.5  # Initial duty cycle (0-1)

    try:
        while True:
            # Change duty cycle
            duty_cycle += 0.05  # Increase duty cycle by 0.1
            if duty_cycle > 1.0:
                duty_cycle = 0.1  # Reset duty cycle if greater than 1
            led.value = duty_cycle
            print("Duty Cycle:", duty_cycle)
            sleep(0.1)

    except KeyboardInterrupt:
        led.close()  # Cleanup GPIO

if __name__ == "__main__":
    main()
