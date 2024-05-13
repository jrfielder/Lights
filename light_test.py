
## Unused
import time
from rpi_ws281x import PixelStrip, Color

# LED strip configuration:
LED_COUNT = 50        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 20  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create NeoPixel object
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
strip.begin()

# Define colors
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
YELLOW = Color(255, 255, 0)
CYAN = Color(0, 255, 255)
MAGENTA = Color(255, 0, 255)
WHITE = Color(255, 255, 255)


# Function to set color of all pixels
def set_color(color):
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()

# Main loop
try:
    while True:
        # Set color to red
        set_color(RED)
        time.sleep(1)

        # Set color to green
        set_color(GREEN)
        time.sleep(1)

        # Set color to blue
        set_color(BLUE)
        time.sleep(1)

except KeyboardInterrupt:
    # Clean up on CTRL+C
    strip.clear()
    strip.show()
