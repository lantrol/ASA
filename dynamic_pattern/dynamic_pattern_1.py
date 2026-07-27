import os
import math
from PIL import Image, ImageDraw

def create_animation_1(output_dir, num_frames=48, dot_size=4, radius=14):
    """
    A small dot moving in circles, only ever moves in the left 32 pixels of the image.
    """
    os.makedirs(output_dir, exist_ok=True)
    center_x = 16
    center_y = 32

    for i in range(num_frames):
        # Create a 64x64 grayscale image
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)

        # Calculate angle for current frame (0 to 2*pi)
        angle = 2 * math.pi * (i / num_frames)

        # Calculate dot position
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        # Draw the dot (using a bounding box for the circle)
        r = dot_size // 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        img.save(f"{output_dir}/frame_{i:02d}.png")

def create_animation_2(output_dir, num_frames=48, line_width=32, line_thickness=2):
    """
    A horizontal line that moves up and down. Only covers the 32 pixels from the right side.
    """
    os.makedirs(output_dir, exist_ok=True)
    center_y = 32

    for i in range(num_frames):
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)

        # Calculate vertical offset (oscillation)
        # Using sin to oscillate between roughly 0 and 64
        # Offset is shifted so it's centered around 32
        offset = 32 * math.sin(2 * math.pi * (i / num_frames))
        y = center_y + offset

        # Draw horizontal line on the right side (x from 32 to 63)
        # We'll use a rectangle to simulate a line with thickness
        draw.rectangle([32, y, 32 + line_width, y + line_thickness], fill=255)

        img.save(f"{output_dir}/frame_{i:02d}.png")

def create_animation_3(output_dir, num_frames=48, max_radius=30, min_radius=5, thickness=1):
    """
    A hollow circle (edge of circle) that shrinks and grows. Covers the full image.
    """
    os.makedirs(output_dir, exist_ok=True)
    center = (32, 32)

    for i in range(num_frames):
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)

        # Calculate current radius oscillating between min and max
        # Using sin to oscillate smoothly
        scale = 0.5 + 0.5 * math.sin(2 * math.pi * (i / num_frames))
        current_radius = min_radius + (max_radius - min_radius) * scale

        # Define the bounding box for the circle
        left_up = (center[0] - current_radius, center[1] - current_radius)
        right_down = (center[0] + current_radius, center[1] + current_radius)

        # Draw the hollow circle (outline)
        draw.ellipse([left_up, right_down], outline=255, width=thickness)

        img.save(f"{output_dir}/frame_{i:02d}.png")

if __name__ == "__main__":
    print("Generating Animation 1: Moving Dot...")
    create_animation_1("animation_1")

    print("Generating Animation 2: Moving Horizontal Line...")
    create_animation_2("animation_2")

    print("Generating Animation 3: Pulsing Hollow Circle...")
    create_animation_3("animation_3", thickness=3)

    print("Done! Images saved in 'animation_1/', 'animation_2/', and 'animation_3/' directories.")
