import os

from PIL import Image, ImageDraw, ImageFont


def create_japanese_character_image(
    character,
    font_path,
    font_size=200,
    bg_color="white",
    font_color="black",
    padding=10,
):
    """
    Creates an image with the given Japanese character, cropped to the character's
    exact bounding box (plus optional padding) to remove excess whitespace.

    Parameters:
        character (str): The Japanese character to render.
        font_path (str): Path to a TTF or TTC font file that supports Japanese characters.
        font_size (int): Font size to render the character.
        bg_color (str): Background color for the image.
        font_color (str): Color of the rendered text.
        padding (int): Number of pixels to add around the text after cropping.
        output_file (str): Output filename for the image.
    """
    # Create the output directory if it doesn't exist.
    os.makedirs("output", exist_ok=True)

    # Create a large enough canvas to draw the text.
    canvas_size = (font_size * 2, font_size * 2)
    image = Image.new("RGBA", canvas_size, bg_color)
    draw = ImageDraw.Draw(image)

    # Load the font.
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text at the top-left corner.
    draw.text((0, 0), character, fill=font_color, font=font)

    # Compute the bounding box of the text using textbbox.
    # Note: textbbox is available in Pillow 8.0.0 and above.
    bbox = draw.textbbox((0, 0), character, font=font)

    # Apply padding and ensure coordinates are non-negative.
    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = bbox[2] + padding
    lower = bbox[3] + padding

    # TODO: Save as a 512x512 image for consistency.

    # Crop the image to the bounding box.
    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(f"output/{character}.png")


if __name__ == "__main__":
    japanese_character = "日"  # Example Japanese character.
    # Replace with the path to your high-quality Japanese font.
    font_file = "data/font.ttf"
    create_japanese_character_image(japanese_character, font_file)
