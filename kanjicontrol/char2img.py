from PIL import Image, ImageDraw, ImageFont


def create_japanese_character_image(
    character,
    font_path,
    font_size=200,
    image_size=(300, 300),
    bg_color="white",
    font_color="black",
    output_file="output.png",
):
    """
    Creates an image with a given Japanese character rendered using a specified font.

    Parameters:
        character (str): The Japanese character to render.
        font_path (str): Path to a TTF or TTC font file supporting Japanese characters.
        font_size (int): The size of the font.
        image_size (tuple): The (width, height) of the output image.
        bg_color (str): Background color for the image.
        font_color (str): Color of the text.
        output_file (str): Path to save the generated image.
    """
    # Create a new image with the specified background color
    image = Image.new("RGBA", image_size, bg_color)
    draw = ImageDraw.Draw(image)

    # Load the font; ensure the font file supports Japanese characters.
    font = ImageFont.truetype(font_path, font_size)

    # Use getbbox to get the bounding box of the character.
    bbox = font.getbbox(character)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate the position to center the character.
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)

    # Draw the character onto the image.
    draw.text(position, character, fill=font_color, font=font)

    # Save the image.
    image.save(output_file)
    print(f"Image saved to {output_file}")


if __name__ == "__main__":
    # Example usage:
    japanese_character = "日"  # Example Japanese character.
    # Replace with the path to your font file.
    font_file = "data/font.ttf"
    create_japanese_character_image(japanese_character, font_file)
