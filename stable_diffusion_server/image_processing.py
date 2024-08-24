from PIL import Image
import math

def aspect_ratio_upscale_and_crop(image: Image.Image, target_size: tuple) -> Image.Image:
    """
    Upscale the image while preserving aspect ratio, then crop to the target size.
    
    :param image: PIL Image object
    :param target_size: Tuple of (width, height) for the desired output size
    :return: PIL Image object with the desired size
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate the aspect ratios
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height
    
    # Determine the resize dimensions
    if original_aspect > target_aspect:
        # Image is wider than target, scale based on height
        new_height = target_height
        new_width = math.ceil(new_height * original_aspect)
    else:
        # Image is taller than target, scale based on width
        new_width = target_width
        new_height = math.ceil(new_width / original_aspect)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate cropping box
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image

def get_stable_diffusion_size(aspect_ratio: str) -> tuple:
    """
    Get the appropriate size for Stable Diffusion based on the aspect ratio.
    
    :param aspect_ratio: String representing the desired aspect ratio
    :return: Tuple of (width, height)
    """
    sizes = {
        "1:1": (1024, 1024),
        "3:2": (1152, 768),
        "2:3": (768, 1152),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "16:9": (1360, 768),
        "9:16": (768, 1360)
    }
    return sizes.get(aspect_ratio, (1024, 1024))  # Default to 1:1 if not found

def process_image_for_stable_diffusion_ar(image: Image.Image, aspect_ratio: str) -> Image.Image:
    """
    Process an image to make it compatible with Stable Diffusion.
    
    :param image: PIL Image object
    :param aspect_ratio: Desired aspect ratio as a string (e.g., "16:9")
    :return: Processed PIL Image object
    """
    target_size = get_stable_diffusion_size(aspect_ratio)
    return aspect_ratio_upscale_and_crop(image, target_size)

def get_closest_stable_diffusion_size(image_width: int, image_height: int) -> tuple:
    """
    Get the closest Stable Diffusion size based on the input image dimensions using aspect ratio comparison.
    
    :param image_width: Width of the input image
    :param image_height: Height of the input image
    :return: Tuple of (width, height) for the closest Stable Diffusion size
    """
    input_aspect_ratio = image_width / image_height
    
    stable_diffusion_sizes = {
        (1024, 1024): "1:1",
        (1152, 768): "3:2",
        (768, 1152): "2:3",
        (1152, 864): "4:3",
        (864, 1152): "3:4",
        (1360, 768): "16:9",
        (768, 1360): "9:16"
    }

    def calculate_aspect_ratio_difference(size):
        sd_aspect_ratio = size[0] / size[1]
        return abs(sd_aspect_ratio - input_aspect_ratio)

    closest_size = min(stable_diffusion_sizes.keys(), key=calculate_aspect_ratio_difference)
    return closest_size

def process_image_for_stable_diffusion(image: Image.Image) -> Image.Image:
    """
    Process an image to make it compatible with Stable Diffusion using the closest aspect ratio.
    
    :param image: PIL Image object
    :return: Processed PIL Image object
    """
    target_size = get_closest_stable_diffusion_size(image.width, image.height)
    return aspect_ratio_upscale_and_crop(image, target_size)
