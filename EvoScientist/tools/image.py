"""Image viewing tool."""

import base64
import mimetypes
import os

from langchain_core.tools import tool

from ..paths import resolve_virtual_path

# Supported image extensions and their MIME types
_IMAGE_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

# Max file size for image viewing (5MB)
_MAX_IMAGE_SIZE = 5 * 1024 * 1024


@tool(parse_docstring=True)
def view_image(image_path: str) -> "list | str":
    """View and analyze an image file.

    Use this tool when you need to see the visual content of an image file
    (PNG, JPEG, GIF, WebP). The image will be displayed so you can describe,
    analyze, or answer questions about it.

    Note: Use this instead of read_file for image files. read_file only
    returns binary data, while view_image lets you actually see the image.

    Args:
        image_path: Path to the image file (relative to workspace or absolute)

    Returns:
        Image content blocks that the model can visually process
    """
    # Resolve virtual workspace paths: /image.png → {workspace}/image.png
    resolved = image_path
    if not os.path.isfile(resolved):
        resolved = str(resolve_virtual_path(image_path))

    if not os.path.isfile(resolved):
        return f"Error: File not found: {image_path}"
    image_path = resolved

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = _IMAGE_EXTENSIONS.get(ext)
    if not mime_type:
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            return f"Error: Not a supported image format: {ext}"

    file_size = os.path.getsize(image_path)
    if file_size > _MAX_IMAGE_SIZE:
        size_mb = file_size / (1024 * 1024)
        return f"Error: Image too large ({size_mb:.1f}MB). Max is 5MB."

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    size_kb = file_size / 1024
    filename = os.path.basename(image_path)

    return [
        {"type": "text", "text": f"Image: {filename} ({size_kb:.0f}KB, {mime_type})"},
        {"type": "image", "base64": data, "mime_type": mime_type},
    ]
